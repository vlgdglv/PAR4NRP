"""Train RowAR c2i on ImageNet VQ codes.

Mirrors autoregressive/train/train_c2i.py for fair comparison:
  - AdamW, lr=1e-4, betas=(0.9, 0.95), wd=5e-2, grad_clip=1.0
  - bf16 mixed precision
  - DDP across GPUs on a single node (use torchrun)
  - Optional EMA
  - Optional warm-start from a LlamaGen c2i checkpoint

Differences:
  - Reads sharded codes via dataset.imagenet_sharded.ShardedCodeDataset
  - No PAR sub-block permutation
  - Model = RowARTransformer (autoregressive/models/rowar.py)
"""
import argparse
import inspect
import math
import os
import time
from copy import deepcopy
from glob import glob
import wandb
import datetime
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from autoregressive.models.rowar import RowAR_models
from dataset.imagenet_sharded import ShardedCodeDataset, ShardedCodeDataseInRAM
from utils.distributed import init_distributed_mode
from utils.ema import requires_grad, update_ema
from utils.logger import create_logger

os.environ["WANDB_SETTINGS_DISABLE_STATS"] = "true"

def create_optimizer(model, weight_decay, lr, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay = [p for p in param_dict.values() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, **(dict(fused=True) if fused_ok else {}))
    logger.info(f"AdamW: decay params={sum(p.numel() for p in decay):,}, "
                f"nodecay params={sum(p.numel() for p in nodecay):,}, fused={fused_ok}")
    return opt


def main(args):
    assert torch.cuda.is_available()
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # ---- experiment dirs ----
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        ckpt_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = create_logger(args.results_dir)
        logger.info(f"args: {args}")

        if args.wandb_project is not None:
            wandb.init(
                project=args.wandb_project,
                name=os.path.basename(args.results_dir.strip("/")),
                dir=args.wandb_dir,
                config=vars(args)
            )
    else:
        logger = create_logger(None)

    # ---- model ----
    latent_size = args.image_size // args.downsample_size
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    model = RowAR_models[args.model](
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        grid_h=latent_size,
        grid_w=latent_size,
        cls_token_num=args.cls_token_num,
        token_dropout_p=args.token_dropout_p,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        class_dropout_prob=args.class_dropout_prob,
    ).to(device)
    logger.info(f"{args.model} params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ---- optional warm-start from LlamaGen ----
    if args.init_from_llamagen:
        ckpt = torch.load(args.init_from_llamagen, map_location="cpu")
        if "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        loaded, skipped = model.load_llamagen_state_dict(sd, verbose=(rank == 0))
        logger.info(f"[init] LlamaGen warm-start: loaded {len(loaded)} keys, skipped {len(skipped)}")
        del ckpt, sd

    # ---- EMA ----
    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
    else:
        ema = None

    optimizer = create_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    def lr_at(step, total):
        """Linear warmup -> cosine decay to lr_min_ratio*lr. Disabled if --no-lr-schedule."""
        if args.no_lr_schedule:
            return args.lr
        warm = max(1, args.warmup_steps)
        if step < warm:
            return args.lr * step / warm
        progress = (step - warm) / max(1, total - warm)
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.lr * (args.lr_min_ratio + (1.0 - args.lr_min_ratio) * cos)

    # ---- data ----
    dataset = ShardedCodeDataseInRAM(
        code_dir=os.path.join(args.code_path, f"imagenet{args.image_size}_codes_sharded")
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // dist.get_world_size(),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
    )
    logger.info(f"dataset: {len(dataset):,} images, {len(dataset.shard_files)} shards")

    # ---- resume ----
    train_steps = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        train_steps = ckpt.get("steps", 0)
        steps_per_epoch = max(len(dataset) // args.global_batch_size, 1)
        start_epoch = train_steps // steps_per_epoch
        logger.info(f"resume from {args.resume} at step {train_steps}, epoch {start_epoch}")
        del ckpt
    elif ema:
        update_ema(ema, model, decay=0)  # mirror initial weights

    if not args.no_compile:
        model = torch.compile(model)

    model = DDP(model.to(device), device_ids=[device])
    model.train()
    if ema:
        ema.eval()

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    running_loss, log_steps = 0.0, 0
    t_log = time.time()
    H = W = latent_size

    logger.info(f"training for {args.epochs} epochs, bs={args.global_batch_size}")
    total_steps = args.epochs * len(loader)

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for x, y in loader:
            # x: [B, num_aug, H*W] long, y: [B, 1] long
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).reshape(-1)
            # pick a random aug index per batch item
            B, num_aug, N = x.shape
            aug_idx = torch.randint(0, num_aug, (B,), device=device)
            tokens_flat = x[torch.arange(B, device=device), aug_idx]   # [B, H*W]
            tokens = tokens_flat.reshape(B, H, W)

            with torch.cuda.amp.autocast(dtype=ptdtype):
                _, loss = model(tokens=tokens, class_idx=y)

            cur_lr = lr_at(train_steps, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                target = model.module._orig_mod if not args.no_compile else model.module
                update_ema(ema, target)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                dt = time.time() - t_log
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                steps_per_sec = log_steps / dt
                time_per_step = dt / log_steps
                remaining_steps = total_steps - train_steps
                eta_seconds = int(max(0, remaining_steps) * time_per_step)
                eta_string = str(datetime.timedelta(seconds=eta_seconds))

                logger.info(f"step={train_steps:07d}/{total_steps} loss={avg_loss:.4f} "
                            f"lr={cur_lr:.2e} "
                            f"steps/s={steps_per_sec:.2f} epoch={epoch}/{args.epochs} "
                            f"eta={eta_string}")

                if rank == 0 and args.wandb_project is not None:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": cur_lr,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                        "train/eta_hours": eta_seconds / 3600.0,
                    }, step=train_steps)

                running_loss, log_steps, t_log = 0.0, 0, time.time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0 and rank == 0:
                target = model.module._orig_mod if not args.no_compile else model.module
                state = {
                    "model": target.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "steps": train_steps,
                    "args": vars(args),
                }
                if ema:
                    state["ema"] = ema.state_dict()
                path = os.path.join(args.results_dir, "checkpoints", f"{train_steps:07d}.pt")
                torch.save(state, path)
                logger.info(f"saved checkpoint to {path}")
            if train_steps % args.ckpt_every == 0:
                dist.barrier()

    logger.info("done")
    if rank == 0 and args.wandb_project is not None:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--code-path", type=str, required=True)
    p.add_argument("--image-size", type=int, choices=[256, 384], default=384)
    p.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--vocab-size", type=int, default=16384)
    # model
    p.add_argument("--model", type=str, choices=list(RowAR_models.keys()), default="RowAR-B")
    p.add_argument("--init-from-llamagen", type=str, default=None,
                   help="path to a LlamaGen c2i checkpoint to warm-start from")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--cls-token-num", type=int, default=1)
    p.add_argument("--class-dropout-prob", type=float, default=0.1)
    p.add_argument("--dropout-p", type=float, default=0.1)
    p.add_argument("--token-dropout-p", type=float, default=0.1)
    p.add_argument("--drop-path-rate", type=float, default=0.0)
    # optim
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="cosine floor as a fraction of peak lr")
    p.add_argument("--no-lr-schedule", action="store_true",
                   help="disable warmup+cosine and use constant --lr")
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--ema", action="store_true")
    # runtime
    p.add_argument("--results-dir", type=str, default="results_rowar")
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb_dir", type=str, default="/dockerdata/bht/LlamaGenNRP/wandb")
    args = p.parse_args()
    main(args)
