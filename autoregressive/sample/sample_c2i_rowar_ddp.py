"""DDP sampler for RowAR. Generates N images and packs them into .npz for
OpenAI's evaluator.py (FID/sFID/IS/Precision/Recall).

Based on autoregressive/sample/sample_c2i_ddp.py but:
- uses RowARTransformer + generate_rowar (row-parallel decoding)
- no PAR sub-block unpermutation step
- defaults tuned for quick 10K subset eval (low-variance relative comparisons)
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from autoregressive.models.rowar import RowAR_models
from autoregressive.sample.sample_c2i_rowar import generate_rowar
from tokenizer.tokenizer_image.vq_model import VQ_models


def create_npz_from_sample_folder(sample_dir, num):
    samples = []
    for i in tqdm(range(num), desc="Building .npz from samples"):
        img = Image.open(f"{sample_dir}/{i:06d}.png")
        samples.append(np.asarray(img).astype(np.uint8))
    samples = np.stack(samples)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz [shape={samples.shape}] to {npz_path}")
    return npz_path


def main(args):
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        print(f"world_size={dist.get_world_size()} seed0={args.global_seed}")

    # ---- VQ ----
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq.load_state_dict(ckpt["model"])
    del ckpt

    # ---- RowAR ----
    latent_size = args.image_size // args.downsample_size
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]

    # Auto-detect whether the ckpt contains a trained AR head, so we rebuild
    # the model with use_head=True / use_glat=False and don't silently drop
    # head.* weights on load. (Training toggles these via RowARArgs defaults;
    # inference must mirror the ckpt's actual architecture.)
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    if "ema" in ckpt and args.use_ema:
        sd = ckpt["ema"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    has_head_in_ckpt = any(k.startswith("head.") for k in sd.keys())
    if rank == 0:
        print(f"[info] ckpt has AR head: {has_head_in_ckpt}")

    model = RowAR_models[args.gpt_model](
        vocab_size=args.codebook_size,
        num_classes=args.num_classes,
        grid_h=latent_size,
        grid_w=latent_size,
        cls_token_num=args.cls_token_num,
        use_head=has_head_in_ckpt,
        use_glat=False,  # GLAT is training-time only
    ).to(device=device, dtype=precision).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if rank == 0:
        print(f"loaded ckpt. missing={len(missing)} unexpected={len(unexpected)}")
        if unexpected:
            print(f"[warn] unexpected keys (first 5): {unexpected[:5]}")
        if missing:
            print(f"[warn] missing keys (first 5): {missing[:5]}")
    del ckpt

    # ---- output dir ----
    ckpt_name = os.path.basename(args.gpt_ckpt).replace(".pt", "").replace(".pth", "")
    ema_tag = "ema" if args.use_ema else "raw"
    folder_name = (f"{args.gpt_model}-{ckpt_name}-{ema_tag}-size{args.image_size}"
                   f"-eval{args.image_size_eval}-cfg{args.cfg_scale}"
                   f"-topk{args.top_k}-topp{args.top_p}-t{args.temperature}"
                   f"-seed{args.global_seed}-n{args.num_fid_samples}")
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # ---- scheduling ----
    n = args.per_proc_batch_size
    world = dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / (n * world)) * n * world)
    if rank == 0:
        print(f"total samples to generate: {total_samples}")
    per_gpu = total_samples // world
    iterations = per_gpu // n

    pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
    total = 0
    for _ in pbar:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        codes, _ = generate_rowar(
            model, c_indices, H=latent_size, W=latent_size,
            cfg_scale=args.cfg_scale, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, greedy=False,
        )
        qshape = [n, args.codebook_embed_dim, latent_size, latent_size]
        imgs = vq.decode_code(codes, qshape)  # [-1, 1]
        if args.image_size_eval != args.image_size:
            imgs = F.interpolate(imgs, size=(args.image_size_eval, args.image_size_eval),
                                 mode="bicubic")
        imgs = torch.clamp(127.5 * imgs + 128.0, 0, 255)
        imgs = imgs.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        for i, img in enumerate(imgs):
            idx = i * world + rank + total
            Image.fromarray(img).save(f"{sample_folder_dir}/{idx:06d}.png")
        total += n * world

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpt-model", type=str, choices=list(RowAR_models.keys()), default="RowAR-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--cls-token-num", type=int, default=1)
    p.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    p.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    p.add_argument("--vq-ckpt", type=str, required=True)
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    p.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    p.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--cfg-scale", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--sample-dir", type=str, default="samples")
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--num-fid-samples", type=int, default=10000,
                   help="10K is PAR's quick-variance setting; use 50K for final numbers")
    p.add_argument("--global-seed", type=int, default=0)
    args = p.parse_args()
    main(args)
