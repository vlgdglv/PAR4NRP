"""Sample class-conditional images from a trained RowAR model.

Row-wise generation loop: H steps, each produces W tokens in parallel.
Uses classifier-free guidance (cond + uncond in one batch), then VQ decode.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from autoregressive.models.rowar import RowAR_models
from tokenizer.tokenizer_image.vq_model import VQ_models


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0,
                      filter_value: float = -float("inf"), min_keep: int = 1):
    # logits: [N, V]
    if top_k > 0:
        k = min(max(top_k, min_keep), logits.size(-1))
        kth = torch.topk(logits, k)[0][..., -1, None]
        logits = torch.where(logits < kth, logits.new_full((), filter_value), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = probs > top_p
        if min_keep > 1:
            remove[..., :min_keep] = False
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        to_remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(to_remove, filter_value)
    return logits


def sample_tokens(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                  greedy: bool = False) -> torch.Tensor:
    # logits: [B, W, V]
    B, W, V = logits.shape
    flat = logits.reshape(B * W, V) / max(temperature, 1e-5)
    flat = top_k_top_p_filter(flat, top_k=top_k, top_p=top_p)
    if greedy:
        return flat.argmax(dim=-1).reshape(B, W)
    probs = F.softmax(flat, dim=-1)
    tok = torch.multinomial(probs, num_samples=1).reshape(B, W)
    return tok


def sample_one(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
               greedy: bool = False) -> torch.Tensor:
    """logits: [B, V] -> [B] long. Per-column sampler for the AR head."""
    flat = logits / max(temperature, 1e-5)
    flat = top_k_top_p_filter(flat, top_k=top_k, top_p=top_p)
    if greedy:
        return flat.argmax(dim=-1)
    probs = F.softmax(flat, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_rowar(model, class_idx: torch.Tensor, H: int, W: int,
                   cfg_scale: float = 1.0, temperature: float = 1.0,
                   top_k: int = 0, top_p: float = 1.0, greedy: bool = False):
    """Returns flat token grid [B, H*W] and per-step latency list.

    Two-stage per row:
      1. Trunk forward -> h_row [B, W, d_trunk] for this row (KV-cached).
      2. AR head walks W columns, each column sees (h_row, already-sampled tokens).
    With CFG, trunk is run on 2B (cond+uncond) and head is run twice per column.
    """
    B = class_idx.shape[0]
    device = class_idx.device
    prev_rows = None
    null_cls = torch.full_like(class_idx, model.config.num_classes)
    use_head = getattr(model, "use_head", False) and hasattr(model, "head")
    step_times = []

    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()

    for r in range(H):
        t0 = time.time()
        # ---- stage 1: trunk -> h_row ----
        if cfg_scale > 1.0:
            both_cls = torch.cat([class_idx, null_cls], dim=0)
            both_prev = None if prev_rows is None else torch.cat([prev_rows, prev_rows], dim=0)
            h_out = model(tokens=None, class_idx=both_cls, prev_rows=both_prev)
        else:
            h_out = model(tokens=None, class_idx=class_idx, prev_rows=prev_rows)

        # ---- stage 2: within-row AR ----
        if use_head:
            if cfg_scale > 1.0:
                h_c, h_u = h_out.chunk(2, dim=0)                         # [B, W, d_trunk] each
            else:
                h_c = h_out
            sampled = torch.empty(B, 0, dtype=torch.long, device=device)
            for c in range(W):
                if cfg_scale > 1.0:
                    l_c = model.head.step_logits(h_c, sampled)           # [B, V]
                    l_u = model.head.step_logits(h_u, sampled)
                    logits_c = l_u + cfg_scale * (l_c - l_u)
                else:
                    logits_c = model.head.step_logits(h_c, sampled)
                tok_c = sample_one(logits_c.float(), temperature, top_k, top_p, greedy=greedy)
                sampled = torch.cat([sampled, tok_c.unsqueeze(1)], dim=1)
            new_row = sampled
        else:
            # No head: legacy factorized sampling from trunk logits.
            if cfg_scale > 1.0:
                h_c, h_u = h_out.chunk(2, dim=0)
                l_c = model.output(h_c).float()
                l_u = model.output(h_u).float()
                logits = l_u + cfg_scale * (l_c - l_u)
            else:
                logits = model.output(h_out).float()
            new_row = sample_tokens(logits, temperature, top_k, top_p, greedy=greedy)

        prev_rows = new_row.unsqueeze(1) if prev_rows is None else torch.cat(
            [prev_rows, new_row.unsqueeze(1)], dim=1
        )
        torch.cuda.synchronize() if device.type == "cuda" else None
        step_times.append(time.time() - t0)

    codes = prev_rows.reshape(B, H * W).to(torch.long)
    return codes, step_times


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- VQ tokenizer ----
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device).eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq.load_state_dict(ckpt["model"])
    del ckpt
    print(f"[info] VQ loaded")

    # ---- RowAR ----
    latent_size = args.image_size // args.downsample_size
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    model = RowAR_models[args.model](
        vocab_size=args.codebook_size,
        num_classes=args.num_classes,
        grid_h=latent_size,
        grid_w=latent_size,
        cls_token_num=args.cls_token_num,
    ).to(device=device, dtype=precision).eval()
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    if "ema" in ckpt and args.use_ema:
        sd = ckpt["ema"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[info] RowAR loaded. missing={len(missing)} unexpected={len(unexpected)}")
    del ckpt

    # ---- sample ----
    if args.class_labels:
        class_labels = [int(c) for c in args.class_labels.split(",")]
    else:
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    cls = torch.tensor(class_labels, device=device)

    t0 = time.time()
    codes, step_times = generate_rowar(
        model, cls, H=latent_size, W=latent_size,
        cfg_scale=args.cfg_scale, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, greedy=args.greedy,
    )
    total = time.time() - t0
    print(f"[gen] {latent_size} rows in {total*1000:.1f} ms "
          f"(mean/step {1000*sum(step_times)/len(step_times):.1f} ms) "
          f"batch={len(class_labels)} cfg={args.cfg_scale}")
    print(f"[gen] sequential step count = {latent_size} (vs raster AR = {latent_size*latent_size})")

    # ---- decode ----
    qshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
    imgs = vq.decode_code(codes, qshape)
    save_image(imgs, args.out, nrow=min(4, len(class_labels)), normalize=True, value_range=(-1, 1))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, choices=list(RowAR_models.keys()), default="RowAR-B")
    p.add_argument("--gpt-ckpt", type=str, required=True)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--cls-token-num", type=int, default=1)
    p.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    # VQ
    p.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    p.add_argument("--vq-ckpt", type=str, required=True)
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    # layout
    p.add_argument("--image-size", type=int, choices=[256, 384], default=384)
    p.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    p.add_argument("--num-classes", type=int, default=1000)
    # sampling
    p.add_argument("--class-labels", type=str, default="",
                   help="comma-separated class ids; blank uses the PAR demo set")
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="sample_rowar.png")
    args = p.parse_args()
    main(args)
