"""DDP code extractor for HF imagenet-1k parquet, sharded output.

Differences vs PAR's extract_codes_c2i.py:
  - Reads parquet directly (no JPEG unpack).
  - batch_size > 1 through the VQ encoder.
  - Writes ~1024 images per .npz shard (not 1 file per image), because
    dop-fuse / NFS-class FS hate 1.28M tiny files.

Output layout (matches what dataset/imagenet_sharded.py expects):
  $CODE_PATH/imagenet{IMG}_codes_sharded/shard_rank{R:03d}_{S:06d}.npz
    keys: codes uint16 [N, num_aug, H*W], labels int32 [N]
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.augmentation import center_crop_arr
from dataset.imagenet_parquet import ImageNetParquetIterable
from tokenizer.tokenizer_image.vq_model import VQ_models
from utils.distributed import init_distributed_mode


def main(args):
    assert torch.cuda.is_available()
    if not args.debug:
        init_distributed_mode(args)
        rank = dist.get_rank()
        world = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank, world, device = 0, 1, "cuda"

    out_dir = os.path.join(args.code_path, f"imagenet{args.image_size}_codes_sharded")
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if not args.debug:
        dist.barrier()

    # --- model ---
    vq = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
    ).to(device)
    vq.eval()
    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq.load_state_dict(ckpt["model"])
    del ckpt
    if rank == 0:
        print(f"[rank0] VQ tokenizer loaded from {args.vq_ckpt}")

    # --- data ---
    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageNetParquetIterable(
        data_path=args.data_path,
        transform=transform,
        rank=rank,
        world_size=world,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    H = args.image_size // 16
    tokens_per_img = H * H

    code_buf = []
    label_buf = []
    shard_idx = 0
    seen = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, code_buf, label_buf
        if not code_buf:
            return
        codes_np = np.concatenate(code_buf, axis=0).astype(np.uint16)  # [N, 2, H*H]
        labels_np = np.concatenate(label_buf, axis=0).astype(np.int32)  # [N]
        path = os.path.join(out_dir, f"shard_rank{rank:03d}_{shard_idx:06d}.npz")
        # uncompressed: faster read at training time
        np.savez(path, codes=codes_np, labels=labels_np)
        shard_idx += 1
        code_buf = []
        label_buf = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.numpy()
        x_flip = torch.flip(x, dims=[-1])
        x_all = torch.cat([x, x_flip], dim=0)  # [2B, C, H, W]
        with torch.no_grad():
            _, _, [_, _, indices] = vq.encode(x_all)
        # indices: [2B, H*W] flat int. Reshape to [B, 2, H*W] with [orig, flip] order.
        indices = indices.reshape(2, x.shape[0], tokens_per_img).permute(1, 0, 2).contiguous()
        code_buf.append(indices.detach().cpu().numpy())
        label_buf.append(y)
        seen += x.shape[0]

        if seen >= args.shard_size * (shard_idx + 1):
            flush()
            if rank == 0:
                rate = seen / max(time.time() - t0, 1e-6)
                print(f"[rank0] shards={shard_idx} imgs/rank={seen} rate={rate:.1f} img/s")

    flush()
    if rank == 0:
        print(f"[rank0] DONE rank0 wrote {shard_idx} shards, {seen} imgs in {time.time()-t0:.1f}s")
    if not args.debug:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True,
                   help="dir containing train-*.parquet")
    p.add_argument("--code-path", type=str, required=True)
    p.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    p.add_argument("--vq-ckpt", type=str, required=True)
    p.add_argument("--codebook-size", type=int, default=16384)
    p.add_argument("--codebook-embed-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=384)
    p.add_argument("--batch-size", type=int, default=32,
                   help="images per VQ forward; 2x after flip")
    p.add_argument("--shard-size", type=int, default=1024,
                   help="images per output shard file")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    main(args)
