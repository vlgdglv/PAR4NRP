"""Quick sanity: parquet -> VQ encode -> decode -> save PNG.

Run from repo root:
  python tools/sanity_vq_roundtrip.py
"""
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models


PARQUET = "/home/vlgd/Data/imagenet-1k/data/train-00000-of-00294.parquet"
VQ_CKPT = "/home/vlgd/Models/LlamaGen/vq_ds16_c2i.pt"
IMG = 384
N = 4
OUT = "sanity_roundtrip.png"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- load a few images from parquet ----
    pf = pq.ParquetFile(PARQUET)
    batch = next(pf.iter_batches(batch_size=N, columns=["image", "label"]))
    rows = batch.to_pylist()
    print(f"loaded {len(rows)} rows; labels = {[r['label'] for r in rows]}")

    transform = transforms.Compose([
        transforms.Lambda(lambda im: center_crop_arr(im, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    imgs = []
    for r in rows:
        pil = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        imgs.append(transform(pil))
    x = torch.stack(imgs).to(device)
    print(f"x shape: {tuple(x.shape)}, range: [{x.min():.2f}, {x.max():.2f}]")

    # ---- load VQ ----
    vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device)
    vq.eval()
    ckpt = torch.load(VQ_CKPT, map_location="cpu")
    vq.load_state_dict(ckpt["model"])
    del ckpt
    print("VQ loaded")

    # ---- encode + decode ----
    with torch.no_grad():
        _, _, [_, _, indices] = vq.encode(x)
        H = IMG // 16
        print(f"indices shape: {tuple(indices.shape)}, dtype: {indices.dtype}, "
              f"min={int(indices.min())}, max={int(indices.max())}")
        codes = indices.reshape(N, H * H)
        recon = vq.decode_code(codes, [N, 8, H, H])
    print(f"recon shape: {tuple(recon.shape)}, range: [{recon.min():.2f}, {recon.max():.2f}]")

    # ---- save side-by-side ----
    panel = torch.cat([x.cpu(), recon.cpu()], dim=0)
    save_image(panel, OUT, nrow=N, normalize=True, value_range=(-1, 1))
    print(f"wrote {OUT} (top row: original, bottom row: reconstruction)")


if __name__ == "__main__":
    main()
