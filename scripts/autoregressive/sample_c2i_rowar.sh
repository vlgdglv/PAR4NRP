#!/usr/bin/env bash
# Sample images from a trained RowAR checkpoint.
set -euo pipefail

GPT_CKPT=${GPT_CKPT:?set GPT_CKPT to a RowAR checkpoint}
VQ_CKPT=${VQ_CKPT:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/vq_ds16_c2i.pt}
MODEL=${MODEL:-RowAR-B}
OUT=${OUT:-sample_rowar.png}
CFG=${CFG:-4.0}
TOPK=${TOPK:-2000}

python autoregressive/sample/sample_c2i_rowar.py \
    --model "$MODEL" \
    --gpt-ckpt "$GPT_CKPT" \
    --vq-ckpt "$VQ_CKPT" \
    --image-size 256 \
    --cfg-scale ${CFG} \
    --top-k ${TOPK} \
    --temperature 1.0 \
    --out "$OUT"
