#!/usr/bin/env bash
# Sample images from a trained RowAR checkpoint.
set -euo pipefail

GPT_CKPT=${GPT_CKPT:?set GPT_CKPT to a RowAR checkpoint}
VQ_CKPT=${VQ_CKPT:-/jizhicfs/pkuhetu/bht/model_home/LlamaGen/vq_ds16_c2i.pt}
MODEL=${MODEL:-RowAR-B}
OUT=${OUT:-sample_rowar.png}
CFG=${CFG:-4.0}
TOPK=${TOPK:-2000}

python -m autoregressive.sample.sample_c2i_rowar \
    --model "$MODEL" \
    --gpt-ckpt "$GPT_CKPT" \
    --vq-ckpt "$VQ_CKPT" \
    --image-size 256 \
    --cfg-scale ${CFG} \
    --top-k ${TOPK} \
    --temperature 1.0 \
    --out "$OUT"


# GPT_CKPT=/dockerdata/bht/LlamaGenNRP/rowar_b_256_fast/rowar-b-img256-bs2048-lr2e-04-ep60-wu1000-ema-llamagen/checkpoints/0035000.pt VQ_CKPT=/jizhicfs/pkuhetu/bht/model_home/LlamaGen/vq_ds16_c2i.pt MODEL=RowAR-B OUT=scratch_diag_10000.png CFG=4.0 bash scripts/autoregressive/sample_c2i_rowar.sh