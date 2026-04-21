#!/usr/bin/env bash
# Train RowAR-B with LlamaGen warm-start on sharded ImageNet codes.
set -euo pipefail

CODE_PATH=${CODE_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes}
LLAMAGEN_CKPT=${LLAMAGEN_CKPT:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/c2i_B_384.pt}
RESULTS=${RESULTS:-/jizhicfs/pkuhetu/bht/results/rowar_b_init}
NPROC=${NPROC:-8}
BS=${BS:-256}
EPOCHS=${EPOCHS:-300}
LR=${LR:-1e-4}
MODEL=${MODEL:-RowAR-B}

mkdir -p "$RESULTS"

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29502} \
    autoregressive/train/train_c2i_rowar.py \
    --code-path "$CODE_PATH" \
    --image-size 384 \
    --model "$MODEL" \
    --init-from-llamagen "$LLAMAGEN_CKPT" \
    --global-batch-size ${BS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --ema \
    --results-dir "$RESULTS" \
    --num-workers 8 \
    --log-every 50 \
    --ckpt-every 5000
