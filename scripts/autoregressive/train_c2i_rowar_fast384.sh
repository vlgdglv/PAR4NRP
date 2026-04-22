#!/usr/bin/env bash
# Fast verification run: RowAR-B at 384, warm-started from LlamaGen-B-384.
# Same recipe as fast256 but scaled for the ~2.25x longer sequence (seq len 1153).
set -euo pipefail

CODE_PATH=${CODE_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes}
LLAMAGEN_CKPT=${LLAMAGEN_CKPT:-/jizhicfs/pkuhetu/bht/model_home/LlamaGen/c2i_B_384.pt}
RESULTS=${RESULTS:-/dockerdata/bht/LlamaGenNRP/rowar_b_384_fast}
NPROC=${NPROC:-8}
BS=${BS:-1024}            # ~half of 256's 2048 since seq len ~2.25x
EPOCHS=${EPOCHS:-60}
LR=${LR:-1.5e-4}          # sqrt(1024/512) * 1e-4 ~= 1.4e-4
WARMUP=${WARMUP:-1000}
MODEL=${MODEL:-RowAR-B}
RESUME=${RESUME:-}

mkdir -p "$RESULTS"

INIT_FLAG="--init-from-llamagen $LLAMAGEN_CKPT"
RESUME_FLAG=""
if [[ -n "$RESUME" ]]; then
    INIT_FLAG=""
    RESUME_FLAG="--resume $RESUME"
fi

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-26384} \
    autoregressive/train/train_c2i_rowar.py \
    --code-path "$CODE_PATH" \
    --image-size 384 \
    --model "$MODEL" \
    $INIT_FLAG \
    $RESUME_FLAG \
    --global-batch-size ${BS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup-steps ${WARMUP} \
    --lr-min-ratio 0.1 \
    --ema \
    --results-dir "$RESULTS" \
    --num-workers 8 \
    --log-every 50 \
    --ckpt-every 2500 \
    --wandb-project LlamaGenNRP
