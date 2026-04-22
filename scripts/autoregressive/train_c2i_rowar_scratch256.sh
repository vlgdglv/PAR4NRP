#!/usr/bin/env bash
# From-scratch ablation: RowAR-B at 256, NO LlamaGen warm-start.
# Same hyperparams as fast256 so the comparison is apples-to-apples except
# for the warm-start source. If from-scratch train loss at epoch 12 beats
# the warm-started 8.0, we've confirmed warm-start is net-negative for this
# task and should drop it for all future runs.
set -euo pipefail

CODE_PATH=${CODE_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes}
RESULTS=${RESULTS:-/dockerdata/bht/LlamaGenNRP/rowar_b_256_scratch}
NPROC=${NPROC:-8}
BS=${BS:-2048}
EPOCHS=${EPOCHS:-25}
LR=${LR:-2e-4}
WARMUP=${WARMUP:-2000}     # longer warmup for cold start (was 1000 warm)
MODEL=${MODEL:-RowAR-B}

mkdir -p "$RESULTS"

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29504} \
    autoregressive/train/train_c2i_rowar.py \
    --code-path "$CODE_PATH" \
    --image-size 256 \
    --model "$MODEL" \
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
