#!/usr/bin/env bash
# Fast verification run: RowAR-B at 256, warm-started from LlamaGen-B-256.
# Target: clean loss curve + usable samples within ~9h on 8xH20.
set -euo pipefail

CODE_PATH=${CODE_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes}
LLAMAGEN_CKPT=${LLAMAGEN_CKPT:-/jizhicfs/pkuhetu/bht/model_home/LlamaGen/c2i_B_256.pt}
RESULTS=${RESULTS:-/dockerdata/bht/LlamaGenNRP/rowar_b_256_fast}
NPROC=${NPROC:-8}
BS=${BS:-2048}          # 80% VRAM on 8xH20 at 256 (seq len 513)
EPOCHS=${EPOCHS:-25}    # sqrt-scaled LR + cosine: 60 is enough for signal
LR=${LR:-2e-4}          # sqrt(BS/512) * 1e-4 = 2e-4 at BS=2048
WARMUP=${WARMUP:-1000}
MODEL=${MODEL:-RowAR-B}

mkdir -p "$RESULTS"

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-26312} \
    autoregressive/train/train_c2i_rowar.py \
    --code-path "$CODE_PATH" \
    --image-size 256 \
    --model "$MODEL" \
    --init-from-llamagen "$LLAMAGEN_CKPT" \
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
