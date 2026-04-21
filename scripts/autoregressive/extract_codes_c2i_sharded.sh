#!/usr/bin/env bash
# Sharded code extraction from HF imagenet-1k parquet on dop-fuse.
# Run from repo root.

set -euo pipefail

DATA_PATH=${DATA_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/data}
CODE_PATH=${CODE_PATH:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/codes}
VQ_CKPT=${VQ_CKPT:-/jizhicfs/pkuhetu/bht/data/imagenet-1k/vq_ds16_c2i.pt}
NPROC=${NPROC:-8}
IMG=${IMG:-384}
BATCH=${BATCH:-32}
SHARD=${SHARD:-1024}
WORKERS=${WORKERS:-8}

# Ranks finish at different times (uneven parquet row counts); bump the NCCL
# watchdog timeout well above the worst-case straggler window. Unit: seconds.
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

mkdir -p "$CODE_PATH"

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29501} \
    autoregressive/train/extract_codes_c2i_sharded.py \
    --data-path "$DATA_PATH" \
    --code-path "$CODE_PATH" \
    --vq-ckpt   "$VQ_CKPT" \
    --image-size ${IMG} \
    --batch-size ${BATCH} \
    --shard-size ${SHARD} \
    --num-workers ${WORKERS}
