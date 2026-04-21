#!/usr/bin/env bash
# Generate N samples from a RowAR ckpt and pack to .npz for OpenAI evaluator.py
# Default: 10K samples, CFG=2.0 — quick variance check per user's eval plan.
set -euo pipefail

GPT_CKPT=${GPT_CKPT:?set GPT_CKPT to a RowAR checkpoint}
VQ_CKPT=${VQ_CKPT:-/jizhicfs/pkuhetu/bht/model_home/LlamaGen/vq_ds16_c2i.pt}
MODEL=${MODEL:-RowAR-B}
IMG=${IMG:-256}
EVAL_IMG=${EVAL_IMG:-256}
NPROC=${NPROC:-8}
N=${N:-10000}
CFG=${CFG:-2.0}
TOPK=${TOPK:-0}
TOPP=${TOPP:-1.0}
TEMP=${TEMP:-1.0}
PER_GPU=${PER_GPU:-32}
SAMPLE_DIR=${SAMPLE_DIR:-/dockerdata/bht/LlamaGenNRP/samples_rowar}
USE_EMA=${USE_EMA:-1}

EMA_FLAG=""
if [[ "$USE_EMA" == "1" ]]; then EMA_FLAG="--use-ema"; fi

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29503} \
    autoregressive/sample/sample_c2i_rowar_ddp.py \
    --gpt-model "$MODEL" \
    --gpt-ckpt  "$GPT_CKPT" \
    --vq-ckpt   "$VQ_CKPT" \
    --image-size      ${IMG} \
    --image-size-eval ${EVAL_IMG} \
    --cfg-scale  ${CFG} \
    --top-k      ${TOPK} \
    --top-p      ${TOPP} \
    --temperature ${TEMP} \
    --per-proc-batch-size ${PER_GPU} \
    --num-fid-samples ${N} \
    --sample-dir "$SAMPLE_DIR" \
    ${EMA_FLAG}
