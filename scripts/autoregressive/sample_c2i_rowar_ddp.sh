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
CFG=${CFG:-4.0}
TOPK=${TOPK:-0}
TOPP=${TOPP:-1.0}
TEMP=${TEMP:-1.0}
PER_GPU=${PER_GPU:-32}
SAMPLE_DIR=${SAMPLE_DIR:-inference_outputs/}
USE_EMA=${USE_EMA:-1}

EMA_FLAG=""
if [[ "$USE_EMA" == "1" ]]; then EMA_FLAG="--use-ema"; fi

torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29503} \
    -m autoregressive.sample.sample_c2i_rowar_ddp \
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


# GPT_CKPT=/dockerdata/bht/LlamaGenNRP/rowar_b_256_fast/checkpoints/LATEST.pt N=10000 CFG=4.0 bash scripts/autoregressive/sample_c2i_rowar_ddp.sh


# for CFG in 1.5 2.0 3.0 4.0; do
#     GPT_CKPT=training_outputs/rowar_b_256_fast/checkpoints/0015000.pt N=10000 CFG=$CFG \
#     bash scripts/autoregressive/sample_c2i_rowar_ddp.sh
# done

# python evaluations/c2i/evaluator.py /jizhicfs/pkuhetu/bht/data/imagenet-1k/VIRTUAL_imagenet256_labeled.npz inference_outputs/RowAR-B-0015000-ema-size256-eval256-cfg4.0-topk0-topp1.0-t1.0-seed0-n10000.npz


# for CFG in 1.0 2.0 3.0 4.0; do
#     GPT_CKPT=/dockerdata/bht/LlamaGenNRP/rowar_b_256_tar/rowar-b-img256-bs2048-lr2e-04-ep25-wu1000-ema-llamagen/checkpoints/0015000.pt SAMPLE_DIR=inference_outputs/samples_25ep_b_tar_img256_cfg${CFG} CFG=${CFG} N=2500 bash scripts/autoregressive/sample_c2i_rowar_ddp.sh
# done


# for CFG in  1.0 2.0 3.0 4.0; do
#     python -m evaluations.c2i.evaluator /jizhicfs/pkuhetu/bht/data/imagenet-1k/VIRTUAL_imagenet256_labeled.npz \
#           inference_outputs/samples_25ep_b_tar_img256_cfg${CFG}/RowAR-XL-0015000-ema-size384-eval256-cfg${CFG}-topk0-topp1.0-t1.0-seed0-n2500.npz
# done

# # 

