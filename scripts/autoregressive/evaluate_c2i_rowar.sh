#!/usr/bin/env bash
# Evaluation
set -euo pipefail

NPZ=${NPZ:-1}

python -m evaluations.c2i.evaluator \
       /jizhicfs/pkuhetu/bht/data/imagenet-1k/VIRTUAL_imagenet256_labeled.npz \
      inference_outputs/RowAR-B-0037500-ema-size256-eval256-cfg4.0-topk0-topp1.0-t1.0-seed0-n10000.npz