#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ir=$(dirname "$(dirname "$profile")")
export PYTHONPATH="$ir/source/reference-local-v5:$ir/source/misc:$ir/source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
numactl --interleave=all python "$profile/bench.py" --output "$ir/performance-bench-v1/${FLUX_JOB_ID}.json"
