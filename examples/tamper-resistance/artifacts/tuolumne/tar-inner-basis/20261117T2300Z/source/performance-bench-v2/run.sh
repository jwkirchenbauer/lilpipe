#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ir=$(dirname "$(dirname "$profile")")
export PYTHONPATH="$ir/source/performance-bench-v1:$ir/source/reference-local-v5:$ir/source/misc:$ir/source/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
ssd="/l/ssd/$USER"
mkdir -p "$ssd"
ram=${FLUX_JOB_TMPDIR:-${TMPDIR:-/tmp}}
if [[ $(findmnt -n -o FSTYPE -T "$ram") != tmpfs ]]; then ram=/tmp; fi
[[ $(findmnt -n -o FSTYPE -T "$ram") == tmpfs ]]
[[ $(findmnt -n -o FSTYPE -T "$ssd") != tmpfs ]]
numactl --interleave=all python "$profile/filebench.py" --ssd "$ssd" --ram "$ram" --output "$ir/performance-bench-v2/${FLUX_JOB_ID}.json"
