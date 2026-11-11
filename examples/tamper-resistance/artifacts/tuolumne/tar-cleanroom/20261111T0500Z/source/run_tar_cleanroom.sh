#!/usr/bin/env bash
# One Flux task with four visible GPUs; invoke with bash -i.
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke with bash -i.' >&2; exit 2; }
mode=${1:?mode required}
run=${2:?absolute run directory required}
shift 2
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
worker_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 WANDB_MODE=offline AXOLOTL_DO_NOT_TRACK=1
export MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$run/logs" "$run/status" "$MPLCONFIGDIR"
python "$worker_dir/tar_cleanroom.py" "$mode" "$run" "$@"
