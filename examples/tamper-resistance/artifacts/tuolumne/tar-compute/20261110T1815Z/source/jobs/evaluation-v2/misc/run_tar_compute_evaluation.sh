#!/usr/bin/env bash
# Submit with launch_tuo.py: one outer task, four GPUs, bash -i THIS evaluate RUN ...
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }
mode=${1:?evaluate or report required}
run=${2:?absolute study directory required}
[[ "$run" == /* ]]
[[ "$mode" == evaluate || "$mode" == report ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
worker_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# A frozen worker lives in source/misc; the editable worker lives in scripts/misc.
if [[ -d "$worker_dir/../src" ]]; then
    code_root=$(cd "$worker_dir/.." && pwd)
else
    code_root=$(cd "$worker_dir/../.." && pwd)
fi
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export PYTHONPATH="$worker_dir:$code_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export WANDB_MODE=offline AXOLOTL_DO_NOT_TRACK=1
# The outer Flux task's rank shim must not become an Accelerate worker rank.
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
python "$worker_dir/tar_compute_evaluation.py" "$@"
