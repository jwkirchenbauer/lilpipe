#!/usr/bin/env bash
# Run inside a one-GPU Matrix allocation with: bash -i this-script
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_CLUSTER_NAME:-}" ]] || {
    echo 'A real Matrix SLURM allocation is required.' >&2
    exit 2
}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
export TAR_RUNTIME_BACKEND=matrix TAR_EVAL_PROCESSES=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TORCH_BLAS_PREFER_CUBLASLT=0 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export WANDB_MODE=offline AXOLOTL_DO_NOT_TRACK=1
unset HIP_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
target=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
export PYTHONPATH="$target/source/eval-onegpu/misc:$target/source/eval-onegpu/src${PYTHONPATH:+:$PYTHONPATH}"
python "$target/source/transition/run_onegpu_calibration.py"
