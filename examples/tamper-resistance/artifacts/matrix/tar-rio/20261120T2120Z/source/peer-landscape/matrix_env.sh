#!/usr/bin/env bash
# Source from the interactive worker before any Torch imports.
# shellcheck shell=bash
[[ $- == *i* ]] || { echo 'Invoke the worker with bash -i.' >&2; exit 2; }
matrix_cluster=${SLURM_CLUSTER_NAME:-}
if [[ ! ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ || ${matrix_cluster,,} != matrix ||
      ! ${SLURM_STEP_ID:-} =~ ^[0-9]+$ ||
      ${SLURM_NTASKS:-} != 1 || ${SLURM_PROCID:-} != 0 || ${SLURM_LOCALID:-} != 0 ]]; then
    echo 'This worker requires a native Matrix Slurm step with one task.' >&2
    exit 2
fi

allocated_gpus=${SLURM_STEP_GPUS:-}
if [[ -n "$allocated_gpus" ]]; then
    if [[ ! $allocated_gpus =~ ^(0|[1-9][0-9]*),(0|[1-9][0-9]*)$ ]] ||
       [[ ${BASH_REMATCH[1]} == "${BASH_REMATCH[2]}" ]]; then
        echo "Expected two distinct SLURM_STEP_GPUS IDs, got '$allocated_gpus'." >&2
        exit 2
    fi
    export TAR_GPU_ASSIGNMENT_MODE=slurm_step_gpus
else
    export TAR_GPU_ASSIGNMENT_MODE=slurm_cgroup
fi

unset TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    export TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
if [[ ${CUDA_VISIBLE_DEVICES+x} != "${TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES+x}" ||
      ${CUDA_VISIBLE_DEVICES-} != "${TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES-}" ]]; then
    echo 'Environment activation changed the inherited CUDA mask.' >&2
    exit 2
fi
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
unset PYTHONOPTIMIZE
export TAR_RUNTIME_BACKEND=matrix TAR_LANDSCAPE_PEER=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 NVIDIA_TF32_OVERRIDE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MPLBACKEND=Agg

cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance || exit
