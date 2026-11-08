#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=artifacts/tamia/logs/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

# Tamia compute nodes use cached packages, models, and datasets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export AXOLOTL_DO_NOT_TRACK=1

# Allow access to the W&B server.
module load httpproxy/1.0

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$PWD/artifacts/tamia/logs"

config=${1:?missing training config}
shift
merge=false
training_args=()
while (( $# )); do
    case $1 in
        --single-gpu) export CUDA_VISIBLE_DEVICES=0 ;;
        --merge) merge=true ;;
        *) training_args+=("$1") ;;
    esac
    shift
done
if [[ ! " ${training_args[*]} " =~ " --launcher " ]]; then
    training_args+=(--launcher python)
fi

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

# --launcher python preserves layer-sharded training across visible GPUs.
axolotl train "$config" "${training_args[@]}"
if [[ "$merge" == true ]]; then
    axolotl merge-lora "$config"
fi
