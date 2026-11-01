#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=artifacts/mila/logs/train-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
# (mila) uv on SLURM_TMPDIR to avoid file read issues on SCRATCH
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_LINK_MODE=copy
export AXOLOTL_DO_NOT_TRACK=1

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DIR="$PWD/artifacts/mila/logs"

config=${1:?missing training config}

uv sync --frozen --group train
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

axolotl train "$config" --launcher python
