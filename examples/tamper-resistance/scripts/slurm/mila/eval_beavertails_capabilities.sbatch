#!/bin/bash
#SBATCH --job-name=eval-beavertails-capabilities
#SBATCH --output=artifacts/mila/logs/eval_beavertails_capabilities-%j.out
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
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_LINK_MODE=copy

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

model_id=${1:?missing model id}
base_model_name_or_path=${2:?missing model path}
method=${3:?missing method}
result_root=${4:?missing result root}

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/evals/evaluate_beavertail_and_capabilities.py single \
    "$model_id" "$base_model_name_or_path" "$method" "$result_root"
