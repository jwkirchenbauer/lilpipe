#!/bin/bash
# Create artifacts/mila/logs before submitting.
#   mkdir -p artifacts/mila/logs
#   sbatch scripts/slurm/mila/analysis_get_param_norm.sbatch [--model MODEL]
#SBATCH --job-name=analysis-get-param-norm
#SBATCH --output=artifacts/mila/logs/analysis_get_param_norm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00

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

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec python scripts/analysis/parameter-norms/get_param_norm.py "$@"
