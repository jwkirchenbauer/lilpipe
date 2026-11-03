#!/bin/bash
#SBATCH --job-name=z7b-layer-noise
#SBATCH --output=artifacts/mila/logs/layer_noise-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/analysis/layer-sensitivity/evaluate_noise.py \
  --output "artifacts/mila/evals/z7b-layer-noise-$SLURM_JOB_ID" \
  "$@"
