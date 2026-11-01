#!/bin/bash
#SBATCH --job-name=eval-beavertails-capabilities-trajectory
#SBATCH --output=artifacts/tamia/logs/eval_beavertails_capabilities_trajectory-%j.out
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

# Allow access to the W&B server.
module load httpproxy/1.0

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python scripts/evals/tamia_evaluation.py beavertails-trajectory "$@"

# Delete only when --delete-trajectory was supplied and all results verify.
python scripts/evals/tamia_evaluation.py beavertails-trajectory "$@" --cleanup-only
