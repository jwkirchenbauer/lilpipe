#!/bin/bash
#SBATCH --job-name=plot-seam-relearning
#SBATCH --output=artifacts/tamia/logs/plot_seam_relearning-%j.out
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export MPLBACKEND=Agg

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
# Select the desired commented config/layout block in the plotter before submission.
python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py "$@"
