#!/bin/bash
#SBATCH --job-name=parameter-norms
#SBATCH --output=artifacts/tamia/logs/parameter_norms-%j.out
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_OFFLINE=1
export UV_LINK_MODE=copy
export OMP_NUM_THREADS=8

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Submit to a CPU partition from examples/tamper-resistance after creating logs/.
uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
exec python scripts/analysis/parameter-norms/get_param_norm.py --device cpu "$@"
