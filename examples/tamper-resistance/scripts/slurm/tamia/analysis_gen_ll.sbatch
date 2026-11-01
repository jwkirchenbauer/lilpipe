#!/bin/bash
#SBATCH --job-name=analysis-gen-ll
#SBATCH --output=artifacts/tamia/logs/analysis_gen_ll-%j.out
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

export CUDA_VISIBLE_DEVICES=0

model=artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16
attack=$model--relearn-fft-lr5e-6-ms128
trajectory=("$model" "$attack"/checkpoint-{16,32,48,64,80,96,112,128})

python scripts/analysis/loss-landscapes/get_ll.py \
  --model "$model" \
  --trajectory "${trajectory[@]}" \
  --dataset cais/wmdp-bio-forget-corpus \
  --radii 1e-1 1e-2 1e-3 \
  --n-pts 9 \
  --batch-size 2 \
  --seed 42 \
  --outdir artifacts/tamia/ll/tar-k16-relearn-ms128 \
  "$@"
