#!/bin/bash
#SBATCH --job-name=analysis-tar-archived-bs8-docs64
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

model=artifacts/tamia/models/archived-2026-11-01-5065b15a-z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16
outdir=artifacts/tamia/ll/tar-k16-archived-5065b15a-bs8-docs64-20261102
figure=artifacts/figures/z7b-wmdp-bio-loss-landscape-3d-tar-archived-5065b15a-zoomout-r125-bs8-docs64

# Claim a fresh output directory atomically; never reuse previous measurements.
mkdir -p "$(dirname "$outdir")"
mkdir "$outdir"
test ! -e "$figure.pdf"
test ! -e "$figure.png"

python scripts/analysis/loss-landscapes/get_ll.py \
  --model "$model" \
  --dataset cais/wmdp-bio-forget-corpus \
  --radii 1000 500 200 125 100 1e-1 1e-2 1e-3 \
  --n-pts 9 \
  --batch-size 8 \
  --limit 64 \
  --seed 42 \
  --outdir "$outdir" \
  "$@"

export MPLBACKEND=Agg
python scripts/analysis/loss-landscapes/plot_ll_3d.py \
  --indir "$outdir" --outpath "$figure.pdf"
python scripts/analysis/loss-landscapes/plot_ll_3d.py \
  --indir "$outdir" --outpath "$figure.png"
