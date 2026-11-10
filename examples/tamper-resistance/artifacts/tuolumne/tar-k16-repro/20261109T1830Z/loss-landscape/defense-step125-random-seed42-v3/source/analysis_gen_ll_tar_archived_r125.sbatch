#!/bin/bash
#SBATCH --job-name=analysis-tar-archived-r125
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
outdir=artifacts/tamia/ll/tar-k16-archived-5065b15a-r125-20261102
figure=artifacts/figures/z7b-wmdp-bio-loss-landscape-3d-tar-archived-5065b15a-zoomout-r125

# Claim a fresh output directory atomically; never reuse previous measurements.
mkdir -p "$(dirname "$outdir")"
mkdir "$outdir"
test ! -e "$figure.pdf"
test ! -e "$figure.png"

python scripts/analysis/loss-landscapes/get_ll.py \
  --model "$model" \
  --dataset cais/wmdp-bio-forget-corpus \
  --radii 125 \
  --n-pts 9 \
  --batch-size 2 \
  --seed 42 \
  --outdir "$outdir" \
  "$@"

python - <<'PY_COMBINE'
import csv
from pathlib import Path
old = Path("artifacts/tamia/ll/tar-k16-archived-5065b15a-r1000-20261102/grid.csv")
root = Path("artifacts/tamia/ll/tar-k16-archived-5065b15a-r125-20261102")
new = root / "grid.csv"
with old.open() as f:
    existing = list(csv.DictReader(f))
with new.open() as f:
    added = list(csv.DictReader(f))
assert len(existing) == 567
assert len(added) == 81
new.rename(root / "grid-r125.csv")
with new.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["radius", "c0", "c1", "loss"])
    writer.writeheader()
    writer.writerows(existing + added)
PY_COMBINE

export MPLBACKEND=Agg
python scripts/analysis/loss-landscapes/plot_ll_3d.py \
  --indir "$outdir" --outpath "$figure.pdf"
python scripts/analysis/loss-landscapes/plot_ll_3d.py \
  --indir "$outdir" --outpath "$figure.png"
