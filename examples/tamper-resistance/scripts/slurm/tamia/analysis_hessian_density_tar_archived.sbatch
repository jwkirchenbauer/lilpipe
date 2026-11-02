#!/bin/bash
#SBATCH --job-name=analysis-pyhessian-tar
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

export MPLBACKEND=Agg
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

model=artifacts/tamia/models/archived-2026-11-01-5065b15a-z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16
runroot=artifacts/tamia/hessian/tar-k16-archived-5065b15a-slq10-p2-bs8-docs64-mb1
figure=artifacts/figures/z7b-wmdp-bio-hessian-density-tar-archived-5065b15a-slq10-p2-bs8-docs64-mb1
mkdir -p "$(dirname "$runroot")"
mkdir "$runroot"
test ! -e "$figure.pdf"
test ! -e "$figure.png"

# Verify second derivatives and GPU memory on one batch before the full subset.
python scripts/analysis/loss-landscapes/get_hessian_density.py \
  --model "$model" --batch-size 8 --micro-batch-size 1 --limit 8 --max-seq-len 512 \
  --lanczos-steps 2 --probes 1 --outdir "$runroot/smoke"

python scripts/analysis/loss-landscapes/get_hessian_density.py \
  --model "$model" --batch-size 8 --micro-batch-size 1 --limit 64 --max-seq-len 512 \
  --lanczos-steps 10 --probes 2 --seed 42 --outdir "$runroot/full"

cp --no-clobber "$runroot/full/spectral-density.pdf" "$figure.pdf"
cp --no-clobber "$runroot/full/spectral-density.png" "$figure.png"
