#!/usr/bin/env bash
set -euo pipefail
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
run=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
export MPLCONFIGDIR="$run/source/matrix-assessment/matplotlib-checkpoint"
export PYTHONUNBUFFERED=1
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
exec python "$run/source/matrix-assessment/checkpoint_probe.py" \
    --control=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/cases/hparam-rio-lr1e-6 \
    --output="$run/matrix-assessment/checkpoint-probe" \
    --dynamics="$run/source/matrix-assessment/tar_dynamics_matrix.py"
