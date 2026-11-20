#!/usr/bin/env bash
set -euo pipefail
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
[[ ${SLURM_JOB_ID:-} ]]
if [[ ${SLURM_STEP_GPUS:-} == 0,1,2,3 ]]; then
    export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
fi
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE
exec python artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-assessment/benchmark_probe.py
