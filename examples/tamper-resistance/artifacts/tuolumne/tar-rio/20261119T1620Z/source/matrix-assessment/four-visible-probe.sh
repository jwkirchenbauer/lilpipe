#!/usr/bin/env bash
set -euo pipefail
mm_activate /usr/workspace/wsb/kirchenb/matrix-torch-cu126-demo
# The site sets a one-device mask even for an unbound four-GPU step.
# Widen visibility only after verifying that this step owns all four devices.
[[ ${SLURM_JOB_ID:-} && ${SLURM_STEP_GPUS:-} == 0,1,2,3 ]]
printf 'Original CUDA mask: %s; allocated step GPUs: %s\n' "${CUDA_VISIBLE_DEVICES:-}" "$SLURM_STEP_GPUS"
export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"
exec python /p/vast1/kirchenb/llnl-tools/matrix_cuda_probe.py --expected_visible_gpus=4 --collective
