#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run=${1:?absolute study directory required}
sha256sum --quiet -c "$profile/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/matplotlib"
export PYTHONPATH="$profile${PYTHONPATH:+:$PYTHONPATH}"
python "$profile/render.py" "$run"
