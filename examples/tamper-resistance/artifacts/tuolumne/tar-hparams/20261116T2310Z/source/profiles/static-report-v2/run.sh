#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=8
mkdir -p "$run/static/cache"
export MPLCONFIGDIR=$(mktemp -d "$run/static/cache/report-v2-matplotlib-XXXXXXXX")
exec python "$code/static_report.py" "$run"
