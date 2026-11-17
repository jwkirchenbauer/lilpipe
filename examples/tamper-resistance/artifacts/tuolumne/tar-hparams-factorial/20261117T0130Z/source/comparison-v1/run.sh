#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute factorial study required}
[[ "$run" == /* ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=8
mkdir -p "$run/static/cache"
export MPLCONFIGDIR=$(mktemp -d "$run/static/cache/factorial-report-matplotlib-XXXXXXXX")
exec 9>"$run/status/static-factorial-report.lock"
flock -n 9
python "$code/factorial_static_report.py" validate "$run"
exec python "$code/factorial_static_report.py" report "$run"
