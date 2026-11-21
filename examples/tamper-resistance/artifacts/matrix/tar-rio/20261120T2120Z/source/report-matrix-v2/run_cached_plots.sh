#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export MPLCONFIGDIR=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/cache/matplotlib-compact-titles
exec python "$(dirname "${BASH_SOURCE[0]}")/render_cached_plots.py"
