#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
bash -i "$run/source/jobs/evaluation-v2/misc/run_tar_compute_evaluation.sh" report "$run"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
export MPLBACKEND=Agg OMP_NUM_THREADS=8
export MPLCONFIGDIR="$run/cache/matplotlib-comparison"
exec 9>"$run/status/comparison-report.lock"
flock 9
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "$code_dir/misc/tar_compute_compare.py" report "$run"
