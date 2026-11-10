#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
bash -i "$run/source/jobs/evaluation-v2/misc/run_tar_compute_evaluation.sh" report "$run"
bash -i "$run/source/jobs/analysis-v3/misc/run_tar_compute_landscapes.sh" compare "$run"
