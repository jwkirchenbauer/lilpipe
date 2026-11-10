#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
bash -i "$run/source/jobs/analysis-v6/misc/run_tar_compute_landscapes.sh" profile-render-group "$run" bootstrap-102 bootstrap-103
bash -i "$run/source/jobs/analysis-v6/misc/run_tar_compute_landscapes.sh" profile-render-group "$run" deep-lr1e-5 deep-lr1e-6
bash -i "$run/source/jobs/analysis-v6/misc/run_tar_compute_landscapes.sh" profile-render-group "$run" deep-lr5e-6 noise-1e-4
