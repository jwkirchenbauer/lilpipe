#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
run=${1:?absolute run directory required}
name=${2:?case name required}
worker="$run/source/profiles/analysis-v1/run_tar_compute_landscapes.sh"
bash -i "$worker" case "$run" "$name"
bash -i "$worker" profiles "$run" "$name"
