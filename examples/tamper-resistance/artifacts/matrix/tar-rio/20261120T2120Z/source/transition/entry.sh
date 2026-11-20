#!/usr/bin/env bash
set -euo pipefail
phase=${1:?eval, analysis, profiles, or train}
run=${2:?absolute study root}
name=${3:?case}
matrix_root=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
export TAR_RUNTIME_BACKEND=matrix
export TAR_TRACE_RUNNER="$matrix_root/source/misc/trace_runner.py"
case "$phase" in
    eval) exec bash -i "$matrix_root/source/misc/run_tar_compute_evaluation.sh" resume "$run" "$name" ;;
    analysis) exec bash -i "$matrix_root/source/misc/run_tar_compute_landscapes.sh" case "$run" "$name" ;;
    profiles) exec bash -i "$matrix_root/source/misc/run_tar_compute_landscapes.sh" profiles "$run" "$name" ;;
    train) exec bash -i "$matrix_root/source/misc/run_tar_compute.sh" case "$run" "$name" ;;
    *) exit 2 ;;
esac
