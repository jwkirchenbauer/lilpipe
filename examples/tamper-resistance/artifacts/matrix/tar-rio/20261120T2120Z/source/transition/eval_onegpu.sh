#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study root}
name=${2:?case}
root=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
export TAR_RUNTIME_BACKEND=matrix TAR_EVAL_PROCESSES=1
exec bash -i "$root/source/eval-onegpu/misc/run_tar_compute_evaluation.sh" resume "$run" "$name"
