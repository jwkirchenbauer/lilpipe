#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
export PYTHONDONTWRITEBYTECODE=1
exec python "$run/source/jobs/finalize-v1/seal_artifacts.py"
