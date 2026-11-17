#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
mode=${1:?check, seal, or verify required}
hr=${2:?absolute main study directory required}
fr=${3:?absolute factorial study directory required}
[[ "$hr" == /* && "$fr" == /* ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
if [[ "$mode" == seal ]]; then
    exec 9>"$hr/status/curation-seal.lock"
    flock -n 9
fi
exec python "$code/seal.py" "$mode" "$hr" "$fr"
