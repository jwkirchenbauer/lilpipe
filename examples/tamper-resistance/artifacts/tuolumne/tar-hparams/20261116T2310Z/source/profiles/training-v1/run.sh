#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$profile/manifest.sha256"
sha256sum --quiet -c /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/operational-inputs.sha256
export PYTHONDONTWRITEBYTECODE=1
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
python /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/source/planning-v1/runtime_guard.py training "$@"
exec bash -i /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/source/misc/run_tar_compute.sh "$@"
