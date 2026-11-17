#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams-factorial/20261117T0130Z
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
(cd "$profile" && sha256sum --quiet -c manifest.sha256)
(cd "$run" && sha256sum --quiet -c operational-inputs.sha256)
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
python "$run/source/planning-v1/runtime_guard.py" training "$@"
exec bash -i "$run/source/misc/run_tar_compute.sh" "$@"
