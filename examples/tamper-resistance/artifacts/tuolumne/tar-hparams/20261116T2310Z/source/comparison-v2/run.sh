#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute hparam run directory required}
[[ "$run" == /* ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg OMP_NUM_THREADS=8
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$run/cache"
export MPLCONFIGDIR=$(mktemp -d "$run/cache/matplotlib-comparison-v2.XXXXXXXX")
exec 9>"$run/status/presentation-layout-v2.lock"
flock -n 9
status="$run/status/presentation-layout-v2.json"
stage=paired
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"stage\":\"%s\",\"exit_code\":%d}\n" "$stage" "$result" > "$status.tmp"; mv "$status.tmp" "$status"; fi' EXIT
printf '{"state":"running","stage":"paired"}\n' > "$status.tmp"
mv "$status.tmp" "$status"
python "$code/paired_layout.py" "$run"
stage=aggregate
printf '{"state":"running","stage":"aggregate"}\n' > "$status.tmp"
mv "$status.tmp" "$status"
python "$code/aggregate.py" "$run"
printf '{"state":"complete"}\n' > "$status.tmp"
mv "$status.tmp" "$status"
