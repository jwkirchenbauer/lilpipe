#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$profile/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
export PYTHONDONTWRITEBYTECODE=1
python -c 'import json,sys; assert json.load(open(sys.argv[1]))["state"] == "complete"' "$run/status/train.json"
for part in validate report; do
    bash -i "$run/source/run.sh" "$part" "$run" 2>&1 | tee "$run/logs/$part.log"
done
printf '{"state":"complete"}\n' > "$run/status/postprocess.json"
