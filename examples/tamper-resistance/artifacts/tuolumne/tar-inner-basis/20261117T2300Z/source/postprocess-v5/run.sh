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
python "$profile/validate_transport.py" "$run"
for part in validate; do
    bash -i "$run/source/reference-local-v5/run.sh" "$part" "$run" 2>&1 | tee "$run/logs/$part.log"
done
export PYTHONPATH="$profile:$run/source/misc:$run/source/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/matplotlib"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["state"] == "passed"; assert p["inner_loss_trace"]["exactly_equal"]' "$run/replay-validation.json"
python "$profile/protocol_summary.py" "$run"
python "$profile/tar_inner_geometry.py" report --spec "$run/source/reference-local-v5/analysis-spec.json" --output "$run/geometry" 2>&1 | tee "$run/logs/report.log"
printf '{"state":"complete"}\n' > "$run/status/postprocess.json"
