#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study root required}
name=${2:?case required}
pilot=${3:?absolute pilot result.json required}
[[ "$run" == /* && "$pilot" == /* && "$name" != */* && "$name" != . && "$name" != .. ]]
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=matrix_env.sh
source "$source_dir/matrix_env.sh"
export PYTHONPATH="$source_dir:$source_dir/misc:$run/source/src${PYTHONPATH:+:$PYTHONPATH}"

folder="$run/cases/$name"
exec 9>"$folder/status/analysis.lock"
if ! flock -n 9; then
    echo "Analysis already running for $name" >&2
    exit 1
fi

python - "$folder" "$run" "$pilot" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from runtime import matrix_runtime, sha256, source_hashes
from tar_dynamics import dump

folder, run, pilot_path = map(Path, sys.argv[1:])
pilot = json.loads(pilot_path.read_text())
assert pilot["state"] == "complete" and all(pilot["gates"].values()), "Pilot did not pass"
sources = source_hashes()
assert pilot["runtime_source_sha256"] == sources, "Pilot tested different runtime sources"
runtime = matrix_runtime()
assert pilot["runtime"]["torch"] == runtime["torch"]

assert json.loads((folder / "status/train.json").read_text())["state"] == "complete"
probes = json.loads((run / "probes.json").read_text())
case = json.loads((folder / "case.json").read_text())
if case["bootstrap_seed"] is None:
    records = [
        json.loads(line)
        for line in (folder / "models/attack/observed-batches.jsonl").read_text().splitlines()
    ]
    order = hashlib.sha256(json.dumps([r["batch_sha256"] for r in records]).encode()).hexdigest()
    expected = probes.get("expected_order_sha256_by_horizon", {}).get(
        str(case["max_steps"]), probes["expected_nonbootstrap_order_sha256"]
    )
    assert order == expected, "Attack data order differs from clean control"

signature = sha256(folder / "study.json")
registry = folder / "status/analysis-input.json"
if registry.exists():
    assert json.loads(registry.read_text())["settings_sha256"] == signature, (
        "Analysis settings changed; use a fresh output directory"
    )
else:
    dump(registry, {"settings_sha256": signature})
dump(
    folder / "status" / f"analysis-peer-{os.environ['SLURM_JOB_ID']}.json",
    {
        "settings_sha256": signature,
        "pilot_result": str(pilot_path),
        "pilot_result_sha256": sha256(pilot_path),
        "runtime_source_sha256": sources,
        "runtime": runtime,
    },
)
print("Landscape GPU assignment:", runtime, flush=True)
PY

export MPLCONFIGDIR="$folder/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
stage=preflight
record_failure() {
    if (( $1 != 0 )); then
        printf '{"state":"failed","stage":"%s","exit_code":%d}\n' \
            "$stage" "$1" > "$folder/status/analysis.json"
    fi
}
trap 'record_failure "$?"' EXIT
printf '{"state":"running"}\n' > "$folder/status/analysis.json"
for stage in geometry surfaces random-frames report; do
    status="$folder/status/$stage.json"
    if [[ -f "$status" ]] && grep -Eq '"state"[[:space:]]*:[[:space:]]*"complete"' "$status"; then
        echo "Reusing completed stage: $stage"
        continue
    fi
    printf '{"state":"running"}\n' > "$status"
    /usr/bin/time -v python "$source_dir/../misc/trace_runner.py" \
        "$source_dir/misc/tar_dynamics.py" "$stage" "$folder" 2>&1 |
        tee "$folder/logs/$stage.log"
    printf '{"state":"complete"}\n' > "$status"
done
printf '{"state":"complete"}\n' > "$folder/status/analysis.json"
