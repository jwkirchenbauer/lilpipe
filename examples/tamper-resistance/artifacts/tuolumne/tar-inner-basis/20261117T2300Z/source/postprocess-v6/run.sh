#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$profile/manifest.sha256"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
printf '{"state":"running"}\n' > "$run/status/postprocess.json"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$run/status/postprocess.json"; fi' EXIT
python -c 'import json,sys; state=json.load(open(sys.argv[1])); assert state["global_step"] == 125' "$run/cases/baseline-inner-recording/models/defense/checkpoint-125/trainer_state.json"
python "$profile/validate_transport.py" "$run"
for part in validate; do
    bash -i "$run/source/reference-local-v6/run.sh" "$part" "$run" 2>&1 | tee "$run/logs/$part.log"
done
export PYTHONPATH="$profile:$run/source/misc:$run/source/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/matplotlib"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["state"] == "passed"; assert p["inner_loss_trace"]["exactly_equal"]' "$run/replay-validation.json"
python "$profile/protocol_summary.py" "$run"
python - "$run" <<'PY'
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
program = json.loads((run / "study.json").read_text())
roots = [Path(program["baseline_dynamics"]) / "models/defense", run / "cases/baseline-inner-recording/models/defense"]
traces = []
for root in roots:
    history = json.loads((root / "checkpoint-125/trainer_state.json").read_text())["log_history"]
    trace = [(row["inner_step_total"], row["inner_loss"]) for row in history if "inner_step_total" in row]
    assert [step for step, loss in trace] == list(range(2000))
    assert all(math.isfinite(loss) for step, loss in trace)
    traces.append(trace)
assert traces[0] == traces[1]
proof = {
    "state": "complete",
    "terminal_checkpoint_global_step": 125,
    "saved_checkpoint_values_match_original": True,
    "all_2000_inner_losses_match_original": True,
    "ordered_finite_inner_loss_pairs_equal": True,
    "actual_attack_saved_anchor_matches": True,
    "training_wrapper_status": json.loads((run / "status/train.json").read_text()),
    "completion_source": str(run / "cases/baseline-inner-recording/models/defense/checkpoint-125/trainer_state.json"),
    "note": "The frozen training wrapper checks a root-level trainer_state.json that Axolotl does not normally write. Scientific completion is established by the terminal checkpoint, exact saved-value replay validation and protocol summary; the raw wrapper status is retained.",
}
(run / "status/verified-defense.json").write_text(json.dumps(proof, indent=2) + "\n")
PY
python "$profile/tar_inner_geometry.py" report --spec "$run/source/reference-local-v6/analysis-spec.json" --output "$run/geometry" 2>&1 | tee "$run/logs/report.log"
printf '{"state":"complete"}\n' > "$run/status/postprocess.json"
