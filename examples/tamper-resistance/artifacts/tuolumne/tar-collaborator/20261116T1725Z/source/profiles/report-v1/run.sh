#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
run=${1:?absolute run directory required}
worker="$run/source/profiles/analysis-v2/run_tar_compute_landscapes.sh"
for horizon in 128 1000; do
    name="collaborator-lr1e-6-ms$horizon"
    folder="$run/cases/$name"
    for phase in train evaluation analysis step-profiles; do
        jq -e '.state=="complete"' "$folder/status/$phase.json" >/dev/null
    done
    bash -i "$worker" render "$run" "$name"
done
bash -i "$worker" checkpoint-compare "$run"
printf '{"state":"complete"}\n' > "$run/status/final-report.json"
