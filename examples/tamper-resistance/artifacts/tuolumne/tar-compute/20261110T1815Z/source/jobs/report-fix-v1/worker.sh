#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
bash -i "$run/source/jobs/comparison-v9/worker.sh" "$run"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
python - "$run" <<'PY'
import csv, json, pathlib, sys
r=pathlib.Path(sys.argv[1])
assert json.loads((r/"comparison-validation.json").read_text())["complete_scientific_cases"]==18
rows=list(csv.DictReader((r/"case-summary.csv").open()))
for row in rows:
    if row["role"]=="defense":
        assert row["defense_ce_source_case"]
        assert all(row[key] for key in ("fixed_ce_start","fixed_ce_final","heldout_ce_start","heldout_ce_final"))
table=[line for line in (r/"report.md").read_text().splitlines() if line.startswith("| ")]
assert not any("pending" in line for line in table)
for line in table:
    if "(defense;" in line:
        assert line.split("|")[6].strip()=="N/A"
        print(line)
PY
exec python "$run/source/jobs/finalize-v1/seal_artifacts.py"
