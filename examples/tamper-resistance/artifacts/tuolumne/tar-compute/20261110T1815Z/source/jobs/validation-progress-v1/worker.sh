#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute study directory required}
[[ "$run" == /* ]]
bash -i "$run/source/jobs/comparison-v7/worker.sh" "$run"
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
python - "$run" <<'PY'
import json, pathlib, runpy, sys
r=pathlib.Path(sys.argv[1]);p=json.loads((r/'study.json').read_text())
ns=runpy.run_path(str(r/'source/jobs/finalize-v1/seal_artifacts.py'))
compare=ns['load_module'](r,p['comparison_report_source'],'progress_compare')
checked={}
for name,case in p['cases'].items():
    d=r/'cases'/name
    status=d/'status/analysis.json'
    if case['role']!='attack' or not status.exists() or json.loads(status.read_text()).get('state')!='complete':continue
    checked[name]=ns['validate_surfaces'](r,d,json.loads((d/'study.json').read_text()),compare)
print(json.dumps({'complete_cases_validated':len(checked),'surface_counts':checked},indent=2))
PY
