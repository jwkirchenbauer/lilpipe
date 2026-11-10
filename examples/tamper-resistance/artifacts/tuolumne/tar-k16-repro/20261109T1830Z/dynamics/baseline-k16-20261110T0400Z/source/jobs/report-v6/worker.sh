#!/usr/bin/env bash
set -euo pipefail
study=${1:?study directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg MPLCONFIGDIR="$study/cache/matplotlib"
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
trap 'result=$?; if (( result != 0 )); then printf "{\"stage\":\"report-v4\",\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$study/status/report.json"; fi' EXIT
for stage in surfaces random-frames early-evaluations; do
    rg -q '"state":"complete"' "$study/status/$stage.json"
done
if [[ ! -d "$study/renderings-v1" ]]; then
    mkdir "$study/renderings-v1"
    for plot in "$study"/*.png "$study"/*.pdf "$study"/*.gif; do
        [[ -f "$plot" ]] || continue
        [[ "${plot##*/}" == random-* ]] && continue
        cp -p "$plot" "$study/renderings-v1/"
    done
fi
python "$study/source/jobs/report-v6/tar_dynamics.py" report "$study" 2>&1 | tee "$study/logs/report.log"
printf '{"stage":"report-v4","state":"complete"}\n' > "$study/status/report.json"
