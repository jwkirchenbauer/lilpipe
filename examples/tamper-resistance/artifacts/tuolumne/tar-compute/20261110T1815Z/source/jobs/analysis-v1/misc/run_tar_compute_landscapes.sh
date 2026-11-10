#!/usr/bin/env bash
# Two independent cases per node; each child sees one explicitly assigned GPU.
set -euo pipefail
mode=${1:?prepare, case, group, geometry, geometry-group, or render required}
run=${2:?absolute run directory required}
shift 2
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
worker=${BASH_SOURCE[0]}
code_dir=$(cd "$(dirname "$worker")" && pwd)
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$code_dir:$run/source/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=${TAR_ANALYSIS_GPU:-0} HIP_VISIBLE_DEVICES=${TAR_ANALYSIS_GPU:-0}
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
if [[ "$mode" == prepare ]]; then
    trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$run/status/probes.json"; fi' EXIT
    printf '{"state":"running"}\n' > "$run/status/probes.json"
    python "$code_dir/tar_compute.py" prepare-analysis "$run"
    printf '{"state":"complete"}\n' > "$run/status/probes.json"
elif [[ "$mode" == group || "$mode" == geometry-group ]]; then
    (( $# > 0 && $# <= 2 ))
    pids=()
    gpu=0
    child_mode=case
    if [[ "$mode" == geometry-group ]]; then child_mode=geometry; fi
    for selected in "$@"; do
        TAR_ANALYSIS_GPU=$gpu bash -i "$worker" "$child_mode" "$run" "$selected" > "$run/cases/$selected/logs/analysis-driver.log" 2>&1 &
        pids+=("$!")
        gpu=$((gpu + 1))
    done
    failures=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then failures=$((failures + 1)); fi
    done
    (( failures == 0 ))
elif [[ "$mode" == case || "$mode" == render || "$mode" == geometry ]]; then
    name=${1:?case required}
    folder="$run/cases/$name"
    export MPLCONFIGDIR="$folder/cache/matplotlib"
    mkdir -p "$MPLCONFIGDIR"
    stage=preflight
    trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"stage\":\"%s\",\"exit_code\":%d}\n" "$stage" "$result" > "$folder/status/analysis.json"; fi' EXIT
    python - "$folder" "$run" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
folder, run = map(Path, sys.argv[1:])
assert json.loads((folder / "status/train.json").read_text())["state"] == "complete"
probes = json.loads((run / "probes.json").read_text())
case = json.loads((folder / "case.json").read_text())
if case["bootstrap_seed"] is None:
    records = [json.loads(line) for line in (folder / "models/attack/observed-batches.jsonl").read_text().splitlines()]
    order = hashlib.sha256(json.dumps([r["batch_sha256"] for r in records]).encode()).hexdigest()
    assert order == probes["expected_nonbootstrap_order_sha256"], "Attack data order differs from clean control"
signature = hashlib.sha256((folder / "study.json").read_bytes()).hexdigest()
registry = folder / "status/analysis-input.json"
if registry.exists():
    assert json.loads(registry.read_text())["settings_sha256"] == signature, "Analysis settings changed; use a fresh output directory"
else:
    registry.write_text(json.dumps({"settings_sha256":signature}) + "\n")
assert torch.cuda.device_count() == 1
assert os.environ["CUDA_VISIBLE_DEVICES"] == os.environ["HIP_VISIBLE_DEVICES"]
print("Landscape GPU assignment:", {"physical_gpu_requested":os.environ["HIP_VISIBLE_DEVICES"], "logical_device":0, "visible_devices":1, "device":str(torch.cuda.get_device_properties(0))}, flush=True)
PY
    phases=(geometry surfaces random-frames report)
    if [[ "$mode" == geometry ]]; then phases=(geometry); fi
    if [[ "$mode" == render ]]; then
        for required in geometry surfaces random-frames; do
            rg -q '"state"[[:space:]]*:[[:space:]]*"complete"' "$folder/status/$required.json"
        done
        phases=(report)
    fi
    for stage in "${phases[@]}"; do
        if [[ "$mode" != render && -f "$folder/status/$stage.json" ]] && rg -q '"state"[[:space:]]*:[[:space:]]*"complete"' "$folder/status/$stage.json"; then
            echo "Reusing completed stage: $stage"
            continue
        fi
        printf '{"state":"running"}\n' > "$folder/status/$stage.json"
        python "$code_dir/tar_dynamics.py" "$stage" "$folder" 2>&1 | tee "$folder/logs/$stage.log"
        printf '{"state":"complete"}\n' > "$folder/status/$stage.json"
    done
    if [[ "$mode" == geometry ]]; then
        printf '{"state":"geometry_complete"}\n' > "$folder/status/analysis.json"
    else
        printf '{"state":"complete"}\n' > "$folder/status/analysis.json"
    fi
else
    echo "Unknown mode: $mode" >&2
    exit 2
fi
