#!/usr/bin/env bash
# Immutable copies run through launch_tuo.py; invoke with bash -i.
set -euo pipefail
mode=${1:?case or bundle required}
run=${2:?absolute study directory required}
name=${3:?case or bundle name required}
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
worker=${BASH_SOURCE[0]}
source_root=$(cd "$(dirname "$worker")/.." && pwd)
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$source_root/misc:$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 AXOLOTL_DO_NOT_TRACK=1
export WANDB_MODE=offline LILPIPE_DEVICE_DEBUG=1
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$run/logs" "$run/status"
if [[ "$mode" == verify ]]; then
    python "$source_root/misc/tar_compute.py" verify-checkpoint-comparison "$run"
elif [[ "$mode" == segment ]]; then
    folder="$run/cases/$name"
    stop_step=${4:?segment stop step required}
    resume_step=${5:-0}
    [[ "$stop_step" =~ ^[1-9][0-9]*$ && "$resume_step" =~ ^[0-9]+$ ]]
    (( resume_step < stop_step ))
    source "$folder/runtime.sh"
    export DYNAMICS_STOP_STEP="$stop_step"
    export WANDB_DIR="$folder/logs"
    mkdir -p "$folder/status" "$folder/logs"
    segment_status="$folder/status/segment-$stop_step.json"
    test ! -e "$segment_status"
    python - "$folder" "$stop_step" "$resume_step" <<'PY'
import json
import sys
from pathlib import Path
import yaml
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
folder, stop, start = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
case = json.loads((folder / "case.json").read_text())
config = yaml.safe_load((folder / "config.yml").read_text())
assert case["role"] == "defense"
assert case["max_steps"] == config["max_steps"] == 125
assert start < stop <= case["max_steps"]
state = PartialState()
assert state.distributed_type == DistributedType.NO and state.num_processes == 1
assert torch.cuda.device_count() == 4
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
print("Segment: one process, four GPUs, hipBLAS/rocBLAS", flush=True)
PY
    if [[ ! -e "$folder/status/train.json" ]]; then
        printf '{"case":"%s","state":"running"}\n' "$name" > "$folder/status/train.json"
    else
        python - "$folder/status/train.json" <<'PY'
import json
import sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text())["state"] == "running"
PY
    fi
    trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$segment_status"; fi' EXIT
    if (( resume_step > 0 )); then
        checkpoint="$folder/models/defense/checkpoint-$resume_step"
        python - "$folder" "$checkpoint" "$resume_step" <<'PY'
import json
import sys
from pathlib import Path
import yaml
folder, checkpoint, start = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
case = json.loads((folder / "case.json").read_text())
config = yaml.safe_load((folder / "config.yml").read_text())
assert config["max_steps"] == case["max_steps"] == 125
assert start in case["state_steps"] or start in (2, 4)
assert json.loads((folder / f"status/segment-{start}.json").read_text())["state"] == "complete"
state_path = checkpoint / "trainer_state.json"
state = json.loads(state_path.read_text())
assert state["global_step"] == start and state["epoch"] < 1
for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
    assert (checkpoint / filename).is_file(), filename
records = [json.loads(line) for line in (folder / "models/defense/observed-batches.jsonl").read_text().splitlines()]
assert [row["step"] for row in records] == list(range(1, start + 1))
steps_per_update = case["inner_steps_per_update"]
inner = [entry["inner_step_total"] for entry in state["log_history"] if "inner_step_total" in entry]
assert inner == list(range(start * steps_per_update)), (len(inner), start * steps_per_update)
print(records[-1]["elapsed_seconds"])
PY
        export DYNAMICS_ELAPSED_OFFSET=$(python - "$folder/models/defense/observed-batches.jsonl" <<'PY'
import json, sys
from pathlib import Path
rows = Path(sys.argv[1]).read_text().splitlines()
print(json.loads(rows[-1])["elapsed_seconds"])
PY
)
        export DYNAMICS_INNER_OFFSET=$(python - "$folder/case.json" "$resume_step" <<'PY'
import json, sys
from pathlib import Path
case = json.loads(Path(sys.argv[1]).read_text())
print(int(sys.argv[2]) * case["inner_steps_per_update"])
PY
)
        resume_args=(--resume-from-checkpoint "$checkpoint")
    else
        test ! -e "$folder/models/defense/observed-batches.jsonl"
        resume_args=()
        export DYNAMICS_ELAPSED_OFFSET=0 DYNAMICS_INNER_OFFSET=0
    fi
    started=$(date -u +%s)
    axolotl train "$TAR_CASE_CONFIG" --launcher python "${resume_args[@]}"
    finished=$(date -u +%s)
    python - "$folder" "$stop_step" "$resume_step" "$started" "$finished" "$source_root/misc/tar_compute.py" <<'PY'
import json
import sys
from pathlib import Path
folder, stop, start, began, ended, validator = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
model = folder / "models/defense"
state = json.loads((model / f"checkpoint-{stop}/trainer_state.json").read_text())
assert state["global_step"] == stop and state["epoch"] < 1
for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
    assert (model / f"checkpoint-{stop}" / filename).is_file(), filename
case = json.loads((folder / "case.json").read_text())
records = [json.loads(line) for line in (model / "observed-batches.jsonl").read_text().splitlines()]
assert [row["step"] for row in records] == list(range(1, stop + 1))
assert all(row["elapsed_seconds"] >= 0 for row in records)
inner = [entry["inner_step_total"] for entry in state["log_history"] if "inner_step_total" in entry]
expected_inner = stop * case["inner_steps_per_update"]
assert inner == list(range(expected_inner)), (len(inner), expected_inner)
status = {"state": "complete", "start_step": start, "stop_step": stop}
(folder / f"status/segment-{stop}.json").write_text(json.dumps(status) + "\n")
runtime = {"start_step": start, "stop_step": stop, "wall_seconds": int(ended)-int(began), "allocated_gpu_seconds": 4*(int(ended)-int(began)), "includes_model_load_resume_observation_and_checkpoint_io": True}
with (folder / "runtime-segments.jsonl").open("a") as stream:
    stream.write(json.dumps(runtime, separators=(",", ":")) + "\n")
if stop == case["max_steps"]:
    import subprocess
    subprocess.run([sys.executable, str(validator), "validate", str(folder)], check=True)
    rows = [json.loads(line) for line in (folder / "runtime-segments.jsonl").read_text().splitlines()]
    assert rows[0]["start_step"] == 0 and rows[-1]["stop_step"] == case["max_steps"]
    assert all(left["stop_step"] == right["start_step"] for left, right in zip(rows, rows[1:]))
    summary = {"wall_seconds": sum(row["wall_seconds"] for row in rows), "allocated_gpu_seconds": sum(row["allocated_gpu_seconds"] for row in rows), "segment_count": len(rows), "includes_model_load_resume_observation_and_checkpoint_io": True, "queue_time_included": False}
    (folder / "runtime.json").write_text(json.dumps(summary, indent=2) + "\n")
    (folder / "status/train.json").write_text(json.dumps({"case": case["name"], "state": "complete"}) + "\n")
PY
    trap - EXIT
elif [[ "$mode" == parity ]]; then
    folder="$run/pilots/$name"
    source "$folder/runtime.sh"
    export WANDB_DIR="$folder/logs"
    mkdir -p "$folder/status" "$folder/logs"
    test ! -e "$folder/status/train.json"
    trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$folder/status/train.json"; fi' EXIT
    printf '{"state":"running"}\n' > "$folder/status/train.json"
    python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
state = PartialState()
assert state.distributed_type == DistributedType.NO and state.num_processes == 1
assert torch.cuda.device_count() == 4
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
print("Parity: actual Axolotl placement, one process and four GPUs", flush=True)
PY
    axolotl train "$TAR_CASE_CONFIG" --launcher python
    python - "$folder" <<'PY'
import json
import sys
from pathlib import Path
folder = Path(sys.argv[1])
model = folder / "models/attack"
result = json.loads((model / "parity/validation.json").read_text())
assert result["state"] == "complete" and result["actual_updates"] == 16
assert result["actual_horizon"] == 1000
state = json.loads((model / "checkpoint-16/trainer_state.json").read_text())
assert state["global_step"] == 16
for step in (1, 2, 4, 8, 16):
    assert (model / f"checkpoint-{step}/model.safetensors").is_file()
    assert (model / f"parity/simulated-{step}/model.safetensors").is_file()
print("Actual/simulated prefix parity:", result["trajectory_parity"], flush=True)
PY
    printf '{"state":"complete"}\n' > "$folder/status/train.json"
elif [[ "$mode" == bundle ]]; then
    python "$source_root/misc/tar_compute.py" list "$run" "$name" > "$run/status/bundle-$name.cases"
    mapfile -t cases < "$run/status/bundle-$name.cases"
    (( ${#cases[@]} > 0 ))
    failures=0
    for selected in "${cases[@]}"; do
        if ! bash -i "$worker" case "$run" "$selected" 2>&1 | tee "$run/logs/$selected.log"; then
            failures=$((failures + 1))
        fi
    done
    if (( failures > 0 )); then
        printf '{"state":"failed","failed_cases":%d}\n' "$failures" > "$run/status/bundle-$name.json"
        exit 1
    fi
    printf '{"state":"complete"}\n' > "$run/status/bundle-$name.json"
elif [[ "$mode" == case ]]; then
    folder="$run/cases/$name"
    unset DYNAMICS_BOOTSTRAP_SEED DYNAMICS_SOURCE_MANIFEST
    source "$folder/runtime.sh"
    export WANDB_DIR="$folder/logs"
    trap 'result=$?; if (( result != 0 )); then printf "{\"case\":\"%s\",\"state\":\"failed\",\"exit_code\":%d}\n" "$name" "$result" > "$folder/status/train.json"; fi' EXIT
    test ! -e "$folder/status/train.json"
    printf '{"case":"%s","state":"running"}\n' "$name" > "$folder/status/train.json"
    started=$(date -u +%s)
    python - "$folder" "$run" <<'PY'
import json
import sys
from pathlib import Path
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
case = json.loads((Path(sys.argv[1]) / "case.json").read_text())
program = json.loads((Path(sys.argv[2]) / "study.json").read_text())
if program.get("kind") == "supplied-checkpoint-comparison":
    validation = json.loads((Path(sys.argv[2]) / "input-validation.json").read_text())
    assert validation["state"] == "passed"
    assert validation["checkpoint"] == program["supplied_checkpoint"]
if case["role"] == "attack":
    defended = Path(case["defense_reference"])
    assert (defended / "model.safetensors").exists() or (defended / "model.safetensors.index.json").exists()
    if defended.is_relative_to(Path(sys.argv[2]) / "cases"):
        validation = json.loads((defended.parents[1] / "training-validation.json").read_text())
        assert validation["state"] == "passed"
state = PartialState()
assert state.distributed_type == DistributedType.NO
assert state.num_processes == 1 and torch.cuda.device_count() == 4
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
print("Compute study: one process, four GPUs, hipBLAS/rocBLAS", flush=True)
PY
    axolotl train "$TAR_CASE_CONFIG" --launcher python
    python "$source_root/misc/tar_compute.py" validate "$folder"
    finished=$(date -u +%s)
    printf '{"wall_seconds":%d,"allocated_gpu_seconds":%d,"includes_loading_observation_and_checkpoint_io":true}\n' \
        "$((finished - started))" "$((4 * (finished - started)))" > "$folder/runtime.json"
    printf '{"case":"%s","state":"complete"}\n' "$name" > "$folder/status/train.json"
else
    echo "Unknown mode: $mode" >&2
    exit 2
fi
