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
if [[ "$mode" == bundle ]]; then
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
    python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
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
