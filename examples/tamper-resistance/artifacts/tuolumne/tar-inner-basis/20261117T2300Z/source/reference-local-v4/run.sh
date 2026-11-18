#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* && -n "${FLUX_JOB_ID:-}" ]]
mode=${1:?all, train, validate, or report required}
run=${2:?absolute study path required}
[[ "$run" == /* ]]
profile=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$profile/manifest.sha256"
sha256sum --quiet -c "$run/source-manifest.sha256"
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$profile:$run/source/misc:$run/source/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false AXOLOTL_DO_NOT_TRACK=1 WANDB_MODE=offline
export WANDB_DIR="$run/logs" LILPIPE_DEVICE_DEBUG=1
export MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$run/logs" "$run/status" "$MPLCONFIGDIR"
status="$run/status/$mode.json"
cache_info=""
cleanup_reference_cache() {
    if [[ -n "$cache_info" && -s "$cache_info/provenance.json" ]]; then
        python "$profile/tar_reference_cache.py" cleanup --provenance "$cache_info/provenance.json"
    fi
}
trap 'result=$?; trap - EXIT; if ! cleanup_reference_cache; then result=1; fi; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$status"; fi; exit "$result"' EXIT
printf '{"state":"running"}\n' > "$status"
if [[ "$mode" == all ]]; then
    for part in train validate report; do
        bash -i "${BASH_SOURCE[0]}" "$part" "$run" 2>&1 | tee "$run/logs/$part.log"
    done
elif [[ "$mode" == train ]]; then
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
    export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
    export DYNAMICS_SAVE_STEPS=1,2,4,8,16,32,64,96,125
    export DYNAMICS_STATE_STEPS=32,64,96,125
    export TAR_INNER_RECORD_ROOT="$run/inner-local-v4"
    export TAR_INNER_REDUCTION_DEVICE=cuda:0 TAR_INNER_CHUNK_ELEMENTS=1048576
    export TAR_INNER_OUTER_STEPS=1,32,64,96,125
    export TAR_INNER_SAVE_STEPS=1,2,4,8,12,16
    export TAR_INNER_REFERENCE_SPEC="$run/references.json"
    python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
state = PartialState()
assert state.distributed_type == DistributedType.NO
assert state.num_processes == 1 and torch.cuda.device_count() == 4
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
print("Inner observation: one process, four visible GPUs, rocBLAS", flush=True)
PY
    python "$profile/gpu_preflight.py" "$run"
    cache_key=${FLUX_JOB_ID//[^a-zA-Z0-9_-]/_}-${BASHPID}-$(date -u +%Y%m%dT%H%M%S)
    cache_info="$run/cache-provenance/$cache_key"
    ram_root=${FLUX_JOB_TMPDIR:-${TMPDIR:-/tmp}}
    numactl --interleave=all python "$profile/tar_reference_cache.py" stage \
        --spec "$run/references.json" --ssd-root "/l/ssd/$USER" \
        --ram-root "$ram_root" --output "$cache_info" \
        --reserve-ssd-bytes 21474836480 --max-ram-bytes 240000000000 \
        --max-model-bytes 450000000000
    export TAR_INNER_REFERENCE_SPEC="$cache_info/references.json"
    numactl --interleave=all axolotl train "$run/cases/baseline-inner-recording/config.yml" --launcher python
    cleanup_reference_cache
    cache_info=""
elif [[ "$mode" == validate ]]; then
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
    export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
    python "$profile/validate_replay.py" "$run" --inner-root "$run/inner-local-v4"
elif [[ "$mode" == report ]]; then
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
    export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
    test -s "$run/replay-validation.json"
    python -c 'import json,sys; assert json.load(open(sys.argv[1]))["state"] == "passed"' "$run/replay-validation.json"
    python "$run/source/misc/tar_inner_geometry.py" report --spec "$profile/analysis-spec.json" --output "$run/geometry"
else
    exit 2
fi
printf '{"state":"complete"}\n' > "$status"
