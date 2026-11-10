#!/usr/bin/env bash
# Invoke with bash -i through launch_tuo.py.
set -euo pipefail
stage=${1:?stage required}
study=${2:?absolute study directory required}
[[ "$study" == /* ]]
if [[ "$stage" == train || "$stage" == defense || "$stage" == attack ]]; then
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
else
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
fi
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$study/source/misc:$study/source/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline WANDB_DIR="$study/logs" AXOLOTL_DO_NOT_TRACK=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MPLBACKEND=Agg MPLCONFIGDIR="$study/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$study/logs" "$study/status" "$MPLCONFIGDIR"
trap 'result=$?; if (( result != 0 )); then printf "{\"stage\":\"%s\",\"state\":\"failed\",\"exit_code\":%d}\n" "$stage" "$result" > "$study/status/$stage.json"; fi' EXIT
if [[ "$stage" == train ]]; then
    bash -i "${BASH_SOURCE[0]}" defense "$study" 2>&1 | tee "$study/logs/defense.log"
    bash -i "${BASH_SOURCE[0]}" attack "$study" 2>&1 | tee "$study/logs/attack.log"
elif [[ "$stage" == defense || "$stage" == attack ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
    export LILPIPE_DEVICE_DEBUG=1
    if [[ "$stage" == defense ]]; then
        export DYNAMICS_SAVE_STEPS=1,2,4,8,16,32,64,96,125
    else
        export DYNAMICS_SAVE_STEPS=1,2,4,8,16,32,48,64,80,96,112,128
    fi
    python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
state = PartialState()
assert state.distributed_type == DistributedType.NO
assert state.num_processes == 1 and torch.cuda.device_count() == 4
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
print("Dynamics training: one process, four GPUs, rocBLAS", flush=True)
PY
    axolotl train "$study/configs/$stage.yml" --launcher python
elif [[ "$stage" == analysis ]]; then
    test -s "$study/status/train.json"
    for phase in geometry surfaces report; do
        bash -i "${BASH_SOURCE[0]}" "$phase" "$study" 2>&1 | tee "$study/logs/$phase.log"
    done
elif [[ "$stage" == geometry || "$stage" == surfaces || "$stage" == report ]]; then
    test -s "$study/status/train.json"
    export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
    python "$study/source/misc/tar_dynamics.py" "$stage" "$study"
else
    echo "Unknown stage: $stage" >&2
    exit 2
fi
printf '{"stage":"%s","state":"complete"}\n' "$stage" > "$study/status/$stage.json"
