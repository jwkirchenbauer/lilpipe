#!/usr/bin/env bash
# Invoke with bash -i so every Python command follows an explicit activation.
set -euo pipefail
stage=${1:?stage required}
run_dir=${2:?absolute run directory required}
[[ "$run_dir" == /* ]]
example_root=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
worker_script=${BASH_SOURCE[0]}
repro_script="$(dirname "$worker_script")/tuolumne_tar_repro.py"
if [[ "$stage" == smoke || "$stage" == defense || "$stage" == attack-* ]]; then
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
else
    mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
fi
cd "$example_root"
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}
export REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-$SSL_CERT_FILE}
export PYTHONPATH="$run_dir/source/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export WANDB_DIR="$run_dir/logs"
export AXOLOTL_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MPLCONFIGDIR="$run_dir/cache/matplotlib"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export PYTHONUNBUFFERED=1
export LOGLEVEL=WARNING
if [[ "$stage" != inputs ]]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
fi
mkdir -p "$run_dir/cache" "$MPLCONFIGDIR" "$run_dir/stage-status"
trap 'result=$?; if (( result != 0 )); then printf "{\"stage\":\"%s\",\"state\":\"failed\",\"exit_code\":%d}\n" "${next_stage:-$stage}" "$result" > "$run_dir/status.json"; fi' EXIT

evaluate() {
    local model=$1 output=$2
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
        --num_cpu_threads_per_process 8 -m lm_eval run \
        --model hf --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
        --include_path "$run_dir/source/lm_eval_tasks" \
        --tasks wmdp_bio_robust,mmlu_no_bio --num_fewshot 0 \
        --batch_size 32 --output_path "$output"
}

case "$stage" in
    inputs|configs|compare)
        if [[ "$stage" == inputs || "$stage" == configs ]]; then
            python -m unittest discover -s "$example_root/../../tests" \
                -p test_tuolumne_tar_repro.py
        fi
        python "$repro_script" "$stage" "$run_dir" --example-root "$example_root"
        ;;
    smoke|defense|attack-*)
        # Use hipBLAS/rocBLAS; this ROCm build stalls in hipBLASLt backward.
        export TORCH_BLAS_PREFER_CUBLASLT=0
        export TORCH_BLAS_PREFER_HIPBLASLT=0
        export LILPIPE_DEVICE_DEBUG=1
        # Flux's rank shim makes Accelerate infer DDP even for one outer task.
        # These configs use one process with layers mapped across four GPUs.
        unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
        python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType

state = PartialState()
assert state.distributed_type == DistributedType.NO, state.distributed_type
assert state.num_processes == 1, state.num_processes
assert state.device.type == "cuda", state.device
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
backend = torch.backends.cuda.preferred_blas_library()
assert backend == torch._C._BlasBackend.Cublas, backend
print(
    "Training launch verified: one process, four visible GPUs; "
    f"BLAS backend: {backend} (hipBLAS/rocBLAS on AMD)",
    flush=True,
)
PY
        if [[ "$stage" == attack-* ]]; then
            [[ -s "$run_dir/models/defense/model.safetensors" || \
               -s "$run_dir/models/defense/model.safetensors.index.json" ]]
        fi
        axolotl train "$run_dir/configs/$stage.yml" --launcher python
        ;;
    base)
        model=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$run_dir/inputs.json")
        if [[ ! -s "$run_dir/base_comparison.json" ]]; then
            evaluate "$model" "$run_dir/evals/base"
        fi
        python "$repro_script" base-check "$run_dir" --example-root "$example_root"
        ;;
    defense-eval)
        evaluate "$run_dir/models/defense" "$run_dir/evals/checkpoint-0"
        ;;
    trajectory-*)
        attack_stage=attack-${stage#trajectory-}
        for step in 16 32 48 64 80 96 112 128; do
            evaluate "$run_dir/models/$attack_stage/checkpoint-$step" \
                "$run_dir/evals/$attack_stage/checkpoint-$step"
        done
        python "$repro_script" compare "$run_dir" --example-root "$example_root" \
            --allow-partial
        ;;
    validate)
        bash -i "$worker_script" configs "$run_dir"
        bash -i "$worker_script" smoke "$run_dir"
        ;;
    pipeline)
        test -s "$run_dir/stage-status/smoke.json"
        python - "$run_dir/models/smoke/checkpoint-1/trainer_state.json" <<'PY'
import json
import math
import sys

state = json.load(open(sys.argv[1]))
assert state["global_step"] == 1, "The smoke test did not complete one update"
for entry in state.get("log_history", []):
    for key in ("loss", "train_loss", "grad_norm", "inner_loss"):
        if key in entry:
            assert math.isfinite(entry[key]), f"Non-finite smoke-test {key}"
PY
        for next_stage in base defense defense-eval \
            attack-lr1e-5 trajectory-lr1e-5 \
            attack-lr1e-6 trajectory-lr1e-6 \
            attack-lr5e-6 trajectory-lr5e-6 compare; do
            printf '{"stage":"%s","state":"running"}\n' "$next_stage" > "$run_dir/status.json"
            bash -i "$worker_script" "$next_stage" "$run_dir" \
                2>&1 | tee "$run_dir/logs/$next_stage.log"
        done
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
printf '{"stage":"%s","state":"complete"}\n' "$stage" \
    > "$run_dir/stage-status/$stage.json"
