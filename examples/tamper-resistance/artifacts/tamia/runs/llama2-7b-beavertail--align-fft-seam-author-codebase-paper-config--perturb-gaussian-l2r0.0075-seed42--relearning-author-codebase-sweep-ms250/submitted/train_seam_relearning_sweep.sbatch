#!/bin/bash
#SBATCH --job-name=seam-relearning-sweep
#SBATCH --output=artifacts/tamia/logs/seam_relearning_sweep-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface-seam-datasets-3.3.1"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Run from examples/tamper-resistance. An optional fourth argument enables a
# one-step optimizer/checkpoint smoke test before the five full attacks.
model=$(realpath "${1:?missing starting model path}")
models_root=$(realpath "${2:?missing model output directory}")
run_root=$(realpath -m "${3:?missing run metadata directory}")
smoke=${4:-false}
# MODEL_SLUG gives cached HF revisions a stable experiment name.
base=${MODEL_SLUG:-$(basename "$model")}
learning_rates=(2e-5 5e-5 8e-5 1e-4 2e-4)
IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ ${#gpus[@]} != 4 ]]; then
    echo "Expected four allocated GPUs" >&2
    exit 1
fi
mkdir -p "$run_root"
export WANDB_DIR="$run_root"
cd ../seam

train_attack() {
    local gpu=$1 lr=$2 output=$3 name=$4
    shift 4
    CUDA_VISIBLE_DEVICES="$gpu" "$seam_env/bin/python" scripts/train.py \
      --config config/eval.yaml \
      --model_name "$model" --output_dir "$output" \
      --attack True --pre_utility False --post_utility False --lora_attack False \
      --attack_size 1000 --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 1 --num_train_epochs 1 \
      --learning_rate "$lr" --save_strategy steps --save_steps 50 \
      --save_only_model True --seed 42 --wandb_run_name "$name" "$@"
}

# Refuse to mix a new sweep with existing attacked models.
for lr in "${learning_rates[@]}"; do
    output="$models_root/$base--data-overlapping--relearn-fft-author-codebase-lr$lr-ms250"
    if [[ -e "$output" ]]; then
        echo "Attack output already exists: $output" >&2
        exit 1
    fi
done

if [[ "$smoke" == true ]]; then
    smoke_output="$models_root/$base--attack-smoke-$SLURM_JOB_ID"
    train_attack "${gpus[0]}" 2e-5 "$smoke_output" "$base--attack-smoke-$SLURM_JOB_ID" \
      --max_steps 1 --save_steps 1 > "$run_root/smoke-$SLURM_JOB_ID.log" 2>&1
    "$seam_env/bin/python" - "$smoke_output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for directory in (root, root / "checkpoint-1"):
    assert json.loads((directory / "trainer_state.json").read_text())["global_step"] == 1
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    assert all((directory / shard).is_file() for shard in set(index["weight_map"].values()))
    assert not (directory / "optimizer.pt").exists()
    assert (directory / "tokenizer_config.json").is_file()
print("One-step optimizer and model-only checkpoint smoke test passed", flush=True)
PY
fi

# Each worker owns one GPU and processes its learning rates sequentially.
# GPU 0 runs the fifth attack after finishing the first.
run_worker() {
    local worker=$1 index lr output name
    for ((index=worker; index<${#learning_rates[@]}; index+=4)); do
        lr=${learning_rates[index]}
        name="$base--data-overlapping--relearn-fft-author-codebase-lr$lr-ms250"
        output="$models_root/$name"
        train_attack "${gpus[worker]}" "$lr" "$output" "$name" \
          > "$run_root/$name.log" 2>&1
    done
}
pids=()
for worker in 0 1 2 3; do
    run_worker "$worker" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
    echo "One or more relearning attacks failed; inspect the per-attack logs" >&2
    exit 1
fi

"$seam_env/bin/python" - "$models_root" "$base" <<'PY'
import json
import sys
from pathlib import Path

root, base = Path(sys.argv[1]), sys.argv[2]
for lr in ("2e-5", "5e-5", "8e-5", "1e-4", "2e-4"):
    model = root / f"{base}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250"
    for step, directory in [(s, model / f"checkpoint-{s}") for s in range(50, 251, 50)] + [(250, model)]:
        state = json.loads((directory / "trainer_state.json").read_text())
        assert state["global_step"] == step, (directory, state["global_step"])
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        assert all((directory / shard).is_file() for shard in set(index["weight_map"].values()))
        assert (directory / "tokenizer_config.json").is_file()
    print(f"Verified all checkpoints and final model: {model.name}", flush=True)
PY
