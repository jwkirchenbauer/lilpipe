#!/bin/bash
#SBATCH --job-name=eval-harmfulness-capabilities
#SBATCH --output=artifacts/tamia/logs/eval_harmfulness_and_capabilities-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy

# Keep the CUDA driver cache off the shared home filesystem.
export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache-$SLURM_JOB_ID"
mkdir -p "$CUDA_CACHE_PATH"

# Optional node-local logging avoids frequent concurrent Lustre log writes.
if [[ ${EVAL_LOCAL_LOG:-0} == 1 ]]; then
    local_log="$SLURM_TMPDIR/eval_harmfulness_and_capabilities-$SLURM_JOB_ID.log"
    archived_log="artifacts/tamia/logs/eval_harmfulness_and_capabilities-$SLURM_JOB_ID.log"
    mkdir -p "$(dirname "$archived_log")"
    echo "Detailed log: $local_log; copied to $archived_log on exit"
    trap 'cp -- "$local_log" "$archived_log"' EXIT
    trap 'exit 143' TERM
    exec > "$local_log" 2>&1
fi

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Usage: [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]
# --checkpoints evaluates every 50 steps through CHECKPOINT_MAX_STEP (default 250).
# The identical final model reuses the last checkpoint evaluation.
checkpoints=false
steps=
baseline=
while [[ ${1:-} == --max-step || ${1:-} == --steps || ${1:-} == --baseline ]]; do
    case $1 in
        --max-step) export CHECKPOINT_MAX_STEP=${2:?missing maximum checkpoint step} ;;
        --steps) steps=${2:?missing comma-separated checkpoint steps} ;;
        --baseline) baseline=${2:?missing baseline model} ;;
    esac
    shift 2
done
if [[ ${1:-} == --checkpoints ]]; then
    checkpoints=true
    shift
fi
if (( $# < 2 || $# % 2 != 0 )); then
    echo "Usage: $0 [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]" >&2
    exit 2
fi
# Keep the authors' harmfulness dependencies separate from the project lm-eval env.
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}
harmfulness_script=../seam/scripts/eval_harmfulness.py
include_harmfulness=${INCLUDE_HARMFULNESS:-1}
if [[ "$include_harmfulness" == 1 ]]; then
    test -x "$seam_env/bin/python"
    test -f "$harmfulness_script"
fi
validation_python="$seam_env/bin/python"

# Task overrides are shared with the combined evaluator; defaults retain the
# original four-task, zero-shot protocol. GSM8K can be run separately at five-shot.
tasks=${CAPABILITY_TASKS:-mmlu,truthfulqa_mc1,arc_easy,hellaswag}
num_fewshot=${NUM_FEWSHOT:-0}
max_batch_size=${MAX_BATCH_SIZE:-128}
include_gsm8k=${INCLUDE_GSM8K:-0}
harmfulness_only=${HARMFULNESS_ONLY:-0}
resume=${RESUME_EVALUATION:-0}
capabilities_subdir=${CAPABILITIES_SUBDIR:-capabilities}
if [[ "$harmfulness_only" != 1 ]]; then
    uv sync --frozen --group eval
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
    validation_python=${EVAL_PYTHON:-python}
fi

# Resume only valid, full-dataset results for the requested model and protocol.
evaluation_complete() {
    [[ "$resume" == 1 ]] || return 1
    "$validation_python" - "$1" "$2" "$3" "${4:-}" "${5:-0}" <<'PYTHON'
import json
import sys
from pathlib import Path

kind, location, model, tasks, shots = sys.argv[1:]
path = Path(location)
files = [path] if kind == "harmfulness" else sorted(path.rglob("results_*.json"))
results = {}
matched = False
for file in files:
    if not file.is_file():
        continue
    try:
        data = json.loads(file.read_text())
    except (ValueError, OSError):
        file.rename(file.with_name(file.name + ".incomplete"))
        continue
    if kind == "harmfulness":
        matched = (
            Path(data.get("model", "")).resolve() == Path(model).resolve()
            and data.get("samples") == 100
            and isinstance(data.get("harmfulness"), (int, float))
            and 0 <= data["harmfulness"] <= 1
        )
    else:
        config = data.get("config", {})
        args = config.get("model_args", {})
        if not isinstance(args, dict):
            continue
        if (Path(args.get("pretrained", "")).resolve() == Path(model).resolve()
                and config.get("limit") is None
                and data.get("configs")
                and all(task.get("num_fewshot") == int(shots)
                        for task in data["configs"].values())):
            results.update(data.get("results", {}))
            matched = all(task in results for task in tasks.split(","))
sys.exit(0 if matched else 1)
PYTHON
}

evaluate_capabilities() {
    local model=$1 output=$2 task_list=$3 shots=$4 max_batch=$5
    if evaluation_complete capabilities "$output" "$model" "$task_list" "$shots"; then
        echo "Reusing completed capabilities: $output"
        return
    fi
    local pretrained=$model
    if [[ -d "$pretrained" ]]; then
        pretrained=$(realpath -s --relative-to="$PWD" "$pretrained")
    fi
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
      --num_cpu_threads_per_process 8 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$pretrained,dtype=bfloat16,trust_remote_code=True" \
      --tasks "$task_list" --num_fewshot "$shots" \
      --batch_size auto:4 --max_batch_size "$max_batch" --output_path "$output"
    if [[ -z $(find "$output" -name 'results_*.json' -print -quit) ]]; then
        echo "lm-eval did not save results under $output" >&2
        exit 1
    fi
    if [[ "$resume" == 1 ]]; then
        evaluation_complete capabilities "$output" "$model" "$task_list" "$shots"
    fi
}

evaluate_model() {
    local model=$1 output=$2
    mkdir -p "$output"
    echo "Evaluating $model at $(date -Is)"
    if [[ "$harmfulness_only" != 1 ]]; then
        evaluate_capabilities "$model" "$output/$capabilities_subdir" "$tasks" "$num_fewshot" "$max_batch_size"
        if [[ "$include_gsm8k" == 1 ]]; then
            evaluate_capabilities "$model" "$output/gsm8k-5shot" gsm8k 5 32
        fi
    fi

    [[ "$include_harmfulness" == 1 ]] || return 0
    if evaluation_complete harmfulness "$output/harmfulness.json" "$model"; then
        echo "Reusing completed harmfulness: $output"
        return
    fi
    HF_DATASETS_CACHE="$XDG_CACHE_HOME/huggingface-seam-datasets-3.3.1" \
    TOKENIZERS_PARALLELISM=false \
    "$seam_env/bin/python" -m accelerate.commands.launch \
      --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision no \
      --num_cpu_threads_per_process 8 \
      "$harmfulness_script" "$model" \
      --batch-size 16 \
      --output "$output/harmfulness.json"
    if [[ "$resume" == 1 ]]; then
        evaluation_complete harmfulness "$output/harmfulness.json" "$model"
    fi
}

while (( $# )); do
    model=$1
    output=$2
    shift 2
    if [[ "$checkpoints" == false ]]; then
        evaluate_model "$model" "$output"
        continue
    fi
    final_step=${CHECKPOINT_MAX_STEP:-250}
    if [[ ! "$final_step" =~ ^[1-9][0-9]*$ ]] || (( final_step < 1 )); then
        echo "CHECKPOINT_MAX_STEP must be positive" >&2
        exit 2
    fi
    first_step=${CHECKPOINT_START_STEP:-50}
    last_step=${CHECKPOINT_END_STEP:-$final_step}
    if [[ -z "$steps" ]] && { [[ ! "$first_step" =~ ^[1-9][0-9]*$ || ! "$last_step" =~ ^[1-9][0-9]*$ ]] ||
       (( first_step % 50 != 0 || last_step % 50 != 0 || first_step > last_step || last_step > final_step )); }; then
        echo "Checkpoint bounds must be multiples of 50 within CHECKPOINT_MAX_STEP" >&2
        exit 2
    fi
    # Confirm that the final model and last checkpoint are interchangeable before
    # using a single evaluation for both. Include tokenizer and generation config.
    "$validation_python" - "$model" "$final_step" <<'PYTHON'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = root / f"checkpoint-{sys.argv[2]}"
index_path = root / "model.safetensors.index.json"
checkpoint_index = checkpoint / index_path.name
if index_path.exists() or checkpoint_index.exists():
    index = json.loads(index_path.read_text())
    assert index == json.loads(checkpoint_index.read_text())
    files = set(index["weight_map"].values())
else:
    files = {"model.safetensors"}
files |= {
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json",
}
# Transformers versions save special tokens/templates in different files. Check
# every asset present on either side, so a missing or changed file cannot alias.
for directory in (root, checkpoint):
    for filename in ("special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
        if (directory / filename).exists():
            files.add(filename)
    files.update(str(path.relative_to(directory))
                 for path in (directory / "chat_templates").rglob("*.jinja"))
for filename in files:
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest(root / filename) == digest(checkpoint / filename), filename
print(f"Final model matches {checkpoint.name}: {root}", flush=True)
PYTHON
    if [[ -n "$steps" ]]; then
        IFS=, read -ra checkpoint_steps <<< "$steps"
    else
        mapfile -t checkpoint_steps < <(seq "$first_step" 50 "$last_step")
    fi
    previous=-1
    for step in "${checkpoint_steps[@]}"; do
        if [[ ! "$step" =~ ^[0-9]+$ ]] || (( step <= previous || step > final_step )); then
            echo "Checkpoint steps must be increasing integers within the final step" >&2
            exit 2
        fi
        previous=$step
        checkpoint="$model/checkpoint-$step"
        if [[ "$step" == 0 && ! -d "$checkpoint" ]]; then
            checkpoint=${baseline:?step zero requires --baseline or checkpoint-0}
        fi
        test -d "$checkpoint"
        evaluate_model "$checkpoint" "$output/checkpoint-$step"
    done
    if (( previous < final_step )); then
        continue
    fi
    "$validation_python" - "$model" "$output" "$include_gsm8k" "$harmfulness_only" "$final_step" "$include_harmfulness" "$capabilities_subdir" <<'PYTHON'
import json
import shutil
import sys
from pathlib import Path

model, output = sys.argv[1], Path(sys.argv[2])
checkpoint = f"checkpoint-{sys.argv[5]}"
if sys.argv[4] != "1":
    shutil.copytree(output / checkpoint / sys.argv[7], output / sys.argv[7], dirs_exist_ok=True)
    if sys.argv[3] == "1":
        shutil.copytree(output / checkpoint / "gsm8k-5shot", output / "gsm8k-5shot", dirs_exist_ok=True)
if sys.argv[6] == "1":
    report = json.loads((output / checkpoint / "harmfulness.json").read_text())
    report["evaluated_model"] = report["model"]
    report["model"] = model
    report["reused_from"] = f"{checkpoint}/harmfulness.json"
    (output / "harmfulness.json").write_text(json.dumps(report, indent=2) + "\n")
(output / "final-evaluation.json").write_text(json.dumps({
    "model": model, "reused_from": checkpoint,
    "reason": "Identical model, tokenizer, and generation configuration (SHA-256 verified)",
}, indent=2) + "\n")
PYTHON
done
