#!/bin/bash
#SBATCH --job-name=eval-wmdp-mmlu
#SBATCH --output=artifacts/tamia/logs/eval_wmdp_mmlu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"

# Tamia compute nodes use cached packages, models, and datasets.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
export UV_LINK_MODE=copy

# Allow access to the W&B server.
module load httpproxy/1.0

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

export CUDA_CACHE_PATH="$SLURM_TMPDIR/cuda-cache-$SLURM_JOB_ID"
mkdir -p "$CUDA_CACHE_PATH"
checkpoints=false
if [[ ${1:-} == --checkpoints ]]; then
    checkpoints=true
    shift
fi
model=${1:?missing model path}
output=${2:?missing output directory}
shift 2
# Legacy deletion flags remain explicit; they never remove the baseline.
remove_checkpoints=${1:-false}
remove_final=${2:-false}
baseline=${3:-}
tasks=${CAPABILITY_TASKS:-mmlu_no_bio,wmdp_bio_robust}

evaluate_model() {
    local pretrained=$1 destination=$2
    if [[ -d "$pretrained" ]]; then
        pretrained=$(realpath -s --relative-to="$PWD" "$pretrained")
    fi
    accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
      --num_cpu_threads_per_process 8 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$pretrained,dtype=bfloat16,trust_remote_code=True" \
      --include_path lm_eval_tasks --tasks "$tasks" --num_fewshot 0 \
      --batch_size "${BATCH_SIZE:-32}" --max_batch_size "${MAX_BATCH_SIZE:-128}" \
      --output_path "$destination"
    python - "$destination" "$pretrained" "$tasks" <<'PYTHON'
import json
import sys
from pathlib import Path

output, model, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split(",")
results = {}
for file in sorted(output.rglob("results_*.json")):
    data = json.loads(file.read_text())
    config = data.get("config", {})
    args = config.get("model_args", {})
    if isinstance(args, str):
        args = dict(part.split("=", 1) for part in args.split(",") if "=" in part)
    if args.get("pretrained") != model or config.get("limit") is not None:
        continue
    results.update(data.get("results", {}))
assert all(task in results for task in tasks), "Missing results for the requested model/tasks"
PYTHON
}

if [[ "$checkpoints" == false ]]; then
    evaluate_model "$model" "$output"
    exit
fi
if [[ -n "$baseline" ]]; then
    evaluate_model "$baseline" "$output/checkpoint-0"
fi
mapfile -t checkpoint_paths < <(find "$model" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if (( ${#checkpoint_paths[@]} == 0 )); then
    echo "No checkpoints found: $model" >&2
    exit 1
fi
for checkpoint in "${checkpoint_paths[@]}"; do
    evaluate_model "$checkpoint" "$output/$(basename "$checkpoint")"
done
# Delete only after the whole requested trajectory has valid results.
if [[ "$remove_checkpoints" == true ]]; then
    for checkpoint in "${checkpoint_paths[@]}"; do
        [[ "$checkpoint" != "$baseline" ]] && rm -r -- "$checkpoint"
    done
fi
if [[ "$remove_final" == true ]]; then
    [[ -n "$model" && "$model" != / && "$model" != . && "$model" != "$baseline" ]]
    rm -r -- "$model"
fi
