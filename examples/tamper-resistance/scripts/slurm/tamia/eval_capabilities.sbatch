#!/bin/bash
#SBATCH --job-name=eval-capabilities
#SBATCH --output=artifacts/tamia/logs/eval_capabilities-%j.out
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

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
model=${1:?missing model path}
output=${2:?missing output directory}
shift 2

# Keep lm-eval result-directory components below the filesystem name limit.
if [[ -d "$model" ]]; then
    model=$(realpath -s --relative-to="$PWD" "$model")
fi

# Task overrides are shared with the combined evaluator; defaults retain the
# original four-task, zero-shot protocol. GSM8K can be run separately at five-shot.
tasks=${CAPABILITY_TASKS:-mmlu,truthfulqa_mc1,arc_easy,hellaswag}
num_fewshot=${NUM_FEWSHOT:-0}
max_batch_size=${MAX_BATCH_SIZE:-128}

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
  --num_cpu_threads_per_process 8 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
  --tasks "$tasks" \
  --num_fewshot "$num_fewshot" \
  --batch_size auto:4 \
  --max_batch_size "$max_batch_size" \
  --output_path "$output" \
  "$@"

if [[ -z $(find "$output" -name 'results_*.json' -print -quit) ]]; then
    echo "lm-eval did not save results under $output" >&2
    exit 1
fi
