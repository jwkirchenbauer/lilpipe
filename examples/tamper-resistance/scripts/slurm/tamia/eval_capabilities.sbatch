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

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
  --num_cpu_threads_per_process 8 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$model,dtype=bfloat16,trust_remote_code=True" \
  --tasks mmlu,truthfulqa_mc1,arc_easy,hellaswag \
  --num_fewshot 0 \
  --batch_size auto:4 \
  --max_batch_size 128 \
  --output_path "$output" \
  "$@"
