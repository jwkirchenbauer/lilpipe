#!/bin/bash
#SBATCH --job-name=eval-wmdp-mmlu
#SBATCH --output=artifacts/mila/logs/eval_wmdp_mmlu-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$SLURM_TMPDIR/.cache/uv-$SLURM_JOB_ID"
export UV_PROJECT_ENVIRONMENT="$SLURM_TMPDIR/.venv-$SLURM_JOB_ID"
export UV_LINK_MODE=copy

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

base_model_name_or_path=${1:?missing model path}
output=${2:?missing output directory}

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

python -m lm_eval run \
  --model hf \
  --model_args "pretrained=$base_model_name_or_path,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks mmlu_no_bio,wmdp_bio_robust \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path "$output"
