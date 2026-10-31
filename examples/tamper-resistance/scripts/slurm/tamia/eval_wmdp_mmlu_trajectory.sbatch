#!/bin/bash
#SBATCH --job-name=eval-wmdp-mmlu-trajectory
#SBATCH --output=artifacts/tamia/logs/eval_wmdp_mmlu_trajectory-%j.out
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

base_model_name_or_path=${1:?missing model path}
output_dir=${2:?missing output directory}
rm_intermediate_ckpts_after_eval=${3:-false}
rm_final_ckpt_after_eval=${4:-false}
baseline_model_name_or_path=${5:-}

for ckpt_name_or_path in "$baseline_model_name_or_path" "$base_model_name_or_path"/checkpoint-*; do
    [[ -d "$ckpt_name_or_path" ]] || continue
    output="$output_dir/${ckpt_name_or_path##*/}"
    if [[ "$ckpt_name_or_path" == "$baseline_model_name_or_path" ]]; then
        output="$output_dir/checkpoint-0"
    fi

    accelerate launch --num_processes=4 -m lm_eval run \
      --model hf \
      --model_args "pretrained=$ckpt_name_or_path,dtype=bfloat16,trust_remote_code=True" \
      --include_path lm_eval_tasks \
      --tasks mmlu_no_bio,wmdp_bio_robust \
      --num_fewshot 0 \
      --batch_size 32 \
      --output_path "$output"

    if [[ "$rm_intermediate_ckpts_after_eval" == true && "$ckpt_name_or_path" != "$baseline_model_name_or_path" ]]; then
        rm -r -- "$ckpt_name_or_path"
    fi
done

if [[ "$rm_final_ckpt_after_eval" == true ]]; then
    rm -r -- "$base_model_name_or_path"
fi
