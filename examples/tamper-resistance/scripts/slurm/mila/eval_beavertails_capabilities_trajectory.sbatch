#!/bin/bash
#SBATCH --job-name=eval-beavertails-capabilities-trajectory
#SBATCH --output=artifacts/mila/logs/eval_beavertails_capabilities_trajectory-%j.out
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

model_id=${1:?missing model id}
baseline_model_name_or_path=${2:?missing baseline model path}
base_model_name_or_path=${3:?missing model path}
learning_rate=${4:?missing learning rate}
attack_data=${5:?missing attack data role}
method=${6:?missing method}
result_root=${7:?missing result root}
rm_intermediate_ckpts_after_eval=${8:-false}
rm_final_ckpt_after_eval=${9:-false}

uv sync --frozen --group eval
source "$UV_PROJECT_ENVIRONMENT/bin/activate"

for ckpt_name_or_path in "$base_model_name_or_path"/checkpoint-*; do
    [[ -d "$ckpt_name_or_path" ]] || continue
    step=${ckpt_name_or_path##*/checkpoint-}

    python scripts/evals/evaluate_beavertail_and_capabilities.py trajectory \
        "$model_id" "$baseline_model_name_or_path" "$base_model_name_or_path" \
        "$learning_rate" "$attack_data" "$method" "$result_root" \
        --checkpoint-step "$step" --checkpoint-path "$ckpt_name_or_path"

    if [[ "$rm_intermediate_ckpts_after_eval" == true ]]; then
        rm -r -- "$ckpt_name_or_path"
    fi
done

if [[ "$rm_final_ckpt_after_eval" == true ]]; then
    rm -r -- "$base_model_name_or_path"
fi
