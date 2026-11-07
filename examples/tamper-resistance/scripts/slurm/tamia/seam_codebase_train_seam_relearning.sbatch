#!/bin/bash
#SBATCH --job-name=seam-codebase-relearning
#SBATCH --output=artifacts/tamia/logs/seam_codebase_relearning-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="$PWD/artifacts/tamia/logs"
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Use the authors' native YAML, just like the author defense launcher.
# Usage: CONFIG MODEL OUTPUT LR [AUTHOR_ARGS...]
config=$(realpath "${1:?missing native author config}")
model=$(realpath "${2:?missing starting checkpoint}")
output=$(realpath -m "${3:?missing attack output directory}")
lr=${4:?missing learning rate}
shift 4
if [[ -e "$output" ]]; then
    echo "Attack output already exists: $output" >&2
    exit 1
fi
name=$(basename "$output")
cd ../seam
exec "$seam_env/bin/python" scripts/train.py \
  --config "$config" --model_name "$model" --output_dir "$output" \
  --learning_rate "$lr" --wandb_run_name "$name" \
  --save_strategy steps --save_steps 50 --save_only_model True "$@"
