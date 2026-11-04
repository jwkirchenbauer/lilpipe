#!/bin/bash
#SBATCH --job-name=seam-defence-eval
#SBATCH --output=artifacts/tamia/logs/seam_defence_eval-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
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
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
model=$(realpath "${1:?missing defended model path}")
config=$(realpath "${2:?missing eval config path}")
output=$(realpath -m "${3:?missing eval output directory}")
run=$(realpath -m "${4:?missing run metadata directory}")
test ! -e "$output/metrics.json"
mkdir -p "$output" "$run/wandb"
export WANDB_DIR="$run/wandb"
cd ../seam
CUDA_VISIBLE_DEVICES=0 "$seam_env/bin/python" scripts/eval_defence.py \
  --config "$config" --model_name "$model" --attack False \
  --pre_utility False --post_utility True \
  --result-json "$output/metrics.json"
