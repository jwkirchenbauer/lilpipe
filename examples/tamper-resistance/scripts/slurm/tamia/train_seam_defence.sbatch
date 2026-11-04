#!/bin/bash
#SBATCH --job-name=seam-defence
#SBATCH --output=artifacts/tamia/logs/seam_defence-%j.out
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
set -euo pipefail

# ============================================
# CLUSTER SPECIFIC CONFIG
# ============================================
export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
seam_env=${SEAM_ENV:-$SCRATCH/lilpipe/seam-official/venv}

# ============================================
# APPLICATION SPECIFIC CONFIG
# ============================================
# Run from examples/tamper-resistance. Pass 1 as the fourth argument for a smoke test.
config=$(realpath "${1:?missing defence config path}")
model=$(realpath -m "${2:?missing model output path}")
run=$(realpath -m "${3:?missing run metadata directory}")
steps=${4:-500}
test ! -e "$model"
test ! -L "$model"
mkdir -p "$run"
export WANDB_DIR="$run/wandb"
mkdir -p "$WANDB_DIR"
cd ../seam
CUDA_VISIBLE_DEVICES=0,1,2,3 "$seam_env/bin/python" -m src.train \
  --config "$config" --output_dir "$model" \
  --max_steps "$steps" --save_steps "$steps" --save_only_model True
checkpoint="$model/checkpoint-$steps"
"$seam_env/bin/python" scripts/verify_checkpoint.py \
  "$checkpoint" "$steps" "$run/defence-validation.json"
for path in "$checkpoint"/*; do
    test ! -e "$model/${path##*/}"
done
for path in "$checkpoint"/*; do
    mv -- "$path" "$model/"
done
rmdir "$checkpoint"
"$seam_env/bin/python" scripts/verify_checkpoint.py \
  "$model" "$steps" "$run/defence-validation.json"
