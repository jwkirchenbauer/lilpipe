#!/bin/bash
#SBATCH --job-name=seam-paper-defence
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/logs/seam-paper-defence-%j.out
set -euo pipefail

export XDG_CACHE_HOME=/scratch/m/mgamal/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR=/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/wandb
mkdir -p "$WANDB_DIR"
cd /home/m/mgamal/projects/lilpipe/examples/seam

test ! -e /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defense
CUDA_VISIBLE_DEVICES=0,1,2,3 /scratch/m/mgamal/lilpipe/seam-official/venv/bin/python -m src.train \
    --config /home/m/mgamal/projects/lilpipe/examples/seam/config/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config.yaml
/scratch/m/mgamal/lilpipe/seam-official/venv/bin/python /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/verify_checkpoint.py \
    /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defense/checkpoint-500 500 /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defence-validation.json
ln -s -- /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defense/checkpoint-500 \
    /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config
