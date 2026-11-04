#!/bin/bash
#SBATCH --job-name=seam-paper-defence-eval
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/logs/seam-paper-defence-eval-%j.out
set -euo pipefail
export SEAM_ROOT=/home/m/mgamal/projects/lilpipe/examples/seam
export XDG_CACHE_HOME=/scratch/m/mgamal/.cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_DATASETS_CACHE="$XDG_CACHE_HOME/huggingface-seam-datasets-3.3.1"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR=/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/wandb
mkdir -p "$WANDB_DIR" /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/evals/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defence
cd "$SEAM_ROOT"
CUDA_VISIBLE_DEVICES=0 /scratch/m/mgamal/lilpipe/seam-official/venv/bin/python /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/generated/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/eval_defence.py \
    --config /home/m/mgamal/projects/lilpipe/examples/seam/config/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--eval.yaml \
    --result-json /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/evals/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/defence/metrics.json
