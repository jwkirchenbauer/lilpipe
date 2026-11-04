#!/bin/bash
#SBATCH --job-name=seam-harmfulness
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/logs/seam-harmfulness-%j.out
set -euo pipefail
export HF_HOME=/scratch/m/mgamal/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/m/mgamal/.cache/huggingface-seam-datasets-3.3.1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd /home/m/mgamal/projects/lilpipe/examples/seam
/scratch/m/mgamal/lilpipe/seam-official/venv/bin/accelerate launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision no \
  --num_cpu_threads_per_process 4 \
  scripts/eval_harmfulness.py \
  /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config \
  --batch-size 16 \
  --output /project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts/tamia/evals/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config/harmfulness-4gpu-bs16/metrics.json
