#!/bin/bash
#SBATCH --job-name=eval-noise-local-logs
#SBATCH --output=artifacts/tamia/logs/eval_noise_local_logs-%j.out
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail
log="$SLURM_TMPDIR/evaluation-$SLURM_JOB_ID.log"
trap 'cp "$log" "artifacts/tamia/logs/eval-noise-local-$SLURM_JOB_ID.log"' EXIT
bash /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-percent1-2-seed42/submitted/eval_harmfulness_and_capabilities-relative.sbatch "$@" > "$log" 2>&1
