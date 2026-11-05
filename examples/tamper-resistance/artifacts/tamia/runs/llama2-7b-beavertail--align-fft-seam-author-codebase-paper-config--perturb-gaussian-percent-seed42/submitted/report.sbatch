#!/bin/bash
#SBATCH --job-name=report-noisy-seam
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tamia/logs/report_noisy_seam-%j.out
set -euo pipefail
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/lilpipe results configs/results/llama2-7b-gaussian-noise.yml > /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-percent-seed42/results.md
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/lilpipe results configs/results/llama2-7b-gaussian-noise.yml --format json > /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-percent-seed42/results.json
