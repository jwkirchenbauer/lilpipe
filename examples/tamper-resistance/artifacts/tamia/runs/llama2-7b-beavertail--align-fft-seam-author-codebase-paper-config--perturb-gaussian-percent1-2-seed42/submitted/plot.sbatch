#!/bin/bash
#SBATCH --job-name=plot-seam-noise-1-2
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tamia/logs/plot_seam_noise_1_2-%j.out
set -euo pipefail
export MPLBACKEND=Agg
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-percent1-2-seed42/submitted/plot_trajectory_beavertails.py
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-percent1-2-seed42/submitted/plot_trajectory_beavertails.py --lr-markers
