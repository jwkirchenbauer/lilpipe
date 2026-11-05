#!/bin/bash
#SBATCH --job-name=plot-noisy-seam-relearning
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tamia/logs/plot_noisy_seam_relearning-%j.out
set -euo pipefail
export MPLBACKEND=Agg
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python /home/m/mgamal/projects/lilpipe/examples/tamper-resistance/artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--perturb-gaussian-l2r0.0075-seed42--relearning-author-codebase-sweep-ms250/submitted/plot_trajectory_beavertails.py
