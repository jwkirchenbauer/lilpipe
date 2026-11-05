#!/bin/bash
#SBATCH --job-name=plot-base-seam
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/tamia/logs/plot_base_seam-%j.out
set -euo pipefail
export MPLBACKEND=Agg
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
/scratch/m/mgamal/lilpipe/envs/parameter-norms-preflight/bin/python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py
