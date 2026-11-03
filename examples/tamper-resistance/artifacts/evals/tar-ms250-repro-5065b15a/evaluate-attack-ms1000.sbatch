#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/%x-%j.out
set -euo pipefail
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
bash scripts/slurm/tamia/eval_wmdp_mmlu_trajectory.sbatch /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/relearn-lr5e-6-ms1000 /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/evals/relearn-lr5e-6-ms1000 false false /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/defence/checkpoint-250
