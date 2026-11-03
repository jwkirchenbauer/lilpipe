#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/%x-%j.out
set -euo pipefail
cd /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance
python3 /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/gate.py
cd /home/m/mgamal/projects/lilpipe/examples/tamper-resistance
bash scripts/slurm/tamia/train.sbatch /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/attack.yml
