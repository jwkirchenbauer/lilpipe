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
export WANDB_MODE=disabled
bash scripts/slurm/tamia/train.sbatch /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/smoke.yml
python3 - <<'CHECK'
from pathlib import Path
p = Path('/scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/smoke-model/checkpoint-1')
assert list(p.glob('*.safetensors')), 'Smoke checkpoint weights missing'
assert (p / 'trainer_state.json').exists(), 'Smoke trainer state missing'
CHECK
rm -r -- /scratch/m/mgamal/lilpipe/runs/tar-ms250-repro-5065b15a/examples/tamper-resistance/artifacts/generated/runs/tar-ms250-repro-5065b15a/smoke-model
