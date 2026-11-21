#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=2
#SBATCH --time=360
#SBATCH --job-name=tar-rio-cuda-analysis-rio-k64-cuda-lr1e-6
#SBATCH --partition=pbatch
#SBATCH --account=verival
#SBATCH --output=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z/launches/analysis-rio-k64-cuda-lr1e-6/logs/matrix-%j.out
#SBATCH --error=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z/launches/analysis-rio-k64-cuda-lr1e-6/logs/matrix-%j.out

set -e
source ~/.bashrc || true
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1


cd -- /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${hosts%%$'\n'*}"
: "${MASTER_ADDR:?No hosts found in the Slurm allocation}"
PORT_PYTHON=$(command -v python3 || command -v python)
MASTER_PORT=$("$PORT_PYTHON" -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
export MASTER_PORT

export OUTPUT_DIR=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z/launches/analysis-rio-k64-cuda-lr1e-6
export LOGDIR="$OUTPUT_DIR/logs"
export RUN_NAME=tar-rio-cuda-analysis-rio-k64-cuda-lr1e-6
export SLURM_EXPORT_ENV=ALL

printf 'Job %s started on %s at %s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$(date -Is)"
exec srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=56 --distribution=block:block --unbuffered --kill-on-bad-exit=1 --gpus-per-node=2 --gpus-per-task=2 --label /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z/launches/analysis-rio-k64-cuda-lr1e-6/launch-zem3x6y2/subrunner.sh
