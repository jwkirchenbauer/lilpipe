#!/usr/bin/env bash
set -euo pipefail
mode=${1:?dryrun or submit}
mm_activate /usr/workspace/wsb/kirchenb/matrix-torch-cu126-demo
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
run=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z
export SBATCH_MEM_PER_NODE=32G
extra=()
if [[ "$mode" == dryrun ]]; then
    extra=(--dryrun)
elif [[ "$mode" != submit ]]; then
    exit 2
fi
python /p/vast1/kirchenb/llnl-tools/launch_matrix.py \
    --run_name=tar-rio-matrix-gpu-probe --minutes=3 \
    --partition=pbatch --account=verival --nodes=1 \
    --gpus_per_node=1 --tasks_per_node=1 --exclusive=false \
    --torchrun=false --launch_once_per_node=true \
    --environment=/usr/workspace/wsb/kirchenb/matrix-torch-cu126-demo \
    --env_act_style=mm_activate --cuda_module= \
    --output_dir="$run/launches" --sub_output_dir_name=matrix-assessment \
    --pass_run_name=false --quiet_launch=true --squonce=false \
    "${extra[@]}" -- bash -i "$run/source/matrix-assessment/probe.sh" gpu
