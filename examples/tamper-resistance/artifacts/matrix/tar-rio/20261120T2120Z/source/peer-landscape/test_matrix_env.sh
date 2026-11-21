#!/usr/bin/env bash
# Test mask selection with synthetic Slurm variables. No Torch or CUDA calls.
set -euo pipefail
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

run_case() {
    bash -ic '
set -euo pipefail
export SLURM_JOB_ID=1 SLURM_STEP_ID=0 SLURM_CLUSTER_NAME=matrix
export SLURM_NTASKS=1 SLURM_PROCID=0 SLURM_LOCALID=0
unset CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS
case "$2" in
    allocated) export SLURM_STEP_GPUS=3,1 CUDA_VISIBLE_DEVICES=0,1 ;;
    allocated-unset) export SLURM_STEP_GPUS=3,1 ;;
    duplicate) export SLURM_STEP_GPUS=3,3 ;;
    single) export SLURM_STEP_GPUS=3 CUDA_VISIBLE_DEVICES=1 ;;
    cgroup-unset) ;;
    cgroup-inherited) export CUDA_VISIBLE_DEVICES=3,1 ;;
    cgroup-empty) export CUDA_VISIBLE_DEVICES= ;;
esac
source "$1"
case "$2" in
    allocated)
        [[ "$TAR_GPU_ASSIGNMENT_MODE" == slurm_step_gpus &&
           "$CUDA_VISIBLE_DEVICES" == 0,1 &&
           "$TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES" == 0,1 ]]
        ;;
    allocated-unset)
        [[ "$TAR_GPU_ASSIGNMENT_MODE" == slurm_step_gpus &&
           ! -v CUDA_VISIBLE_DEVICES && ! -v TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES ]]
        ;;
    cgroup-unset)
        [[ "$TAR_GPU_ASSIGNMENT_MODE" == slurm_cgroup &&
           ! -v CUDA_VISIBLE_DEVICES && ! -v TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES ]]
        ;;
    cgroup-inherited)
        [[ "$TAR_GPU_ASSIGNMENT_MODE" == slurm_cgroup &&
           "$CUDA_VISIBLE_DEVICES" == 3,1 &&
           "$TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES" == 3,1 ]]
        ;;
    cgroup-empty)
        [[ "$TAR_GPU_ASSIGNMENT_MODE" == slurm_cgroup &&
           -v CUDA_VISIBLE_DEVICES && -z "$CUDA_VISIBLE_DEVICES" &&
           -v TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES &&
           -z "$TAR_SLURM_INHERITED_CUDA_VISIBLE_DEVICES" ]]
        ;;
esac
' bash "$source_dir/matrix_env.sh" "$1"
}

for scenario in allocated allocated-unset cgroup-unset cgroup-inherited cgroup-empty; do
    run_case "$scenario"
done
for scenario in duplicate single; do
    rc=0
    run_case "$scenario" || rc=$?
    [[ "$rc" == 2 ]]
done
