#!/bin/bash
set -e

cd -- /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export RANK="${SLURM_PROCID:?SLURM_PROCID is required}"
export LOCAL_RANK="${SLURM_LOCALID:?SLURM_LOCALID is required}"
export WORLD_SIZE="${SLURM_NTASKS:?SLURM_NTASKS is required}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export PYTHONFAULTHANDLER=1
export TORCHINDUCTOR_AUTOTUNE_CACHE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCH_LOGS="recompiles"


export PYTHONPATH=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance"${PYTHONPATH:+:$PYTHONPATH}"

printf 'host=%s job_id=%s node_id=%s nodes=%s tasks_per_node=%s cpus_per_task=%s\n' \
    "$(hostname)" "$SLURM_JOB_ID" "$SLURM_NODEID" "$SLURM_NNODES" \
    "${SLURM_NTASKS_PER_NODE:-${SLURM_TASKS_PER_NODE:-}}" "$SLURM_CPUS_PER_TASK"
printf 'rank=%s local_rank=%s world_size=%s step_gpus=%s cuda_visible_devices=%s\n' \
    "$RANK" "$LOCAL_RANK" "$WORLD_SIZE" "${SLURM_STEP_GPUS-<unset>}" "${CUDA_VISIBLE_DEVICES-<unset>}"
printf 'master_addr=%s master_port=%s pwd=%s\n' "$MASTER_ADDR" "$MASTER_PORT" "$PWD"
exec bash -i /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/source/transition/entry.sh profiles /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio-cuda/20261121T0610Z rio-k64-cuda-lr1e-6
