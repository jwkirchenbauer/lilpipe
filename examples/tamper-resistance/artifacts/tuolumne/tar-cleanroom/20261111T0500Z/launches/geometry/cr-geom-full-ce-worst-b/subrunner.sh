#!/bin/bash
############################### FLUX to SLURM Spoofing #############################################
export TASKS_PER_NODE=1
export GPUS_PER_TASK=1
export GPUS_PER_NODE=4
export CPUS_PER_TASK=96
export SLURM_CPUS_PER_TASK=$CPUS_PER_TASK

export SLURM_JOB_NODELIST=$(flux getattr hostlist)
export SLURM_NTASKS_PER_NODE=$(( $FLUX_JOB_SIZE / $FLUX_JOB_NNODES ))

# Okay so lightning does actually need this. On flux we don't have this variable
# going to recreate it under the assumption that the verbose hostlist output is *ordered*
# if this is wrong then the nodeIDs will be mixed up. It's not clear whether this would actually
# impact performance but should be fixed if possible.
# export SLURM_NODEID=$(( $(flux hostlist -led'\n' | grep -n $(hostname) | cut -d: -f1) - 1 ))
# the better choice from Nikoli (they agree in my testing, fwiw)
export SLURM_NODEID=$(flux getattr rank)

export SLURM_NNODES=$FLUX_JOB_NNODES 
export SLURM_NTASKS=$FLUX_JOB_SIZE
export SLURM_NTASKS_PER_NODE=$TASKS_PER_NODE
export SLURM_JOB_NUM_NODES=$SLURM_NNODES

# should be same as $TASKS_PER_NODE
if [ $SLURM_NTASKS_PER_NODE -ne $TASKS_PER_NODE ]; then
    echo "Error: SLURM_NTASKS_PER_NODE is not equal to TASKS_PER_NODE"
    exit 1
fi

export SLURM_LOCALID=$FLUX_TASK_LOCAL_ID
export SLURM_PROCID=$FLUX_TASK_RANK

export LOCAL_RANK=$FLUX_TASK_LOCAL_ID
export RANK=$FLUX_TASK_RANK
export WORLD_SIZE=$FLUX_JOB_SIZE

# logic within lightning slurm plugin uses the slurm jobid to set the port ...
# so we are not setting this for now
# export SLURM_JOB_ID=$FLUX_JOB_ID

# show topology and localization descriptors
echo "FLUX_JOB_ID: $FLUX_JOB_ID"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID-}"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "LOCAL_RANK: $LOCAL_RANK"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (will be overwritten with 0,1,2,3)"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES (will be overwritten with 0,1,2,3)"
echo "ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES (will be overwritten with 0,1,2,3)"



############################### Modules and perf enhancing env vars ################################

echo $(date -u) "Loading modules"

rocm_version=7.2.0
source /etc/profile.d/z00_lmod.sh
module swap PrgEnv-cray PrgEnv-gnu
module load rocm/$rocm_version
module load craype-accel-amd-gfx942
module load cray-pmi/6.1.15
module load cray-libpals/1.2.11
module load libfabric/2.1


export PLUGIN_LOC=/collab/usr/global/tools/rccl/toss_4_x86_64_ib_cray/rocm-7.2.0/install/lib
export LD_LIBRARY_PATH="$PLUGIN_LOC${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # point to directory with plugin library

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
export ROCR_VISIBLE_DEVICES=0,1,2,3


# export NCCL_DEBUG=ERROR
export NCCL_DEBUG=WARN
# export NCCL_DEBUG=INFO  # See what the heck is going on

export NCCL_NET_GDR_LEVEL=3
export NCCL_NET_GDR=3 # is this different?

# https://github.com/pytorch/torchtune/issues/1185
# suggests could help with alloc/reserved gap, but no conclusive demo yet
# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

export FI_CXI_ATS=0 # NOTE this is not set by LBANN, but is found elsewhere https://github.com/search?q=FI_CXI_ATS&type=code
export NCCL_CROSS_NIC=1 # NOTE is set by LBANN, consolidated above


# Slingshot plugin specific verbosity
# export FI_LOG_LEVEL=ERROR
# export FI_LOG_LEVEL=WARN
# export FI_LOG_LEVEL=INFO

# controlling interconnect specific environment variables
# potentially not essential at lower node counts?

# helps fix hangs and improve stacktrace, per nikoli/LBANN
ml rccl/fast-env-slows-mpi
    

# warning about MSCCL++ not being avail
export RCCL_MSCCLPP_ENABLE=0

######################################## Environment/torch/python management #######################

# Compile flags
export TORCHINDUCTOR_AUTOTUNE_CACHE=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
# can maybe save memory if that is an issue
# export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/l/ssd/$USER
export TORCH_LOGS="recompiles"

# debugging flags
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1

# Remove ipv4 warnings:
export TORCH_CPP_LOG_LEVEL=ERROR


source ~/.bashrc
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
echo $(date -u) "Activated environment on $FLUX_TASK_RANK"


export PYTHONPATH=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export PYTHONPATH=/opt/rocm-7.2.0/share/amd_smi:$PYTHONPATH
export PYTHONPATH=/opt/rocm-7.2.0/bin/rocm-smi:$PYTHONPATH

cd -- /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance



######################################## Launch ####################################################

echo $(date -u) "Launching python on $FLUX_TASK_RANK..."

bash -i /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/source/profiles/geometry-v2/run_tar_cleanroom.sh geometry /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z --case full-ce-worst --cache-check-ce --add-planes reference fit
