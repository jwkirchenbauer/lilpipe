# Matrix two-GPU landscape pilot

The `TAR_LANDSCAPE_PEER=1` option keeps the evaluation model and BF16
centers on logical CUDA device 0. Logical device 1 holds the two full FP32
directions. The worker copies two FP32 slices per parameter to logical device
0 before evaluating the original perturbation expression. When Slurm
publishes `SLURM_STEP_GPUS`, the entry points require two distinct global
GPU IDs. They always preserve the inherited `CUDA_VISIBLE_DEVICES` mask,
which can use cgroup-relative indices. With no published step IDs they rely
on Slurm's device cgroup. Both modes require Torch to see exactly two
distinct H100s before loading the model.

The pilot uses the archived CUDA checkpoint probe for the RIO K16 learning rate
1e-6 case. It reads the same defended checkpoint, random basis and offsets,
dataset revision and ten fixed documents. Both storage layouts evaluate
the same model instance on GPU 0, with eager attention and TF32 disabled.
The CPU-anchor measurement calls the original frozen worker functions.
After that measurement, the pilot frees the GPU 0 basis and allocator cache,
then restores the original center from the independent CPU anchor.

## Predeclared acceptance gates

- All nine CUDA pairs have finite cross-entropy and absolute differences at
  most 0.0001 nats.
- Every evaluated parameter has the same bytes across the two layouts at
  the warm-up coordinate (-125, 125), verified by matching SHA-256 digests.
- The sum of the nine synchronized perturbation-plus-forward times for the
  peer layout is strictly less than half that of the CPU-anchor layout.
  Setup, warm-up and the parameter digest are outside this timed sum.

The result includes separate setup and warm-up times, end-to-end wall time,
device allocation and reservation peaks, process peak host memory, peer
capability, topology when available, and source and input provenance. The old
CUDA losses are a secondary comparison. A missed gate records
`state=do_not_switch` and exits nonzero.

## Pilot dry run on Matrix

Run from the tamper-resistance example directory on the Matrix login host.
The 224 GiB node memory request is passed to native Slurm through
`SBATCH_MEM_PER_NODE`. Review both generated scripts before submitting the
same command with `--dryrun` removed.

```bash
bash -ic '
set -e
prefix=/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
mm_activate "$prefix"
target="$PWD/artifacts/matrix/tar-rio/20261120T2120Z"
SBATCH_MEM_PER_NODE=224G python /p/vast1/kirchenb/llnl-tools/launch_matrix.py \
    --run_name=tar-rio-peer-landscape-pilot --partition=pbatch --account=verival \
    --nodes=1 --tasks_per_node=1 --cpus_per_task=56 --gpus_per_node=2 --gpus_per_task=2 \
    --exclusive=false --torchrun=false --launch_once_per_node=true \
    --environment="$prefix" --env_act_style=mm_activate --cuda_module= \
    --minutes=45 --output_dir="$target/pilots" --pass_run_name=false \
    --squonce=false --dryrun -- bash -i "$target/source/peer-landscape/pilot.sh"
'
```

The pilot writes
`pilots/tar-rio-peer-landscape-pilot/results-<Slurm job ID>/result.json`.

## Production entry after a passing pilot

Launch `bash -i <peer source>/analysis.sh <absolute study root> <case name>
<absolute pilot result.json>` as one Matrix task with two H100s. The entry
acquires the existing analysis lock and checks the pilot gates, source
hashes, original study settings digest and attack input order.
It uses the original stage, row and coordinate caches and records the runtime
in `status/analysis-peer-<Slurm job ID>.json`. Keep profile jobs on the
existing one-GPU entry point.
