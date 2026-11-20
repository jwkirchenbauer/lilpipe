#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute Matrix study root}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1
origin=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
[[ ${SLURM_JOB_ID:-} ]]
if [[ ${SLURM_STEP_GPUS:-} == 0,1,2,3 ]]; then export CUDA_VISIBLE_DEVICES="$SLURM_STEP_GPUS"; fi
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE HIP_VISIBLE_DEVICES
unset DYNAMICS_SOURCE_MANIFEST DYNAMICS_BOOTSTRAP_SEED
export PYTHONPATH="$run/source/transition:$origin/source/segmented-v3/misc:$origin/source/segmented-v3/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export LILPIPE_DEVICE_DEBUG=1 WANDB_MODE=offline AXOLOTL_DO_NOT_TRACK=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export DYNAMICS_ELAPSED_OFFSET=0 DYNAMICS_INNER_OFFSET=0
pilot="$run/pilots/cuda-training"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$pilot/status/worker-failure.json"; if [[ ! -s "$pilot/status/validation.json" ]]; then cp "$pilot/status/worker-failure.json" "$pilot/status/validation.json"; fi; fi' EXIT
python - <<'PY'
import torch
from accelerate import PartialState
from accelerate.utils import DistributedType
state=PartialState()
assert state.distributed_type == DistributedType.NO and state.num_processes == 1
assert torch.__version__ == '2.12.1+cu126' and torch.version.hip is None
assert torch.cuda.device_count() == 4
print('CUDA training pilot: one process and four H100s, original 125-update schedule', flush=True)
PY
export DYNAMICS_SAVE_STEPS=1,2,4 DYNAMICS_STATE_STEPS=4 DYNAMICS_STOP_STEP=4
axolotl train "$pilot/k4/config.yml" --launcher python 2>&1 | tee "$pilot/k4/train.log"
export DYNAMICS_SAVE_STEPS=1 DYNAMICS_STATE_STEPS=1 DYNAMICS_STOP_STEP=1
axolotl train "$pilot/k256/config.yml" --launcher python 2>&1 | tee "$pilot/k256/train.log"
python "$run/source/transition/validate_training_pilot.py" "$run"
