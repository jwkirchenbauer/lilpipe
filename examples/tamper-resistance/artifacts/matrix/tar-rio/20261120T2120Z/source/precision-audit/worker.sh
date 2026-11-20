#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && ${SLURM_CLUSTER_NAME:-} == matrix ]] || { echo 'A Matrix allocation is required.' >&2; exit 2; }
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
root=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
unset HIP_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
python - <<'PY'
import torch
assert torch.version.hip is None and torch.version.cuda == "12.6"
assert torch.cuda.device_count() == 1
print("Precision audit device:", torch.cuda.get_device_name(0), flush=True)
PY
python "$root/source/precision-audit/precision_audit.py" --run "$root" --device cuda:0 --chunk-elements 1000000
