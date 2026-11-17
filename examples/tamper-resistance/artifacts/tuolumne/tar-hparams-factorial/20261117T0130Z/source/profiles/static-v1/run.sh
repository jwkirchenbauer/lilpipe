#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams-factorial/20261117T0130Z
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
(cd "$code" && sha256sum --quiet -c manifest.sha256)
(cd "$run" && sha256sum --quiet -c operational-inputs.sha256)
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mode=${1:?case or inputs required}
shift
mkdir -p "$run/static/cache"
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d "$run/static/cache/matplotlib-XXXXXXXX")
case "$mode" in
  case|inputs)
    name=${1:?case name required}
    case "$name" in hparam-ri-defense|hparam-ro-defense|hparam-io-defense) ;; *) exit 2 ;; esac
    export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
    exec python "$code/static_worker.py" "$mode" "$run" "$name"
    ;;
  *) exit 2 ;;
esac
