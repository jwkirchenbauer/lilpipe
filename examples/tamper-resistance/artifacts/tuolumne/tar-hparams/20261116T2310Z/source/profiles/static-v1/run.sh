#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
code=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
sha256sum --quiet -c "$code/manifest.sha256"
sha256sum --quiet -c /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/operational-inputs.sha256
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mode=${1:?case or report}
shift
mkdir -p /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/static/cache
export MPLCONFIGDIR=$(mktemp -d /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/static/cache/matplotlib-XXXXXXXX)
case "$mode" in
  case)
    export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
    exec python "$code/static_worker.py" case "$@"
    ;;
  report)
    export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
    exec python "$code/static_report.py" "$@"
    ;;
  *) exit 2 ;;
esac
