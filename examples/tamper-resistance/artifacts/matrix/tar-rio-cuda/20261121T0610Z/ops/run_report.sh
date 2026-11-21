#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
export TAR_RUNTIME_BACKEND=matrix
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/artifacts/matrix/tar-rio-cuda/20261121T0610Z/cache/report-matplotlib"
mkdir -p "$MPLCONFIGDIR"
unset HIP_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
REPORT_STATUS="$PWD/artifacts/matrix/tar-rio-cuda/20261121T0610Z/status/report.json"
printf '{"state":"running"}\n' > "$REPORT_STATUS"
report_exit() {
    rc=$?
    if (( rc != 0 )); then
        printf '{"state":"failed","exit_code":%s}\n' "$rc" > "$REPORT_STATUS"
    fi
}
trap report_exit EXIT
python artifacts/matrix/tar-rio-cuda/20261121T0610Z/ops/finish_report.py
