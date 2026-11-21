#!/usr/bin/env bash
# Render the final Matrix RIO report only after its declared inputs validate.
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }

TARGET=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
PROJECT=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
WORKER="$TARGET/source/report-matrix-v2"
EVAL_PREFIX=/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1
RUN=${1:-$TARGET}

[[ "$RUN" == "$TARGET" ]] || { echo "This worker is pinned to $TARGET" >&2; exit 2; }
[[ -f "$RUN/study.json" && -f "$RUN/control-study/base-reference/validation.json" ]] || {
    echo 'The Matrix study or fresh CUDA base-reference validation is missing.' >&2
    exit 2
}

mm_activate "$EVAL_PREFIX"
cd "$PROJECT"
export TAR_RUNTIME_BACKEND=matrix
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$RUN/cache/matplotlib-rio-final"
mkdir -p "$MPLCONFIGDIR"
unset HIP_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
python "$WORKER/finish_report.py" "$RUN"
