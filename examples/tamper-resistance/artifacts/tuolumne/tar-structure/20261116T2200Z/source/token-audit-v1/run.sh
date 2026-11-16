#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${FLUX_JOB_ID:-}" ]]
run=${1:?absolute structure-study directory required}
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$run/token-audit/cache/matplotlib" "$run/logs" "$run/status"
export MPLCONFIGDIR="$run/token-audit/cache/matplotlib"
exec 9>"$run/status/token-audit.lock"
flock -n 9
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$run/status/token-audit.json"; fi' EXIT
printf '{"state":"running"}\n' > "$run/status/token-audit.json"
python "$code_dir/audit_tokens.py" "$run" 2>&1 | tee "$run/logs/token-audit-values.log"
python "$code_dir/audit_tokens.py" "$run" --report-only 2>&1 | tee "$run/logs/token-audit-report.log"
printf '{"state":"complete"}\n' > "$run/status/token-audit.json"
