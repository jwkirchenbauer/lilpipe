#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
run=${1:?absolute run directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg MPLCONFIGDIR="$run/cache/replay-matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$run/status" "$run/cache/replay-matplotlib"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$run/status/replay.json"; fi' EXIT
printf '{"state":"running"}\n' > "$run/status/replay.json"
python "$code_dir/replay.py" "$run" 2>&1 | tee "$run/logs/module-replay.log"
printf '{"state":"complete"}\n' > "$run/status/replay.json"
