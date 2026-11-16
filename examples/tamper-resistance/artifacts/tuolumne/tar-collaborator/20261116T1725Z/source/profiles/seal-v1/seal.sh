#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
run=${1:?absolute run directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
python "$code_dir/seal.py" "$run"
