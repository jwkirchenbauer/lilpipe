#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
study=${1:?absolute study directory required}
name=${2:?case name required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
folder="$study/cases/$name"
output="$folder/displacement-concentration"
export MPLCONFIGDIR="$output/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$output/status"
exec 9>"$output/status/analysis.lock"
flock -n 9
python "$study/source/concentration-v1/tar_basis_structure.py" concentration_comparison "$folder" 2>&1 | tee "$output/analysis.log"
