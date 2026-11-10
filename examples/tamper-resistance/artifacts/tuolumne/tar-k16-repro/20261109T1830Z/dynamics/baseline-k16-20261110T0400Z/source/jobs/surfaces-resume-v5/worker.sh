#!/usr/bin/env bash
set -euo pipefail
study=${1:?study directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$study/source/misc:$study/source/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
trap 'result=$?; if (( result != 0 )); then printf "{\"stage\":\"surfaces\",\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$study/status/surfaces.json"; fi' EXIT
printf '{"stage":"surfaces","state":"running"}\n' > "$study/status/surfaces.json"
python "$study/source/jobs/surfaces-resume-v5/tar_dynamics.py" surfaces "$study" 2>&1 | tee "$study/logs/surfaces-resumed.log"
printf '{"stage":"surfaces","state":"complete"}\n' > "$study/status/surfaces.json"
