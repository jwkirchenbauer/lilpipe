#!/usr/bin/env bash
set -euo pipefail
run=${1:?absolute hparam run directory required}
[[ "$run" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export MPLBACKEND=Agg OMP_NUM_THREADS=8
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python "$run/source/comparison-v1/aggregate.py" "$run"
