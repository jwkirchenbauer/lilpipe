#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
export CUDA_VISIBLE_DEVICES=0,1,2,3 HIP_VISIBLE_DEVICES=0,1,2,3
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 TAR_CLEANROOM_TEST_MULTI_GPU=1
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
sha256sum scripts/misc/tar_cleanroom_gradients.py ../../tests/test_tar_cleanroom_gradients.py
python -m unittest discover -s ../../tests -p test_tar_cleanroom_gradients.py -k sharded_norm_and_dot_transfer_only_scalar_reductions -v
