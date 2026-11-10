# Tuolumne runtimes

The `train` and `eval` projects preserve the example's dependency versions while
using PyTorch 2.12.1 for ROCm 7.2 and Triton ROCm 3.7.1. The training project also
provides Black for the repository's formatting check.

Axolotl requires bitsandbytes and TorchAO imports during trainer setup, including
for ordinary BF16 training. Keep both dependencies. TorchAO uses its Python
wheel; the job wrapper sets `TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1`. The selected
configs use ordinary PyTorch AdamW. CUDA xformers and the unused linear-attention
packages are excluded. The Triton Python namespace comes from `triton-rocm`.

Training uses the named configs' explicit four-GPU layer map in one process,
with `torchrun=false`. The wrapper clears the launcher's distributed-rank shim
before starting Axolotl and verifies Accelerate sees one process and four GPUs.
It sets `TORCH_BLAS_PREFER_CUBLASLT=0` and the equivalent HIP alias to select
hipBLAS/rocBLAS: this build stalled in hipBLASLt's kernel search during backward.
`LILPIPE_DEVICE_DEBUG=1` checks parameter placement against the layer map and logs
the loaded model, first batch, and copied attack model.

Follow the root `AGENTS.md` and `/p/vast1/kirchenb/llnl-tools/README.md` for
environment creation. Use the locked manifests with the maintainer's fresh-prefix
installer, interactive Bash, and absolute-prefix `mm_activate`. Set
`UV_SYSTEM_CERTS=true` so uv uses the cluster's certificate authorities.

`scripts/misc/prepare_tuolumne_tar.py RUN_DIRECTORY` preserves these manifests and
creates a run-local installer copy that activates the new prefix before running
its Python commands. It also freezes the training source and evaluation tasks.
The current run's installation logs and provenance are in its artifact directory.
