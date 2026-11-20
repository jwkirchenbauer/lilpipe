# Matrix evaluation runtime

`eval/pyproject.toml` and `eval/uv.lock` provide a CUDA 12.6 counterpart to
`runtime/tuolumne/eval`. They preserve all 102 common non-Torch package versions
in the Tuolumne lock. PyTorch changes from `2.12.1+rocm7.2` to `2.12.1+cu126`;
CUDA libraries and Triton replace the ROCm runtime. CUDA-specific inherited
constraints were removed so the CUDA wheel selects its compatible libraries.
This environment covers evaluation and landscape analysis; it does not include
the Axolotl training environment.

Follow the root `AGENTS.md` and `/p/vast1/kirchenb/llnl-tools/README.md` for
installation. Creating a new prefix requires explicit user authorization.
Use the maintainer's fresh-prefix Micromamba+uv installer, adapt its project
directory to `runtime/matrix/eval`, activate absolute prefixes inside interactive
Bash with `set -e`, and sync with `uv sync --locked`. Existing prefixes must
remain independent. Use `UV_SYSTEM_CERTS=true` for the cluster certificates.

The proposed prefix is
`/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1`. The first assessment and
prepared installer are recorded under
`artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-assessment`.
As of preparation on 2026-11-20, this prefix has not been installed.

Launch on Matrix through `/p/vast1/kirchenb/llnl-tools/launch_matrix.py`, using
`pbatch` and account `verival`. Its default CUDA module is unavailable on Matrix;
use `--cuda_module=` for the wheel-provided CUDA runtime. Review a dry run first.
Specify host memory explicitly: the partition defaults to all node memory.

The frozen TAR wrappers still assume the Tuolumne environment and Flux. They
need an explicit Slurm/CUDA adaptation before production use. Validate a saved
checkpoint against recorded Tuolumne losses, preserve the saved visualization
directions, and measure H100 memory use before moving workloads. Changing the
training backend also changes the experiment; bitwise replay checks remain on
Tuolumne unless separately validated. Queue estimates and starter probes do not
establish model-level equivalence or production throughput.

Landscape settings can enable `cpu_landscape_anchor: true` to keep saved
centers on the CPU. Perturbations copy each center tensor to the model's device
before the original FP32 arithmetic. This avoids holding both centers and two
full FP32 directions in H100 memory. The default retains device-resident
centers. The current Tuolumne jobs use their unchanged frozen source.
