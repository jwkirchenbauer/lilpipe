#!/usr/bin/env bash
# Run the unchanged 16-update parity-prefix callback on Matrix/CUDA.
set -euo pipefail
[[ $- == *i* ]] || { echo 'Invoke this worker with bash -i.' >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'A Matrix allocation is required.' >&2; exit 2; }
root=/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
unset HIP_VISIBLE_DEVICES LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
export PYTHONPATH="$root/source/parity-matrix/misc:$root/source/parity-matrix/src${PYTHONPATH:+:$PYTHONPATH}"
python - "$root" <<'PY'
import ast
import inspect
from pathlib import Path
import sys
import trainers.tr_tar as tr_tar

root = Path(sys.argv[1])
expected = root / "source/parity-matrix/src/trainers/tr_tar.py"
actual = Path(inspect.getsourcefile(tr_tar)).resolve()
assert actual == expected.resolve(), f"Imported wrong tr_tar module: {actual}"
tree = ast.parse(actual.read_text())
assert any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and isinstance(node.func.value, ast.Name)
    and node.func.value.id == "torch"
    and node.func.attr == "clear_autocast_cache"
    for node in ast.walk(tree)
), "Matched trainer source is missing torch.clear_autocast_cache()"
print("Parity trainer source preflight:", actual, "autocast-cache refresh present", flush=True)
PY
export TAR_RUNTIME_BACKEND=matrix
exec bash -i "$root/source/parity-matrix/misc/run_tar_compute.sh" parity "$root" rio-prefix-new-anchor
