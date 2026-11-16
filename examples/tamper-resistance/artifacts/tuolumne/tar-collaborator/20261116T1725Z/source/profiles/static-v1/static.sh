#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
run=${1:?absolute study directory required}
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
code_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0 AMD_COMGR_CACHE=0
export TORCHAO_FORCE_SKIP_LOADING_SO_FILES=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
out="$run/static"
mkdir -p "$out/cache/matplotlib"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$run/status/static.json"; fi' EXIT
printf '{"state":"running"}\n' > "$run/status/static.json"
export MPLCONFIGDIR="$out/cache/matplotlib"
jq -e '.state=="passed"' "$run/input-validation.json" >/dev/null
model=$(jq -r '.supplied_checkpoint.path' "$run/study.json")
python - "$run" "$code_dir" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
import torch
from datasets import load_dataset
run, code = map(Path, sys.argv[1:])
study = json.loads((run / "study.json").read_text())
probes = json.loads((run / "probes.json").read_text())
original = Path(study["original_run"]) / "loss-landscape/defense-step125-random-seed42-v3"
prior = json.loads((original / "provenance.json").read_text())
assert torch.cuda.device_count() == 1
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
corpus = load_dataset("cais/wmdp-bio-forget-corpus", split="train", revision=probes["forget_revision"])
hashes = [hashlib.sha256(corpus[i]["text"].encode()).hexdigest() for i in probes["fixed_source_indices"]]
assert hashes == prior["document_sha256"]
source = hashlib.sha256((code / "get_ll.py").read_bytes()).hexdigest()
assert source == prior["source_sha256"]["get_ll.py"]
assert torch.__version__ == prior["torch"]
result = {
    "checkpoint": study["supplied_checkpoint"], "original_static": str(original),
    "radii": [1000, 500, 200, 125, 100, 10, 1, 0.1], "n_pts": 9,
    "seed": 42, "probe_documents": 10, "batch_size": 2, "max_seq_len": 512,
    "document_sha256": hashes, "dataset_revision": probes["forget_revision"],
    "get_ll_source_sha256": source, "torch": torch.__version__, "hip": torch.version.hip,
    "gpu": torch.cuda.get_device_name(0), "flux_job_id": os.environ["FLUX_JOB_ID"],
    "convention": "Original seed42 globally normalized BF16 random directions, original perturbation arithmetic and default attention. Dynamic views separately use shared FP32 orthonormal bases and eager attention.",
}
(run / "static/provenance.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
PY
python "$code_dir/get_ll.py" --model "$model" --trajectory "$model" \
    --dataset cais/wmdp-bio-forget-corpus --radii 1000 500 200 125 100 10 1 0.1 \
    --n-pts 9 --batch-size 2 --limit 10 --max-seq-len 512 --seed 42 --outdir "$out"
python "$code_dir/plot_ll_3d.py" --indir "$out" --outpath "$out/landscape-3d.png"
python "$code_dir/plot_ll_3d.py" --indir "$out" --outpath "$out/landscape-3d.pdf"
python "$code_dir/compare.py" "$run"
printf '{"state":"complete"}\n' > "$run/status/static.json"
