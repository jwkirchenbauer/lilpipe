#!/usr/bin/env bash
# Run with bash -i through launch_tuo.py.
set -euo pipefail
run_dir=${1:?absolute reproduction directory required}
out_dir=${2:?absolute landscape directory required}
reuse_dir=${3:-}
[[ "$run_dir" == /* && "$out_dir" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MPLBACKEND=Agg MPLCONFIGDIR="$out_dir/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
test ! -e "$out_dir/grid.csv"
mkdir -p "$MPLCONFIGDIR"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$out_dir/status.json"; fi' EXIT
printf '%s\n' '{"state":"running"}' > "$out_dir/status.json"

python - "$run_dir" "$out_dir" "$reuse_dir" <<'PY'
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset

run, out = map(Path, sys.argv[1:3])
reuse = Path(sys.argv[3]) if sys.argv[3] else None
state = json.loads((run / "models/defense/checkpoint-125/trainer_state.json").read_text())
assert state["global_step"] == 125
visibility = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
print("Landscape GPU visibility:", visibility, "device count:", torch.cuda.device_count(), flush=True)
assert torch.cuda.device_count() == 1, visibility
assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
data = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
selected = list(itertools.islice((row["text"] for row in data if len(row["text"]) > 50), 10))
inputs = json.loads((run / "inputs.json").read_text())
info = {
    "checkpoint": str(run / "models/defense"),
    "defense_step": 125,
    "dataset": "cais/wmdp-bio-forget-corpus",
    "dataset_revision": inputs["datasets"]["cais/wmdp-bio-forget-corpus"]["revision"],
    "dataset_fingerprint": data._fingerprint,
    "document_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in selected],
    "limit": 10,
    "batch_size": 2,
    "max_seq_len": 512,
    "radii": [1000, 500, 200, 125, 100, 10, 1, 0.1],
    "computed_radii": [10, 1] if reuse else [1000, 500, 200, 125, 100, 10, 1, 0.1],
    "reused_from": str(reuse) if reuse else None,
    "n_pts": 9,
    "direction_seed": 42,
    "directions": "two independent random directions, global L2 normalization in BF16",
    "loss": "mean of batch causal-language-model cross-entropies; padding masked",
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "blas_backend": str(torch.backends.cuda.preferred_blas_library()),
    "flux_job_id": os.environ.get("FLUX_JOB_ID"),
    "reference_recipe": "scripts/misc/analysis_gen_ll_tar_archived_zoomout.sbatch plus the r125 recipe",
    "source_sha256": {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in (out / "source").glob("*") if file.is_file()
    },
}
if reuse:
    previous = json.loads((reuse / "provenance.json").read_text())
    for key in ("checkpoint", "dataset", "dataset_revision", "dataset_fingerprint",
                "document_sha256", "limit", "batch_size", "max_seq_len", "n_pts",
                "direction_seed", "directions", "torch", "hip", "blas_backend"):
        assert info[key] == previous[key], f"Incompatible landscape setting: {key}"
    for name in ("get_ll.py", "plot_ll_3d.py", "plot_ll_2d.py"):
        assert info["source_sha256"][name] == previous["source_sha256"][name], name
(out / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2), flush=True)
PY

if [[ -n "$reuse_dir" ]]; then
    compute_dir="$out_dir/new-radii"
    mkdir "$compute_dir"
    grid_radii=(10 1)
else
    compute_dir="$out_dir"
    grid_radii=(1000 500 200 125 100 10 1 0.1)
fi
python "$out_dir/source/get_ll.py" \
    --model "$run_dir/models/defense" \
    --trajectory "$run_dir/models/defense" \
    --dataset cais/wmdp-bio-forget-corpus \
    --radii "${grid_radii[@]}" \
    --n-pts 9 --batch-size 2 --limit 10 --max-seq-len 512 --seed 42 \
    --outdir "$compute_dir"

python - "$out_dir" "$reuse_dir" <<'PY'
import json
import math
import sys
from pathlib import Path

import pandas as pd

out = Path(sys.argv[1])
if sys.argv[2]:
    reuse = Path(sys.argv[2])
    added = pd.read_csv(out / "new-radii/grid.csv")
    assert len(added) == 162 and set(added.radius) == {10, 1}
    previous = pd.read_csv(reuse / "grid.csv")
    kept = previous[previous.radius.isin([1000, 500, 200, 125, 100, 0.1])]
    assert len(kept) == 486
    grid = pd.concat([kept, added], ignore_index=True)
    grid.sort_values(["radius", "c0", "c1"], ascending=[False, True, True]).to_csv(out / "grid.csv", index=False)
    pd.read_csv(out / "new-radii/trajectory.csv").to_csv(out / "trajectory.csv", index=False)
grid = pd.read_csv(out / "grid.csv")
assert len(grid) == 648, len(grid)
assert set(grid.radius) == {1000, 500, 200, 125, 100, 10, 1, 0.1}
assert not grid.duplicated(["radius", "c0", "c1"]).any()
assert grid.groupby("radius").size().eq(81).all()
assert grid.loss.map(math.isfinite).all()
center = grid[(grid.c0 == 0) & (grid.c1 == 0)]
assert len(center) == 8
assert center.loss.max() - center.loss.min() < 1e-5
trajectory = pd.read_csv(out / "trajectory.csv")
assert len(trajectory) == 1
assert trajectory.loc[0, ["c0", "c1"]].eq(0).all()
for name, radii in (("overview", [1000, 500, 200, 125, 100, 10, 1, 0.1]), ("radius-125", [125])):
    directory = out / name
    directory.mkdir()
    grid[grid.radius.isin(radii)].to_csv(directory / "grid.csv", index=False)
    trajectory.to_csv(directory / "trajectory.csv", index=False)
summary = {
    "state": "passed", "points": len(grid), "center_loss": float(center.loss.iloc[0]),
    "loss_range_by_radius": {
        str(radius): {"min": float(rows.loss.min()), "max": float(rows.loss.max())}
        for radius, rows in grid.groupby("radius")
    },
}
(out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY

for view in "$out_dir" "$out_dir/overview" "$out_dir/radius-125"; do
    for extension in pdf png; do
        python "$out_dir/source/plot_ll_3d.py" --indir "$view" \
            --outpath "$view/landscape-3d.$extension"
    done
done
python "$out_dir/source/plot_ll_2d.py" --indir "$out_dir/radius-125" \
    --outpath "$out_dir/radius-125/landscape-2d.pdf"
printf '%s\n' '{"state":"complete"}' > "$out_dir/status.json"
