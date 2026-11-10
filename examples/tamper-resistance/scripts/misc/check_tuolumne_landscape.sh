#!/usr/bin/env bash
# Verify common nonzero coordinates across radius runs, then compare scales.
set -euo pipefail
landscape=${1:?absolute landscape directory required}
check_dir=${2:?absolute diagnostic directory required}
[[ "$landscape" == /* && "$check_dir" == /* ]]
mm_activate /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
cd /p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance
export HF_HOME=/p/vast1/kirchenb/tar-root/cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=0 TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MPLBACKEND=Agg MPLCONFIGDIR="$check_dir/cache/matplotlib"
unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE
mkdir -p "$MPLCONFIGDIR"
trap 'result=$?; if (( result != 0 )); then printf "{\"state\":\"failed\",\"exit_code\":%d}\n" "$result" > "$check_dir/status.json"; fi' EXIT
printf '%s\n' '{"state":"running"}' > "$check_dir/status.json"
test ! -e "$check_dir/probe/grid.csv"
model=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$landscape/provenance.json")

# A 3x3 radius-25 grid lies on the saved radius-100 grid. A radius-10
# grid shares its center and corners with the separately measured zoom.
python "$check_dir/source/get_ll.py" \
    --model "$model" --trajectory "$model" \
    --dataset cais/wmdp-bio-forget-corpus --radii 25 10 --n-pts 3 \
    --batch-size 2 --limit 10 --max-seq-len 512 --seed 42 \
    --outdir "$check_dir/probe"

python - "$landscape" "$check_dir" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

landscape, out = map(Path, sys.argv[1:])
grid = pd.read_csv(landscape / "grid.csv")
probe = pd.read_csv(out / "probe/grid.csv")
assert len(probe) == 18
comparisons = []
for row in probe.itertuples():
    reference_radius = 100 if row.radius == 25 else 10
    match = grid[(grid.radius == reference_radius) & (grid.c0 == row.c0) & (grid.c1 == row.c1)]
    assert len(match) == 1
    expected = float(match.loss.iloc[0])
    difference = abs(row.loss - expected)
    assert difference < 1e-5, (reference_radius, row.c0, row.c1, difference)
    comparisons.append({"reference_radius": reference_radius, "c0": row.c0,
                        "c1": row.c1, "saved_loss": expected,
                        "recomputed_loss": row.loss, "difference": difference})
zoom = grid[grid.radius == 10]
design = np.column_stack([np.ones(len(zoom)), zoom.c0, zoom.c1])
coefficients = np.linalg.lstsq(design, zoom.loss.to_numpy(), rcond=None)[0]
residual = zoom.loss.to_numpy() - design @ coefficients
fit_r2 = 1 - float(np.sum(residual ** 2) / np.sum((zoom.loss - zoom.loss.mean()) ** 2))
centers = grid[(grid.c0 == 0) & (grid.c1 == 0)]
summary = {
    "state": "passed", "recomputed_points": len(comparisons),
    "maximum_recomputed_difference": max(row["difference"] for row in comparisons),
    "center_loss_by_radius": {str(row.radius): row.loss for row in centers.itertuples()},
    "loss_range_by_radius": {
        str(radius): {"min": float(rows.loss.min()), "max": float(rows.loss.max()),
                      "span": float(rows.loss.max() - rows.loss.min())}
        for radius, rows in grid.groupby("radius")
    },
    "radius10_plane_fit": {"intercept": float(coefficients[0]),
                           "slope_c0": float(coefficients[1]),
                           "slope_c1": float(coefficients[2]), "r_squared": fit_r2},
    "shared_zlim": [8.9, 11.5], "comparisons": comparisons,
}
(out / "centering-check.json").write_text(json.dumps(summary, indent=2) + "\n")
grid[grid.radius.isin([100, 10, 1])].to_csv(out / "grid.csv", index=False)
print(json.dumps({key: value for key, value in summary.items() if key != "comparisons"}, indent=2), flush=True)
PY
for extension in png pdf; do
    python "$check_dir/source/plot_ll_3d.py" --indir "$check_dir" \
        --zlim 8.9 11.5 --outpath "$check_dir/shared-scale.$extension"
done
printf '%s\n' '{"state":"complete"}' > "$check_dir/status.json"
