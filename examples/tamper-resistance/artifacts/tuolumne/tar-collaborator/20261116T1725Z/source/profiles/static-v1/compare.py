"""Compare the original static recipe with identical per-radius plotting scales."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

run = Path(sys.argv[1])
out = run / "static"
metadata = json.loads((out / "provenance.json").read_text())
old = pd.read_csv(Path(metadata["original_static"]) / "grid.csv")
new = pd.read_csv(out / "grid.csv")
assert len(new) == 648 and new.loss.map(np.isfinite).all()
assert not new.duplicated(["radius", "c0", "c1"]).any()
assert new.groupby("radius").size().eq(81).all()
centers = new[(new.c0 == 0) & (new.c1 == 0)]
assert len(centers) == 8 and centers.loss.max() - centers.loss.min() < 1e-5
assert len(pd.read_csv(out / "trajectory.csv")) == 1
for name, radii in (
    ("large-radii", [1000, 500, 200, 125, 100]),
    ("small-radii", [10, 1, 0.1]),
):
    fig, axes = plt.subplots(
        2,
        len(radii),
        figsize=(5 * len(radii), 9),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for column, radius in enumerate(radii):
        before, supplied = old[old.radius == radius], new[new.radius == radius]
        low = min(before.loss.min(), supplied.loss.min())
        high = max(before.loss.max(), supplied.loss.max())
        pad = max((high - low) * 0.05, 1e-4)
        low, high = low - pad, high + pad
        for row, (label, values) in enumerate(
            (("Local K16", before), ("Collaborator", supplied))
        ):
            grid = values.pivot(index="c1", columns="c0", values="loss")
            assert grid.shape == (9, 9) and np.isfinite(grid.values).all()
            x, y = np.meshgrid(grid.columns, grid.index)
            axis = axes[row, column]
            axis.plot_surface(x, y, grid.values, cmap="viridis", vmin=low, vmax=high)
            axis.view_init(elev=27, azim=-58)
            axis.set(
                xlabel="Random coordinate 1",
                ylabel="Random coordinate 2",
                zlabel="Forget CE (nats)",
                zlim=(low, high),
                title=f"{label} | half-width {radius:g}\ncenter CE {float(grid.loc[0, 0]):.4f}",
            )
    fig.suptitle(
        "Original static recipe | same axes and common CE scales within each radius",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(
            out / f"comparison-{name}.{extension}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)
validation = {
    "state": "passed",
    "points": len(new),
    "center_ce": float(centers.loss.iloc[0]),
    "center_maximum_disagreement": float(centers.loss.max() - centers.loss.min()),
    "common_scales": "Within each radius across the two checkpoint rows; radii have separate scales to preserve small-scale relief.",
    "loss_range_by_radius": {
        str(radius): {"min": float(rows.loss.min()), "max": float(rows.loss.max())}
        for radius, rows in new.groupby("radius")
    },
    "artifacts": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.iterdir())
        if path.suffix in (".csv", ".png", ".pdf")
    },
}
(out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps(validation, indent=2), flush=True)
