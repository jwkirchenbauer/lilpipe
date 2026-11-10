"""Plot the loss grids saved by loss-landscapes/get_ll.py as 2D contours."""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--zlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.indir, "grid.csv"))
    # Earlier runs used alpha/beta for the same grid coefficients.
    df = df.rename(columns={"alpha": "c0", "beta": "c1"})
    trajectory_path = osp.join(args.indir, "trajectory.csv")
    trajectory = (
        pd.read_csv(trajectory_path).rename(columns={"alpha": "c0", "beta": "c1"})
        if osp.exists(trajectory_path)
        else None
    )
    configs = [
        {
            "radius": radius,
            "grid": rows.pivot(index="c1", columns="c0", values="loss"),
        }
        for radius, rows in df.sort_values("radius", ascending=False).groupby(
            "radius", sort=False
        )
    ]

    sns.set_theme(context="paper", style="ticks", font_scale=1.2)
    fig, axes = plt.subplots(
        1,
        len(configs),
        figsize=(5 * len(configs), 4.5),
        squeeze=False,
        layout="constrained",
    )
    for cfg, ax in zip(configs, axes.flat, strict=True):
        grid = cfg["grid"]
        c0, c1 = grid.columns, grid.index
        loss = grid.to_numpy()
        low, high = args.zlim if args.zlim else (float(loss.min()), float(loss.max()))
        if low == high:
            padding = max(abs(low), 1.0) * 0.01
            low, high = low - padding, high + padding
        filled = ax.contourf(
            c0,
            c1,
            loss,
            levels=np.linspace(low, high, 11),
            cmap="viridis",
            vmin=low,
            vmax=high,
        )
        ax.contour(
            c0,
            c1,
            loss,
            levels=filled.levels,
            colors="black",
            linewidths=0.5,
            alpha=0.5,
        )
        if trajectory is not None:
            ax.scatter(trajectory["c0"], trajectory["c1"], color="red", s=16, zorder=3)
        fig.colorbar(filled, ax=ax, label="Forget loss", shrink=0.8, format="%.6f")
        ax.set(
            title=f"Radius {cfg['radius']:.2e}",
            xlabel="c0",
            ylabel="c1",
            aspect="equal",
        )

    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    fig.savefig(args.outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
