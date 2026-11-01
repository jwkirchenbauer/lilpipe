"""Plot loss-landscapes/get_ll.py CSV grids as 3D surfaces with contour insets."""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--zlim", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    df = pd.read_csv(osp.join(args.indir, "grid.csv"))
    df = df.rename(columns={"alpha": "c0", "beta": "c1"})
    configs = [
        {
            "radius": radius,
            "grid": rows.pivot(index="c1", columns="c0", values="loss"),
        }
        for radius, rows in df.sort_values("radius", ascending=False).groupby(
            "radius", sort=False
        )
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1,
        len(configs),
        figsize=(5 * len(configs), 4),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for cfg, ax in zip(configs, axes.flat, strict=True):
        grid = cfg["grid"]
        c0, c1 = grid.columns, grid.index
        loss = grid.to_numpy()
        low, high = args.zlim if args.zlim else (float(loss.min()), float(loss.max()))
        if low == high:
            padding = max(abs(low), 1.0) * 0.01
            low, high = low - padding, high + padding

        c0, c1 = np.meshgrid(c0, c1)
        ax.plot_surface(c0, c1, loss, cmap="viridis", vmin=low, vmax=high, linewidth=0)
        ax.set(
            title=f"Radius {cfg['radius']:.2e}",
            xlabel="c0",
            ylabel="c1",
            zlabel="Forget loss",
            xticks=grid.columns[::2],
            yticks=grid.index[::2],
            zlim=(low, high),
        )
        ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.6f"))
        ax.zaxis.labelpad = 24
        ax.tick_params(axis="z", labelsize=8)

        contour = ax.inset_axes([0.0, 0.68, 0.3, 0.3])
        contour.contour(
            c0,
            c1,
            loss,
            levels=np.linspace(low, high, 11),
            cmap="viridis",
            vmin=low,
            vmax=high,
        )
        contour.set(xticks=[], yticks=[], xlim=ax.get_xlim(), ylim=ax.get_ylim())

    fig.tight_layout()
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    fig.savefig(args.outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
