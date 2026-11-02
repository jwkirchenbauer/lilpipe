"""Plot saved CIFAR-10 landscapes with radii as rows and methods as columns."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def plot_landscapes(result, output):
    """Show multiple zoom levels with local loss scales and contour insets."""
    figure, panels = plt.subplots(
        len(result["radii"]),
        len(result["models"]),
        figsize=(3.5 * len(result["models"]), 3.5 * len(result["radii"])),
        squeeze=False,
        subplot_kw={"projection": "3d"},
    )
    for column, record in enumerate(result["models"]):
        for row, grid in enumerate(record["grids"]):
            panel = panels[row, column]
            x, y = np.meshgrid(grid["axes"], grid["axes"])
            losses = np.array(grid["losses"]).T
            low, high = float(losses.min()), float(losses.max())
            if low == high:
                padding = max(abs(low), 1.0) * 0.01
                low, high = low - padding, high + padding
            panel.plot_surface(
                x, y, losses, cmap="viridis", vmin=low, vmax=high, linewidth=0
            )
            center = len(grid["axes"]) // 2
            panel.scatter(0, 0, losses[center, center], color="red", s=15)
            name = (
                record["name"]
                .replace("Random initialization", "Random init")
                .replace(", defended", "")
                .replace(", rho=0.01", "")
            )
            panel.set(title=name if row == 0 else "", zlim=(low, high))
            if column == 0:
                panel.text2D(
                    -0.15,
                    0.5,
                    f"Radius {grid['radius']:,.0f}",
                    transform=panel.transAxes,
                    rotation=90,
                    va="center",
                )
            panel.xaxis.set_major_locator(MaxNLocator(3))
            panel.yaxis.set_major_locator(MaxNLocator(3))
            panel.zaxis.set_major_locator(MaxNLocator(3))
            panel.zaxis.set_major_formatter(FormatStrFormatter("%.2g"))
            panel.zaxis.labelpad = 16
            panel.tick_params(axis="z", labelsize=8)
            inset = panel.inset_axes([0, 0.68, 0.3, 0.3])
            inset.contour(
                x, y, losses, levels=np.linspace(low, high, 11), cmap="viridis"
            )
            inset.set(
                xticks=[], yticks=[], xlim=panel.get_xlim(), ylim=panel.get_ylim()
            )
    figure.suptitle("CIFAR-10 cross-entropy · local loss scales · SAM ρ=0.01")
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output", default="artifacts/figures/resnet18-cifar10-tar-loss-landscapes.pdf"
    )
    args = parser.parse_args()
    with open(args.input) as stream:
        result = json.load(stream)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_landscapes(result, args.output)
