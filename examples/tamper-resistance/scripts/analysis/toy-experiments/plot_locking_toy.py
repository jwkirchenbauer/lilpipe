"""Plot the penalty-based locking toy (DLR-Lock, Sakamoto et al., 2026, Fig. 2)."""

import argparse
import json
import os
import os.path as osp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

panels = [
    (
        "distance",
        "relative_distance",
        r"$\Omega = \|\Delta\theta\|^2$",
        r"$\|\Delta\theta\| / \|\theta_0\|$",
    ),
    (
        "hessian",
        "relative_hessian_trace",
        r"$\Omega = \mathrm{Tr}(H_{CE})$",
        "Relative Hessian trace",
    ),
]


def ramp(name, low=0.3):
    # Single-hue sequential ramp; skip the near-white end so light steps stay visible.
    return mpl.colors.LinearSegmentedColormap.from_list(
        name, plt.get_cmap(name)(np.linspace(low, 1.0, 256))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/analysis/locking-toy.json")
    parser.add_argument(
        "--projected",
        help="Optional --optimizer projected results to overlay, labelled by epsilon.",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Plot Omega of the rebalanced network canon(theta).",
    )
    parser.add_argument("--output", default="artifacts/figures/locking-toy.pdf")
    args = parser.parse_args()

    with open(args.input) as handle:
        results = json.load(handle)
    projected = []
    if args.projected:
        with open(args.projected) as handle:
            projected = json.load(handle)["runs"]

    log_lambdas = np.log10(results["lambdas"])
    norm = mpl.colors.Normalize(log_lambdas.min(), log_lambdas.max())
    cmap = ramp("Blues")
    tolerances = sorted({run["tolerance"] for run in projected})
    tolerance_colors = dict(
        zip(tolerances, ramp("Oranges", 0.5)(np.linspace(0, 1, len(tolerances))))
    )

    figure, axes = plt.subplots(1, len(panels), figsize=(10, 4), layout="constrained")
    for axis, (penalty, key, title, ylabel) in zip(axes, panels):
        if args.canonical:
            key, ylabel = (
                f"canonical_{key}",
                f"Canonical {ylabel[0].lower()}{ylabel[1:]}",
            )
        runs = [run for run in results["runs"] if run["penalty"] == penalty]
        # Thin lines first so large-lambda trajectories sit on top.
        for run in sorted(runs, key=lambda run: run["lambda"]):
            color = cmap(norm(np.log10(run["lambda"])))
            x = [point["relative_mse"] for point in run["trajectory"]]
            y = [point[key] for point in run["trajectory"]]
            axis.plot(x, y, color=color, linewidth=1, alpha=0.8)
            axis.plot(x[0], y[0], "o", color=color, markersize=4)
            axis.plot(x[-1], y[-1], "s", color=color, markersize=4)

        for run in [run for run in projected if run["penalty"] == penalty]:
            color = tolerance_colors[run["tolerance"]]
            x = [point["relative_mse"] for point in run["trajectory"]]
            y = [point[key] for point in run["trajectory"]]
            axis.plot(x, y, color=color, linewidth=2, zorder=4)
            axis.plot(x[-1], y[-1], "s", color=color, markersize=6, zorder=4)
            axis.annotate(
                rf"$\epsilon$={run['tolerance']:.0e}",
                (x[-1], y[-1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="0.2",
            )

        # Symmetry solutions have exactly zero MSE, so pin them to the left edge.
        # Rebalancing maps them back onto theta_0, so they vanish on canonical axes.
        if not args.canonical:
            floor = min(
                point["relative_mse"] for run in runs for point in run["trajectory"]
            )
            floor /= 10
            symmetry = results["symmetry_solutions"]
            axis.plot(
                [floor] * len(symmetry),
                [point[key] for point in symmetry],
                "*",
                color="#d62728",
                markersize=11,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=r"$(a^{-1}W_1, aW_2)$, MSE $= 0$",
                zorder=5,
            )
            axis.legend(loc="lower right", frameon=False, fontsize=8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Relative MSE")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="major", color="0.9", linewidth=0.6)

    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02
    )
    colorbar.set_label(r"$\log_{10}\lambda$ (penalized SGD)")

    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
