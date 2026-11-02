"""Plot a saved PyHessian density estimate on linear and logarithmic scales."""

import argparse
import json
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir", required=True)
    parser.add_argument("--outpath", required=True)
    args = parser.parse_args()
    with open(osp.join(args.indir, "quadrature.json")) as handle:
        metadata = json.load(handle)
    data = np.loadtxt(osp.join(args.indir, "density.csv"), delimiter=",", skiprows=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for axis, scale in zip(axes, ("linear", "log"), strict=True):
        axis.plot(data[:, 0], data[:, 1], color="#4477AA")
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set(
            xlabel="Hessian eigenvalue",
            ylabel="Estimated spectral density",
            yscale=scale,
            title=f"{scale.capitalize()} density scale",
        )
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(
        f"TAR archived checkpoint · {metadata['documents']} documents · "
        f"{metadata['lanczos_steps']} Lanczos steps × "
        f"{len(metadata['eigenvalues'])} probes · FP32",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    fig.savefig(args.outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
