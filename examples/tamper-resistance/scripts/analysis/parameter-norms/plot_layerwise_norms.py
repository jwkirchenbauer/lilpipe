"""Plot layerwise L2 norm and RMS for Zephyr-7B parameters."""

import argparse
import json
import os


def tidy_rows(result):
    rows = []
    for layer in result["layers"]:
        for measure, key in (("L2 norm", "l2_norm"), ("RMS", "rms")):
            rows.append(
                {
                    "Layer": layer["name"],
                    "Position": layer["position"],
                    "Measure": measure,
                    "Value": layer[key],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/analysis/z7b-layerwise-norms.json"
    )
    parser.add_argument("--output", default="artifacts/figures/z7b-layerwise-norms.pdf")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    with open(args.input) as handle:
        result = json.load(handle)
    frame = pd.DataFrame(tidy_rows(result))

    sns.set_theme(context="paper", style="whitegrid", font_scale=1.25)
    figure, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for axis, measure in zip(axes, ("L2 norm", "RMS")):
        data = frame[frame["Measure"] == measure]
        sns.lineplot(data=data, x="Position", y="Value", marker="o", ax=axis)
        axis.set_ylabel(measure)
        axis.set_xlabel("")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Layer")

    labels = [
        layer["name"]
        for layer in sorted(result["layers"], key=lambda row: row["position"])
    ]
    positions = sorted(layer["position"] for layer in result["layers"])
    axes[1].set_xticks(positions, labels, rotation=90)
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    main()
