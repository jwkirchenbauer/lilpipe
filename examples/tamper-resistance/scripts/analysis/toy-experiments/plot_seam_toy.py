"""Plot the CPU-only SEAM toy relearning trajectory."""

import argparse
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="artifacts/analysis/seam-toy.json")
    parser.add_argument("--output", default="artifacts/figures/seam-toy.pdf")
    args = parser.parse_args()

    with open(args.input) as handle:
        results = json.load(handle)

    rows = []
    for run in results["runs"]:
        for result in run["trajectory"]:
            rows.extend(
                [
                    {
                        "Relearning step": result["step"],
                        "Accuracy (%)": 100 * result["harmful_accuracy"],
                        "Method": run["method"],
                        "Beta": run["beta"],
                        "Evaluation": "Harmful",
                    },
                    {
                        "Relearning step": result["step"],
                        "Accuracy (%)": 100 * result["utility_accuracy"],
                        "Method": run["method"],
                        "Beta": run["beta"],
                        "Evaluation": "Utility",
                    },
                ]
            )

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Accuracy (%)",
        col="Evaluation",
        hue="Method",
        size="Beta",
        sizes=(1, 4),
        kind="line",
        height=3.2,
        aspect=1.25,
    )
    for axis in grid.axes.flat:
        axis.axhline(100 / 3, color="black", linestyle="--", linewidth=1)

    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    grid.figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
