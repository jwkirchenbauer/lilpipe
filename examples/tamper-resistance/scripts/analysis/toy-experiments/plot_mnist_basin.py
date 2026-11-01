"""Plot MNIST/CIFAR-10 SFT/BaSFT loss, test accuracy, and gradient norm in one PDF."""

import argparse
import json
import os

import pandas as pd
import seaborn as sns


def main(results, output):
    with open(results) as stream:
        history = json.load(stream)["history"]
    rows = []
    for record in history:
        for metric in ("loss", "test_accuracy", "gradient_norm"):
            rows.append(
                {
                    "method": record["method"],
                    "phase": record["phase"],
                    "step": record["step"],
                    "metric_name": metric,
                    "metric_value": record[metric],
                }
            )
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plot = sns.relplot(
        data=pd.DataFrame(rows),
        x="step",
        y="metric_value",
        row="phase",
        col="metric_name",
        hue="method",
        style="method",
        kind="line",
        height=2.6,
        aspect=1.25,
        facet_kws={"sharex": False, "sharey": False, "margin_titles": True},
    )
    plot.set_titles(row_template="{row_name}", col_template="{col_name}")
    plot.set_ylabels("")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plot.savefig(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    parser.add_argument("--results")
    parser.add_argument("--output")
    args = parser.parse_args()
    results = args.results or os.path.join(
        os.environ.get("SCRATCH", "."), f"lilpipe/{args.dataset}-sft-basft/results.json"
    )
    output = (
        args.output
        or f"artifacts/figures/resnet18-{args.dataset}-relearning-trajectory.pdf"
    )
    main(results, output)
