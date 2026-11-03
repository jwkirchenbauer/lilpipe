#!/usr/bin/env python3
"""Compare 1,000-step attacks on the 125-step and 250-step TAR defences."""

import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from plot_trajectory_wmdp import legend_with_base, results, tasks

if __name__ == "__main__":
    run = "artifacts/evals/tar-ms250-repro-5065b15a"
    configs = [
        {
            "path": (
                f"{run}/historical-evals/"
                "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
                f"--relearn-fft-lr{lr}-ms1000"
            ),
            "method": "TAR-125",
            "lr": lr,
        }
        for lr in ("1e-6", "5e-6", "1e-5")
    ] + [
        {
            "path": f"{run}/evals/relearn-lr5e-6-ms1000",
            "method": "TAR-250",
            "lr": "5e-6",
        }
    ]
    output = "artifacts/figures/" "z7b-wmdp-mmlu-relearning-trajectory--ms1000.pdf"

    rows = []
    for config in configs:
        for step in range(0, 1001, 50):
            scores = {}
            pattern = osp.join(
                config["path"], f"checkpoint-{step}", "**", "results*.json"
            )
            for path in sorted(glob.glob(pattern, recursive=True)):
                with open(path) as handle:
                    scores.update(json.load(handle)["results"])
            for task, label in tasks.items():
                rows.append(
                    {
                        "Relearning step": step,
                        "Accuracy (%)": 100 * scores[task]["acc,none"],
                        "Method": config["method"],
                        "LR": config["lr"],
                        "Eval": label,
                    }
                )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="ticks", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Accuracy (%)",
        col="Eval",
        col_order=list(tasks.values()),
        hue="Method",
        size="LR",
        size_order=["1e-6", "5e-6", "1e-5"],
        sizes={"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
        kind="line",
        errorbar=None,
        height=3.4,
        aspect=1,
    )
    base = {label: score for _, label, score in results("z7b-base")}
    for label, axis in grid.axes_dict.items():
        axis.axhline(base[label], color="black")

    legend = dict(
        zip(
            [text.get_text() for text in grid.legend.texts],
            grid.legend.legend_handles,
        )
    )
    legend = legend_with_base(legend, Line2D([], [], color="black"))
    grid.legend.remove()
    grid.add_legend(legend_data=legend)
    grid.set_axis_labels("Relearning step", "Accuracy (%)")
    os.makedirs(osp.dirname(output), exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")
    grid.figure.savefig(
        output.removesuffix(".pdf") + ".png", bbox_inches="tight", dpi=150
    )
