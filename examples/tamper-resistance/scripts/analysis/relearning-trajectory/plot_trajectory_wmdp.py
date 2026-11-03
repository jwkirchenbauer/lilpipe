#!/usr/bin/env python3
"""Plot WMDP relearning trajectories; uncomment one selection block below."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

rootdir = "artifacts/tamia/evals"
tasks = {"wmdp_bio_robust": "WMDP-Bio robust ↓", "mmlu_no_bio": "MMLU-no-bio ↑"}

tar = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
seam = "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam"
configs_main = [
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft",
        "method": "RMU",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2--relearn-fft",
        "method": "RMU",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-rmu-wd--relearn-fft",
        "method": "RMU",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo--relearn-fft",
        "method": "NPO",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--relearn-fft",
        "method": "NPO",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-npo-wd--relearn-fft-s2",
        "method": "NPO",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me--relearn-fft",
        "method": "MaxEnt",
        "defence": "None",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft",
        "method": "MaxEnt",
        "defence": "SAM",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-wd--relearn-fft",
        "method": "MaxEnt",
        "defence": "Weight distance",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft",
        "method": "MaxEnt",
        "defence": "TAR",
    },
    {
        "model": "z7b-wmdp-bio--5pp--unlearn-fft-gd-seam--relearn-fft",
        "method": "GD",
        "defence": "SEAM",
    },
]

configs_lrs = [
    {"model": f"{model}--relearn-fft-lr{lr}-ms128", "method": method, "lr": lr}
    for model, method in ((tar, "TAR"), (seam, "SEAM"))
    for lr in ("1e-6", "5e-6", "1e-5")
]
configs_tar_lrs = [
    {
        "model": f"{tar}--relearn-fft-ms128--hpo--grid-001--{index:03d}",
        "method": "TAR",
        "lr": lr,
        "baseline": tar,
    }
    for index, lr in enumerate(("1e-6", "5e-6", "1e-5"), 1)
]
configs_ms1000 = [
    {"model": f"{tar}--relearn-fft-lr{lr}-ms1000", "method": "TAR", "lr": lr}
    for lr in ("1e-6", "5e-6", "1e-5")
]
configs_perturbed_ms1000 = [
    {
        "model": f"{tar}--relearn-perturb-and-fft-sigma{sigma}-lr1e-5-ms1000",
        "method": "TAR",
        "sigma": sigma,
    }
    for sigma in ("1e-4", "1e-3", "1e-2", "1e-1")
]


def results(experiment, baseline=None):
    if baseline is not None:
        for _, label, score in results(baseline):
            yield 0, label, score
    pattern = osp.join(rootdir, experiment, "checkpoint-*", "**", "results*.json")
    paths = glob.glob(pattern, recursive=True)
    for path in sorted(
        paths, key=lambda path: int(path.split("checkpoint-")[1].split(os.sep)[0])
    ):
        with open(path) as handle:
            result = json.load(handle)["results"]
        step = int(path.split("checkpoint-")[1].split(os.sep)[0])
        for task, label in tasks.items():
            if task in result:
                yield step, label, 100 * result[task]["acc,none"]


def legend_with_base(legend, base):
    """Place Base inside the Method section when that section exists."""
    if "Method" in legend:
        method = legend.pop("Method")
        return {"Method": method, "Base": base, **legend}
    return {"Base": base, **legend}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--tar-lrs", action="store_true")
    args = parser.parse_args()

    # Uncomment one complete selection block.
    configs = configs_main
    steps = range(0, 17, 4)
    style = {
        "hue": "Method",
        "style": "Defence",
        "markers": True,
        "dashes": False,
        "palette": {
            "RMU": "#4477AA",
            "NPO": "#EE6677",
            "MaxEnt": "#228833",
            "GD": "#AA3377",
        },
    }
    output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf"

    # configs = configs_lrs
    # steps = range(0, 129)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--ms128.pdf"

    # configs = configs_tar_lrs
    # steps = range(0, 129)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--tar-k16-lrs.pdf"

    # configs = configs_ms1000
    # steps = range(0, 1001)
    # style = {
    #     "hue": "Method",
    #     "size": "LR",
    #     "size_order": ["1e-6", "5e-6", "1e-5"],
    #     "sizes": {"1e-6": 0.6, "5e-6": 1.5, "1e-5": 3.0},
    # }
    # output = "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--ms1000.pdf"

    # configs = configs_perturbed_ms1000
    # steps = range(0, 1001)
    # style = {
    #     "hue": "Method",
    #     "size": "σ",
    #     "size_order": ["1e-4", "1e-3", "1e-2", "1e-1"],
    #     "sizes": {"1e-4": 0.6, "1e-3": 1.2, "1e-2": 2.0, "1e-1": 3.0},
    # }
    # output = (
    #     "artifacts/figures/"
    #     "z7b-wmdp-mmlu-relearning-trajectory-perturbed--ms1000.pdf"
    # )

    if args.tar_lrs:
        configs = configs_tar_lrs
        steps = range(0, 129, 16)
        style = {
            "hue": "LR",
            "hue_order": ["1e-6", "5e-6", "1e-5"],
            "markers": True,
            "marker": "o",
            "dashes": False,
        }
        output = (
            "artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--tar-k16-new-api.pdf"
        )

    rows = []
    for config in configs:
        for step, evaluation, accuracy in results(
            config["model"], config.get("baseline")
        ):
            if step in steps:
                rows.append(
                    {
                        "Relearning step": step,
                        "Accuracy (%)": accuracy,
                        "Method": config["method"],
                        "Defence": config.get("defence", ""),
                        "LR": config.get("lr", ""),
                        "σ": config.get("sigma", ""),
                        "Eval": evaluation,
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
        kind="line",
        errorbar=None,
        height=3.4,
        aspect=1,
        **style,
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
    output = args.output or output
    os.makedirs(osp.dirname(output) or ".", exist_ok=True)
    grid.figure.savefig(output, bbox_inches="tight")
