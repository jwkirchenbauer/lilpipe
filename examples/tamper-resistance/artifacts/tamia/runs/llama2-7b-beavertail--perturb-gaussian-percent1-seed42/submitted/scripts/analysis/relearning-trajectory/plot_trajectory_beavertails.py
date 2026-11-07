#!/usr/bin/env python3
"""Compare relearning from SEAM-defended and starting models."""

import argparse
import glob
import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns


def get_lm_eval_results(path):
    """Load task metrics, letting later result files overwrite earlier ones."""
    results = {}
    for result_path in sorted(
        glob.glob(
            osp.join(path, "**", "results_*.json"), recursive=True, include_hidden=True
        )
    ):
        with open(result_path) as handle:
            results.update(json.load(handle)["results"])
    return results


task_to_metric = {
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Include only method/LR trajectories with every plotted checkpoint saved",
    )
    parser.add_argument(
        "--variant",
        choices=("reduced", "comparison", "all-methods"),
        default="reduced",
        help="Choose a compact comparison or all methods in separate rows",
    )
    parser.add_argument(
        "--outpath",
        help="Output PDF path (each variant has its own default filename)",
    )
    args = parser.parse_args()
    suffix = {
        "reduced": "",
        "comparison": "--lr-markers",
        "all-methods": "--all-methods",
    }
    outpath = args.outpath or (
        "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms250"
        "--author-codebase-paper-config"
        f"{suffix[args.variant]}.pdf"
    )

    models = {
        "SEAM": "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config",
        "Base": "llama2-7b-beavertail",
        "Base + 1% noise": "llama2-7b-beavertail--perturb-gaussian-l2r0.01-seed42",
        # "SEAM + 0.75% noise": (
        #     "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
        #     "--perturb-gaussian-l2r0.0075-seed42"
        # ),
        "SEAM + 1% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.01-seed42"
        ),
        "SEAM + 2% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.02-seed42"
        ),
        "SEAM + 5% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.05-seed42"
        ),
    }
    if args.variant in ("comparison", "all-methods"):
        models["SEAM + 0.75% noise"] = (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.0075-seed42"
        )
    learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    if args.variant == "comparison":
        learning_rates = [
            "2e-5",
            "5e-5",
            "8e-5",
            "1e-4",
            "2e-4",
        ]
    method_learning_rates = {method: learning_rates for method in models}
    method_learning_rates["Base"] = [
        # "2e-5",
        # "5e-5",
        # "8e-5",
        "1e-4",
        # "2e-4",
    ]
    if args.variant == "comparison":
        method_learning_rates["Base"] = [
            # "2e-5",
            # "5e-5",
            "8e-5",
            # "1e-4",
            # "2e-4",
        ]
    if args.variant == "all-methods":
        models = {
            method: models[method]
            for method in (
                "Base",
                "Base + 1% noise",
                "SEAM",
                "SEAM + 0.75% noise",
                "SEAM + 1% noise",
                "SEAM + 2% noise",
                "SEAM + 5% noise",
            )
        }
        method_learning_rates = {method: learning_rates for method in models}
    configs = [
        {
            "path": osp.join(args.rootdir, model),
            "capabilities": "capabilities-4gpu",
            "harmfulness": (
                "harmfulness-4gpu-bs16/metrics.json"
                if method == "SEAM"
                else "harmfulness.json"
            ),
            "method": method,
            "lr": lr,
            "step": 0,
        }
        for method, model in models.items()
        for lr in method_learning_rates[method]
    ] + [
        {
            "path": osp.join(
                args.rootdir,
                f"{model}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250",
                f"checkpoint-{step}",
            ),
            "capabilities": "capabilities",
            "harmfulness": "harmfulness.json",
            "method": method,
            "lr": lr,
            "step": step,
        }
        for method, model in models.items()
        for lr in method_learning_rates[method]
        for step in range(50, 251, 50)
    ]

    if args.completed_only:
        incomplete = set()
        for cfg in configs:
            if not osp.isfile(
                osp.join(cfg["path"], cfg["harmfulness"])
            ) or not glob.glob(
                osp.join(cfg["path"], cfg["capabilities"], "**", "results_*.json"),
                recursive=True,
                include_hidden=True,
            ):
                incomplete.add((cfg["method"], cfg["lr"]))
        configs = [
            cfg for cfg in configs if (cfg["method"], cfg["lr"]) not in incomplete
        ]

    rows = []
    for cfg in configs:
        task_results = get_lm_eval_results(osp.join(cfg["path"], cfg["capabilities"]))
        capability = sum(
            task_results[task][metric] for task, metric in task_to_metric.items()
        ) / len(task_to_metric)
        scores = {"Capability ↑": capability}
        if cfg["harmfulness"] is not None:
            with open(osp.join(cfg["path"], cfg["harmfulness"])) as handle:
                scores["Harmfulness ↓"] = json.load(handle)["harmfulness"]
        for evaluation, score in scores.items():
            rows.append(
                {
                    "Relearning step": cfg["step"],
                    "Score (%)": 100 * score,
                    "Method": cfg["method"],
                    "LR": float(cfg["lr"]),
                    "Eval": evaluation,
                }
            )

    df = pd.DataFrame(rows)
    palette = dict(
        zip(
            [
                "SEAM",
                "SEAM + 0.75% noise",
                "SEAM + 1% noise",
                "SEAM + 2% noise",
                "SEAM + 5% noise",
            ],
            sns.color_palette("colorblind", 5),
        )
    )
    palette["Base"] = "black"
    palette["Base + 1% noise"] = sns.color_palette("colorblind", 6)[5]
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    facet_options = {
        "hue": "Method",
        "hue_order": [method for method in models if method in df["Method"].unique()],
        "palette": palette,
        "height": 3.4,
    }
    if args.variant == "all-methods":
        facet_options = {
            "row": "Method",
            "row_order": list(models),
            "color": "black",
            "height": 3.2,
            "aspect": 1.5,
        }
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Eval",
        col_order=["Harmfulness ↓", "Capability ↑"],
        **facet_options,
        size="LR",
        style="LR",
        legend="full",
        kind="line",
        markers=True,
        dashes=False,
        errorbar=None,
    )
    if args.variant == "all-methods":
        base_final = df[
            (df["Method"] == "Base")
            & (df["LR"] == 1e-4)
            & (df["Relearning step"] == 250)
        ].set_index("Eval")["Score (%)"]
        for (_, evaluation), ax in grid.axes_dict.items():
            ax.axhline(
                base_final[evaluation],
                color="0.45",
                linestyle="--",
                linewidth=1.4,
                label="Base final (LR 1e-04)",
                zorder=1,
            )
        grid.figure.suptitle("Dashed: Base at step 250, LR 1e-04", y=1.01, fontsize=12)
    for text in grid.legend.texts:
        try:
            text.set_text(f"{float(text.get_text()):.0e}")
        except ValueError:
            pass
    grid.set(xticks=list(range(0, 251, 50)), ylim=(0, 100))
    grid.set_titles(
        "{row_name} | {col_name}" if args.variant == "all-methods" else "{col_name}"
    )
    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
