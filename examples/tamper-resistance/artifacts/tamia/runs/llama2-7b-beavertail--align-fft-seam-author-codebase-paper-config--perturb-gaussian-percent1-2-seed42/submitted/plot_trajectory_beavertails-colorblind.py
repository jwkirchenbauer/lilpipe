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
        glob.glob(osp.join(path, "**", "results_*.json"), recursive=True)
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
        "--lr-markers",
        action="store_true",
        help="Include the fuller comparison with SEAM + 0.75% noise",
    )
    parser.add_argument(
        "--outpath",
        help="Output PDF path (defaults to a separate PDF when --lr-markers is set)",
    )
    args = parser.parse_args()
    outpath = args.outpath or (
        "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms250"
        "--author-codebase-paper-config"
        f"{'--lr-markers' if args.lr_markers else ''}.pdf"
    )

    models = {
        "SEAM": "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config",
        "Base": "llama2-7b-beavertail",
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
    }
    if args.lr_markers:
        models["SEAM + 0.75% noise"] = (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.0075-seed42"
        )
    learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    if args.lr_markers:
        learning_rates = [
            "2e-5",
            "5e-5",
            "8e-5",
            # "1e-4",
            # "2e-4",
        ]
    method_learning_rates = {method: learning_rates for method in models}
    method_learning_rates["Base"] = [
        # "2e-5",
        # "5e-5",
        # "8e-5",
        "1e-4",
        # "2e-4",
    ]
    if args.lr_markers:
        method_learning_rates["Base"] = [
            # "2e-5",
            # "5e-5",
            "8e-5",
            # "1e-4",
            # "2e-4",
        ]
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
            ["SEAM", "SEAM + 0.75% noise", "SEAM + 1% noise", "SEAM + 2% noise"],
            sns.color_palette("colorblind", 4),
        )
    )
    palette["Base"] = "black"
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Eval",
        col_order=["Harmfulness ↓", "Capability ↑"],
        hue="Method",
        hue_order=[method for method in models if method in df["Method"].unique()],
        palette=palette,
        size="LR",
        style="LR",
        legend="full",
        kind="line",
        markers=True,
        dashes=False,
        errorbar=None,
        height=3.4,
    )
    for text in grid.legend.texts:
        try:
            text.set_text(f"{float(text.get_text()):.0e}")
        except ValueError:
            pass
    grid.set(xticks=list(range(0, 251, 50)), ylim=(0, 100))
    grid.set_titles("{col_name}")
    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
