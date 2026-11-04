#!/usr/bin/env python3
"""Plot the authors' SEAM relearning sweep with our four-GPU evaluations."""

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
        "--outpath",
        default=(
            "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms250"
            "--author-codebase-paper-config.pdf"
        ),
    )
    args = parser.parse_args()

    base = "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
    learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    configs = [
        {
            "path": osp.join(args.rootdir, base),
            "capabilities": "capabilities-4gpu",
            "harmfulness": "harmfulness-4gpu-bs16/metrics.json",
            "lr": lr,
            "step": 0,
        }
        for lr in learning_rates
    ] + [
        {
            "path": osp.join(
                args.rootdir,
                f"{base}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250",
                f"checkpoint-{step}",
            ),
            "capabilities": "capabilities",
            "harmfulness": "harmfulness.json",
            "lr": lr,
            "step": step,
        }
        for lr in learning_rates
        for step in range(50, 251, 50)
    ]

    rows = []
    for cfg in configs:
        task_results = get_lm_eval_results(osp.join(cfg["path"], cfg["capabilities"]))
        capability = sum(
            task_results[task][metric] for task, metric in task_to_metric.items()
        ) / len(task_to_metric)
        with open(osp.join(cfg["path"], cfg["harmfulness"])) as handle:
            harmfulness = json.load(handle)["harmfulness"]
        for evaluation, score in (
            ("Harmfulness ↓", harmfulness),
            ("Capability ↑", capability),
        ):
            rows.append(
                {
                    "Relearning step": cfg["step"],
                    "Score (%)": 100 * score,
                    "LR": cfg["lr"],
                    "Eval": evaluation,
                }
            )

    df = pd.DataFrame(rows)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Eval",
        hue="LR",
        hue_order=learning_rates,
        kind="line",
        marker="o",
        errorbar=None,
        height=3.4,
    )
    grid.set(xticks=list(range(0, 251, 50)), ylim=(0, 100))
    grid.set_titles("{col_name}")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
