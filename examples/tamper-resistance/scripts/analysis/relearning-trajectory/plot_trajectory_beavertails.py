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
        choices=("reduced", "comparison", "all-methods", "gsm8k", "own-trainer"),
        default="reduced",
        help="Choose a compact comparison or all methods in separate rows",
    )
    parser.add_argument(
        "--outpath",
        help="Output PDF path (each variant has its own default filename)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        choices=(250, 1000),
        default=250,
        help="Attack length; 1000 is supported for the own-trainer variant",
    )
    args = parser.parse_args()
    if args.max_steps != 250 and args.variant != "own-trainer":
        parser.error("--max-steps 1000 requires --variant own-trainer")
    suffix = {
        "reduced": "",
        "comparison": "--lr-markers",
        "all-methods": "--all-methods",
        "gsm8k": "--all-methods-gsm8k-flexible",
        "own-trainer": "--own-relearn-trainer",
    }
    expanded = args.variant in ("all-methods", "gsm8k")
    if args.variant == "gsm8k":
        task_to_metric["gsm8k"] = "exact_match,flexible-extract"
    outpath = args.outpath or (
        f"artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms{args.max_steps}"
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
    for percent, fraction in ((10, "0.1"), (20, "0.2")):
        models[f"SEAM + {percent}% noise"] = (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            f"--perturb-gaussian-l2r{fraction}-seed42"
        )
    if args.variant in ("comparison", "all-methods", "gsm8k"):
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
    if expanded:
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
                "SEAM + 10% noise",
                "SEAM + 20% noise",
            )
        }
        if args.variant == "gsm8k":
            models = {
                method: model
                for method, model in models.items()
                if method not in ("SEAM", "SEAM + 10% noise", "SEAM + 20% noise")
            }
        method_learning_rates = {method: learning_rates for method in models}
    relearn_backend = "author-codebase-"
    if args.variant == "own-trainer":
        models = {"SEAM": models["SEAM"]}
        method_learning_rates = {
            "SEAM": (
                ["1e-5", "5e-5", "1e-4", "5e-4"]
                if args.max_steps == 1000
                else ["1e-5", "5e-5"]
            )
        }
        relearn_backend = ""
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
                f"{model}--data-overlapping--relearn-fft-{relearn_backend}lr{lr}-ms{args.max_steps}",
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
        for step in range(50, args.max_steps + 1, 50)
    ]

    if args.completed_only or args.variant == "gsm8k":
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
            if args.variant == "gsm8k":
                gsm_results = get_lm_eval_results(osp.join(cfg["path"], "gsm8k-5shot"))
                if "exact_match,flexible-extract" not in gsm_results.get("gsm8k", {}):
                    incomplete.add((cfg["method"], cfg["lr"]))
        configs = [
            cfg for cfg in configs if (cfg["method"], cfg["lr"]) not in incomplete
        ]

    rows = []
    for cfg in configs:
        task_results = get_lm_eval_results(osp.join(cfg["path"], cfg["capabilities"]))
        if args.variant == "gsm8k":
            task_results.update(
                get_lm_eval_results(osp.join(cfg["path"], "gsm8k-5shot"))
            )
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
    palette["SEAM + 10% noise"] = sns.color_palette("colorblind", 8)[6]
    palette["SEAM + 20% noise"] = sns.color_palette("colorblind", 8)[7]
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    facet_options = {
        "hue": "Method",
        "hue_order": [method for method in models if method in df["Method"].unique()],
        "palette": palette,
        "height": 3.4,
    }
    if expanded:
        facet_options = {
            "row": "Method",
            "row_order": (
                [method for method in models if method in df["Method"].unique()]
                if args.variant == "gsm8k"
                else list(models)
            ),
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
    if expanded:
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
        title = "Dashed: Base at step 250, LR 1e-04"
        if args.variant == "gsm8k":
            title += "\nCapability: mean of five tasks, including GSM8K flexible"
        grid.figure.suptitle(title, y=1.01, fontsize=12)
    for text in grid.legend.texts:
        try:
            text.set_text(f"{float(text.get_text()):.0e}")
        except ValueError:
            pass
    grid.set(
        xticks=list(
            range(0, args.max_steps + 1, 200 if args.max_steps == 1000 else 50)
        ),
        ylim=(0, 100),
    )
    if args.variant == "own-trainer":
        grid.figure.suptitle(
            "Our SEAM relearn trainer | Capability: mean of four tasks",
            y=1.04,
            fontsize=12,
        )
    grid.set_titles("{row_name} | {col_name}" if expanded else "{col_name}")
    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    grid.figure.savefig(outpath, bbox_inches="tight")
