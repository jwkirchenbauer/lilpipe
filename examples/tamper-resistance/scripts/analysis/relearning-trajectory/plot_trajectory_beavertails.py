#!/usr/bin/env python3
"""Plot the selected BeaverTails relearning runs; edit the config blocks below."""

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


# Capability average. Uncomment GSM8K to use five tasks instead of four.
task_to_metric = {
    "mmlu": "acc,none",
    "truthfulqa_mc1": "acc,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    # "gsm8k": "exact_match,flexible-extract",
}

base = "llama2-7b-beavertail"
seam = base + "--align-fft-seam-author-codebase-paper-config"
author_attack = "--data-overlapping--relearn-fft-author-codebase-"
our_attack = "--data-overlapping--relearn-fft-"

# 1. Authors' 250-step attacks: method-comparison or expanded method-row plots.
# Uncomment this block, and comment the active block below. Comment individual
# entries to select methods/LRs. For the earlier reduced plot, keep Base at 1e-4;
# the earlier marker comparison kept Base at 8e-5. Row plots included every LR.
# configs = [
#     # Base
#     {"path": (base) + author_attack + "lr2e-5-ms250",
#      "baseline": base, "method": "Base", "lr": 2e-5},
#     {"path": (base) + author_attack + "lr5e-5-ms250",
#      "baseline": base, "method": "Base", "lr": 5e-5},
#     {"path": (base) + author_attack + "lr8e-5-ms250",
#      "baseline": base, "method": "Base", "lr": 8e-5},
#     {"path": (base) + author_attack + "lr1e-4-ms250",
#      "baseline": base, "method": "Base", "lr": 1e-4},
#     {"path": (base) + author_attack + "lr2e-4-ms250",
#      "baseline": base, "method": "Base", "lr": 2e-4},
#     # Base + 1% noise
#     {"path": (base + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": base + "--perturb-gaussian-l2r0.01-seed42", "method": "Base + 1% noise", "lr": 2e-5},
#     {"path": (base + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": base + "--perturb-gaussian-l2r0.01-seed42", "method": "Base + 1% noise", "lr": 5e-5},
#     {"path": (base + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": base + "--perturb-gaussian-l2r0.01-seed42", "method": "Base + 1% noise", "lr": 8e-5},
#     {"path": (base + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": base + "--perturb-gaussian-l2r0.01-seed42", "method": "Base + 1% noise", "lr": 1e-4},
#     {"path": (base + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": base + "--perturb-gaussian-l2r0.01-seed42", "method": "Base + 1% noise", "lr": 2e-4},
#     # SEAM
#     {"path": (seam) + author_attack + "lr2e-5-ms250",
#      "baseline": seam, "method": "SEAM", "lr": 2e-5},
#     {"path": (seam) + author_attack + "lr5e-5-ms250",
#      "baseline": seam, "method": "SEAM", "lr": 5e-5},
#     {"path": (seam) + author_attack + "lr8e-5-ms250",
#      "baseline": seam, "method": "SEAM", "lr": 8e-5},
#     {"path": (seam) + author_attack + "lr1e-4-ms250",
#      "baseline": seam, "method": "SEAM", "lr": 1e-4},
#     {"path": (seam) + author_attack + "lr2e-4-ms250",
#      "baseline": seam, "method": "SEAM", "lr": 2e-4},
#     # SEAM + 0.75% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.0075-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.0075-seed42", "method": "SEAM + 0.75% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.0075-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.0075-seed42", "method": "SEAM + 0.75% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.0075-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.0075-seed42", "method": "SEAM + 0.75% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.0075-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.0075-seed42", "method": "SEAM + 0.75% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.0075-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.0075-seed42", "method": "SEAM + 0.75% noise", "lr": 2e-4},
#     # SEAM + 1% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.01-seed42", "method": "SEAM + 1% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.01-seed42", "method": "SEAM + 1% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.01-seed42", "method": "SEAM + 1% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.01-seed42", "method": "SEAM + 1% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.01-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.01-seed42", "method": "SEAM + 1% noise", "lr": 2e-4},
#     # SEAM + 2% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.02-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.02-seed42", "method": "SEAM + 2% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.02-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.02-seed42", "method": "SEAM + 2% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.02-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.02-seed42", "method": "SEAM + 2% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.02-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.02-seed42", "method": "SEAM + 2% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.02-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.02-seed42", "method": "SEAM + 2% noise", "lr": 2e-4},
#     # SEAM + 5% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.05-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.05-seed42", "method": "SEAM + 5% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.05-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.05-seed42", "method": "SEAM + 5% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.05-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.05-seed42", "method": "SEAM + 5% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.05-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.05-seed42", "method": "SEAM + 5% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.05-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.05-seed42", "method": "SEAM + 5% noise", "lr": 2e-4},
#     # SEAM + 10% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.1-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.1-seed42", "method": "SEAM + 10% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.1-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.1-seed42", "method": "SEAM + 10% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.1-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.1-seed42", "method": "SEAM + 10% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.1-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.1-seed42", "method": "SEAM + 10% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.1-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.1-seed42", "method": "SEAM + 10% noise", "lr": 2e-4},
#     # SEAM + 20% noise
#     {"path": (seam + "--perturb-gaussian-l2r0.2-seed42") + author_attack + "lr2e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.2-seed42", "method": "SEAM + 20% noise", "lr": 2e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.2-seed42") + author_attack + "lr5e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.2-seed42", "method": "SEAM + 20% noise", "lr": 5e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.2-seed42") + author_attack + "lr8e-5-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.2-seed42", "method": "SEAM + 20% noise", "lr": 8e-5},
#     {"path": (seam + "--perturb-gaussian-l2r0.2-seed42") + author_attack + "lr1e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.2-seed42", "method": "SEAM + 20% noise", "lr": 1e-4},
#     {"path": (seam + "--perturb-gaussian-l2r0.2-seed42") + author_attack + "lr2e-4-ms250",
#      "baseline": seam + "--perturb-gaussian-l2r0.2-seed42", "method": "SEAM + 20% noise", "lr": 2e-4},
# ]
# steps = list(range(0, 251, 50))
# outpath = "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms250--author-codebase-paper-config.pdf"
# For expanded rows, use the row-layout block below and suffix --all-methods.
# For five-task plots, also uncomment GSM8K above and omit runs missing GSM8K.

# 2. Our trainer, 250 steps: uncomment this block and comment block 3.
# configs = [
#     {"path": seam + our_attack + "lr1e-5-ms250", "baseline": seam, "method": "SEAM", "lr": 1e-5},
#     {"path": seam + our_attack + "lr5e-5-ms250", "baseline": seam, "method": "SEAM", "lr": 5e-5},
#     {"path": seam + our_attack + "lr2e-4-ms250-warmup0.5", "baseline": seam, "method": "SEAM (50% warmup)", "lr": 2e-4},
# ]
# steps = list(range(0, 251, 50))
# outpath = "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms250--author-codebase-paper-config--own-relearn-trainer.pdf"

# 3. Our trainer, 1,000 steps: produces the four-LR trajectory PDF.
configs = [
    {
        "path": seam + our_attack + "lr1e-5-ms1000",
        "baseline": seam,
        "method": "SEAM",
        "lr": 1e-5,
    },
    {
        "path": seam + our_attack + "lr5e-5-ms1000",
        "baseline": seam,
        "method": "SEAM",
        "lr": 5e-5,
    },
    {
        "path": seam + our_attack + "lr1e-4-ms1000",
        "baseline": seam,
        "method": "SEAM",
        "lr": 1e-4,
    },
    {
        "path": seam + our_attack + "lr5e-4-ms1000",
        "baseline": seam,
        "method": "SEAM",
        "lr": 5e-4,
    },
]
steps = list(range(0, 1001, 50))
outpath = "artifacts/figures/llama2-7b-beavertail-relearning-trajectory-ms1000--author-codebase-paper-config--own-relearn-trainer.pdf"

# Layout: uncomment ONE assignment. LR always controls thickness and markers.
facet = {"hue": "Method", "height": 3.4}
# facet = {"row": "Method", "color": "black", "height": 3.2, "aspect": 1.5}

# Expanded author plots used this dashed reference. Uncomment when selected.
reference = None
# reference = {"method": "Base", "lr": 1e-4, "step": 250}


def get_results(configs, rootdir, steps, task_to_metric):
    """Load the explicitly selected runs; missing requested metrics are errors."""
    rows = []
    for cfg in configs:
        for step in steps:
            directory = osp.join(rootdir, cfg["baseline"] if step == 0 else cfg["path"])
            if step != 0:
                directory = osp.join(directory, f"checkpoint-{step}")
            capabilities = "capabilities-4gpu" if step == 0 else "capabilities"
            tasks = get_lm_eval_results(osp.join(directory, capabilities))
            if "gsm8k" in task_to_metric:
                tasks.update(get_lm_eval_results(osp.join(directory, "gsm8k-5shot")))
            capability = sum(
                tasks[task][metric] for task, metric in task_to_metric.items()
            ) / len(task_to_metric)
            harmfulness = osp.join(directory, "harmfulness.json")
            if step == 0 and not osp.isfile(harmfulness):
                harmfulness = osp.join(directory, "harmfulness-4gpu-bs16/metrics.json")
            with open(harmfulness) as handle:
                harmfulness = json.load(handle)["harmfulness"]
            for evaluation, score in [
                ("Harmfulness ↓", harmfulness),
                ("Capability ↑", capability),
            ]:
                rows.append(
                    {
                        "Relearning step": step,
                        "Score (%)": 100 * score,
                        "Method": cfg["method"],
                        "LR": float(cfg["lr"]),
                        "Eval": evaluation,
                    }
                )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rootdir", default="artifacts/tamia/evals")
    parser.add_argument("--outpath", default=outpath)
    args = parser.parse_args()

    df = pd.DataFrame(get_results(configs, args.rootdir, steps, task_to_metric))
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
    palette = dict(
        zip(
            df["Method"].unique(),
            sns.color_palette("colorblind", df["Method"].nunique()),
        )
    )
    palette["Base"] = "black"
    grid = sns.relplot(
        data=df,
        x="Relearning step",
        y="Score (%)",
        col="Eval",
        col_order=["Harmfulness ↓", "Capability ↑"],
        **facet,
        **({"palette": palette} if "hue" in facet else {}),
        size="LR",
        style="LR",
        markers=True,
        dashes=False,
        legend="full",
        kind="line",
        errorbar=None,
    )
    if reference is not None:
        scores = df[
            (df["Method"] == reference["method"])
            & (df["LR"] == reference["lr"])
            & (df["Relearning step"] == reference["step"])
        ].set_index("Eval")["Score (%)"]
        for key, ax in grid.axes_dict.items():
            evaluation = key[-1] if isinstance(key, tuple) else key
            ax.axhline(scores[evaluation], color="0.45", linestyle="--", linewidth=1.4)
    for label in grid.legend.get_texts():
        try:
            label.set_text(f"{float(label.get_text()):.0e}")
        except ValueError:
            pass
    grid.set(ylim=(0, 100), xticks=steps[:: max(1, (len(steps) - 1) // 5)])
    grid.set_titles("{row_name} | {col_name}" if "row" in facet else "{col_name}")
    grid.figure.suptitle(
        f"Capability: mean of {len(task_to_metric)} tasks", y=1.04, fontsize=12
    )
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")
