#!/usr/bin/env python3
"""Plot the BeaverTails relearning trajectories."""

import json
import os
import os.path as osp

import pandas as pd
import seaborn as sns

root = "artifacts/tamia/evals/llama2-7b-beavertail"
seam_og_root = osp.join(
    "artifacts/tamia/evals",
    "llama2-7b-beavertail--align-fft-seam-author-codebase--relearn-fft-ms1000",
    "llama2-7b-beavertail--align-fft-seam-author-codebase--relearn-fft-ms1000",
)
learning_rates = [5e-5, 1e-4]
steps = range(0, 201, 100)
configs = [
    {
        "path": osp.join(
            root,
            "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
            f"relearn-fft-ms250--hpo--disjoint--{index:03d}/checkpoint-{step}/"
            "summary.json",
        ),
        "method": "SEAM",
        "lr": lr,
        "step": step,
    }
    for index, lr in enumerate(learning_rates, 1)
    for step in steps
] + [
    {
        "path": osp.join(
            root,
            "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--"
            f"relearn-fft-ms250/checkpoint-{step}/summary.json",
        ),
        "method": "SEAM (legacy reproduced)",
        "lr": 1e-4,
        "step": step,
    }
    for step in steps
]

# configs += [
#         {
#             "path": osp.join(
#                 root,
#                 "llama2-7b-beavertail--align-fft-seam-author-paper--data-disjoint--"
#                 "relearn-fft-ms250--hpo--disjoint--002/"
#                 f"checkpoint-{step}/summary.json",
#             ),
#             "method": "SEAM (legacy old)",
#             "lr": 1e-4,
#             "step": step,
#         }
#         for step in steps
#     ]

# configs = [
#     {
#         "path": (
#             "artifacts/mila/evals/qwen3-06b-beavertail/"
#             # Uncomment the desired model's result path.
#             # "qwen3-0.6b-beavertail--align-fft-seam--relearn--hpo"
#             "qwen3-0.6b-beavertail--align-fft-seam-legacy--relearn--hpo"
#             f"--beta{beta}--{lr_index:03d}/"
#             f"checkpoint-{step}/summary.json"
#         ),
#         "method": f"seam-{beta}",
#         "lr": lr,
#         "step": step,
#     }
#     for beta in (0.01, 0.1, 1.0)
#     for lr_index, lr in enumerate(learning_rates, 1)
#     for step in steps
# ]

# configs = [
#     {
#         "path": osp.join(
#             root,
#             "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
#             f"relearn-fft-ms1000--hpo--disjoint--{index:03d}/checkpoint-{step}/"
#             "summary.json",
#         ),
#         "method": "SEAM",
#         "lr": lr,
#         "step": step,
#     }
#     for index, (lr, interval) in enumerate(
#         [(2e-5, 50), (5e-5, 50), (8e-5, 50), (1e-4, 100), (2e-4, 100)], 1
#     )
#     for step in range(0, 1001, interval)
# ] + [
#     {
#         "path": osp.join(seam_og_root, f"checkpoint-{step}/summary.json"),
#         "method": "SEAM-OG",
#         "lr": 5e-5,
#         "step": step,
#     }
#     for step in range(0, 1001, 50)
# ]

rows = []
for config in configs:
    with open(config["path"]) as f:
        result = json.load(f)
    for evaluation, metric in (
        ("Harmfulness", "harmfulness"),
        ("Capability", "capability"),
    ):
        rows.append(
            {
                "Relearning step": config["step"],
                "Score (%)": 100 * result["post"][metric],
                "Method": config["method"],
                "LR": config["lr"],
                "Eval": evaluation,
            }
        )

df = pd.DataFrame(rows)
sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)
g = sns.relplot(
    data=df,
    x="Relearning step",
    y="Score (%)",
    col="Eval",
    hue="Method",
    size="LR",
    kind="line",
    marker="o",
    height=3.4,
)
for text in g.legend.texts:
    try:
        text.set_text(f"{float(text.get_text()):.0e}")
    except ValueError:
        pass

output = (
    "artifacts/figures/" "llama2-7b-beavertail-relearning-trajectory-ms250--lrs.pdf"
)
os.makedirs(osp.dirname(output), exist_ok=True)
g.figure.savefig(output, bbox_inches="tight")
