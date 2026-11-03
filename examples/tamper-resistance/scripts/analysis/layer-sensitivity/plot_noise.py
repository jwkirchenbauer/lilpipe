"""Plot component-by-noise-scale MMLU accuracy drops."""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute", action="store_true")
    parser.add_argument("--jobs", nargs="+", type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()
    jobs = (
        (11505778, 11505777, 11504999, 11505000)
        if args.absolute
        else range(11492755, 11492760)
    )
    jobs = args.jobs or jobs
    configs = [
        {"path": f"artifacts/mila/evals/z7b-layer-noise-{job}/summary.csv"}
        for job in jobs
    ]
    rows = []
    for cfg in configs:
        if osp.exists(cfg["path"]):
            rows.extend(pd.read_csv(cfg["path"]).to_dict("records"))
    df = pd.DataFrame(rows)
    seeds = df["seed"].unique()
    baselines = df["baseline_accuracy"].unique()
    if len(seeds) != 1 or len(baselines) != 1:
        raise ValueError("Plot one noise seed with a consistent baseline at a time")
    order = ["embedding", *[f"layer-{i}" for i in range(32)], "final-norm", "lm-head"]
    matrix = df.pivot(index="group", columns="scale", values="drop_pp").loc[order]
    scales = (
        [0.0001, 0.001, 0.01, 0.03] if args.absolute else [0.01, 0.03, 0.1, 0.3, 1.0]
    )
    matrix = matrix.reindex(columns=scales)
    if matrix.isna().any().any() or matrix.shape != (35, len(scales)):
        raise ValueError("Expected results for every component and selected scale")
    outpath = "artifacts/figures/z7b-layer-noise-mmlu-grid"
    if args.absolute:
        outpath = "artifacts/figures/z7b-layer-noise-absolute-mmlu-grid"
    outpath = args.output_prefix or outpath
    os.makedirs(osp.dirname(outpath) or ".", exist_ok=True)
    matrix.to_csv(outpath + ".csv", float_format="%.4f")

    sns.set_theme(context="paper", style="white", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.5, 12.5))
    sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.4,
        cbar_kws={"label": "MMLU accuracy drop (percentage points)", "shrink": 0.6},
        yticklabels=[
            "Embedding",
            *[f"Block {i}" for i in range(32)],
            "Final norm",
            "LM head",
        ],
        xticklabels=[f"{scale:g}" for scale in matrix.columns],
    )
    ax.set_facecolor("#e0e0e0")
    for row, col in zip(*matrix.isna().to_numpy().nonzero()):
        ax.text(
            col + 0.5,
            row + 0.5,
            "pending",
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
    ax.set_xlabel(
        r"Absolute noise std. $\sigma$ (no RMS scaling)"
        if args.absolute
        else r"Relative noise scale $\sigma$ (noise std. = $\sigma$ × component weight RMS)"
    )
    ax.set_ylabel("")
    ax.set_title(
        "Z7B: MMLU drop after noise in one component\n"
        f"1,140 fixed questions · baseline {100 * baselines[0]:.2f}% · noise seed {seeds[0]}",
        pad=14,
    )
    ax.tick_params(axis="y", rotation=0)
    for tick in ax.get_yticklabels():
        if tick.get_text() == "LM head":
            tick.set_fontweight("bold")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(outpath + "." + extension, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(matrix.to_string(float_format=lambda value: f"{value:.2f}"))
