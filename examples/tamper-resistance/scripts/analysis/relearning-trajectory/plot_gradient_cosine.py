"""Plot harmful/benign gradient cosine from Trainer log histories."""

import argparse
import json
import os
import os.path as osp

run_dir = "artifacts/tamia/runs/llama2-7b-seam-own-trainer-relearning"
configs = [
    {"path": osp.join(run_dir, "trainer-state-lr1e-5.json"), "lr": 1e-5},
    {"path": osp.join(run_dir, "trainer-state-lr5e-5.json"), "lr": 5e-5},
]


def get_results(configs):
    """Read each cosine once, retaining the Trainer's pre-update step numbers."""
    rows = []
    for cfg in configs:
        with open(cfg["path"]) as handle:
            history = json.load(handle)["log_history"]
        for record in history:
            cosine = record.get("cosine_similarity", record.get("loss_sd"))
            if cosine is not None:
                rows.append(
                    {"Step": record["step"], "Cosine": cosine, "LR": float(cfg["lr"])}
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("LR", "TRAINER_STATE"),
        help="Repeat for each learning rate and Trainer state JSON to compare.",
    )
    parser.add_argument(
        "--outpath",
        default="artifacts/figures/llama2-7b-beavertail-relearning-gradient-cosine-ms250.pdf",
    )
    args = parser.parse_args()
    selected = (
        [{"lr": float(lr), "path": path} for lr, path in args.input]
        if args.input
        else configs
    )

    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(get_results(selected))
    sns.set_theme(context="paper", style="ticks", font_scale=1.3)
    grid = sns.relplot(
        data=df,
        x="Step",
        y="Cosine",
        size="LR",
        style="LR",
        color="black",
        sizes=(1, 2.5),
        markers=True,
        markevery=20,
        dashes=False,
        legend="full",
        estimator=None,
        errorbar=None,
        kind="line",
        height=3.4,
        aspect=1.7,
    )
    grid.set_axis_labels("Completed optimizer steps", "Gradient cosine")
    grid.ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
    for label in grid.legend.get_texts():
        label.set_text(f"{float(label.get_text()):.0e}")
    os.makedirs(osp.dirname(args.outpath) or ".", exist_ok=True)
    grid.figure.savefig(args.outpath, bbox_inches="tight")


if __name__ == "__main__":
    main()
