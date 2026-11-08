"""Plot harmful attack trajectories and gradient cosines from named run artifacts."""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=MODEL_ID")
    parser.add_argument(
        "--early-run", action="append", default=[], help="LABEL=MODEL_ID"
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    configs = []
    for spec in args.run:
        name, model = spec.split("=", 1)
        configs.append(
            {
                "name": name,
                "model": model,
                "path": os.path.join(args.artifact_root, "models", model),
            }
        )
    rows = []
    training = []
    records = []
    for config in configs:
        checkpoints = glob.glob(
            os.path.join(config["path"], "checkpoint-*", "trainer_state.json")
        )
        if checkpoints:
            latest = max(
                checkpoints,
                key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1]),
            )
            with open(latest) as handle:
                state = json.load(handle)
            for row in state["log_history"]:
                if "ghost_cosine_mean" in row:
                    training.append(
                        {
                            "name": config["name"],
                            "step": row["step"],
                            "cosine": row["ghost_cosine_mean"],
                        }
                    )
                if "loss" in row:
                    records.append(
                        {
                            "name": config["name"],
                            "step": row["step"],
                            "loss": row["loss"],
                        }
                    )
        for path in glob.glob(
            os.path.join(
                args.artifact_root,
                "evals",
                "ghost-cosine",
                config["model"],
                "checkpoint-*",
                "summary.json",
            )
        ):
            with open(path) as handle:
                summary = json.load(handle)
            rows.append(
                {
                    "name": config["name"],
                    "step": summary["checkpoint_step"],
                    "harmfulness": summary["post"]["harmfulness"],
                }
            )
    harmful = pd.DataFrame(rows, columns=["name", "step", "harmfulness"]).sort_values(
        ["name", "step"]
    )
    cosine = pd.DataFrame(training, columns=["name", "step", "cosine"]).sort_values(
        ["name", "step"]
    )
    losses = pd.DataFrame(records, columns=["name", "step", "loss"]).sort_values(
        ["name", "step"]
    )
    for name, frame in [("harmfulness", harmful), ("cosine", cosine), ("loss", losses)]:
        frame.to_csv(os.path.join(args.output, name + ".csv"), index=False)
    sns.set_theme(style="whitegrid", context="talk")
    palette = {
        "Refusal only": "#c44e52",
        "Cosine 0.01": "#4c72b0",
        "Cosine 0.1": "#dd8452",
        "Cosine 1": "#55a868",
    }
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    if not harmful.empty:
        sns.lineplot(
            data=harmful,
            x="step",
            y="harmfulness",
            hue="name",
            marker="o",
            palette=palette,
            ax=axes[0],
        )
        axes[0].set_ylim(0, 1)
    if not cosine.empty:
        cosine["cosine_rolling20"] = cosine.groupby("name")["cosine"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        sns.lineplot(
            data=cosine,
            x="step",
            y="cosine_rolling20",
            hue="name",
            palette=palette,
            ax=axes[1],
        )
        axes[1].axhline(0, color="gray", linewidth=1)
    if not losses.empty:
        losses["loss_rolling20"] = losses.groupby("name")["loss"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        sns.lineplot(
            data=losses,
            x="step",
            y="loss_rolling20",
            hue="name",
            palette=palette,
            ax=axes[2],
        )
    for ax, title, ylabel in zip(
        axes,
        ["Held-out harmfulness", "Harmful-gradient cosine", "Harmful fine-tuning loss"],
        [
            "Classifier score",
            "Off-diagonal cosine (20-step mean)",
            "Cross entropy (20-step mean)",
        ],
    ):
        ax.set(title=title, xlabel="Attack optimizer steps", ylabel=ylabel)
        if ax.get_legend():
            ax.get_legend().set_title("")
    fig.tight_layout()
    fig.savefig(
        os.path.join(args.output, "attack_comparison.png"), dpi=180, bbox_inches="tight"
    )
    fig.savefig(os.path.join(args.output, "attack_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)
    early_rows = []
    for spec in args.early_run:
        name, model = spec.split("=", 1)
        for path in glob.glob(
            os.path.join(
                args.artifact_root,
                "evals",
                "ghost-cosine",
                model,
                "checkpoint-*",
                "summary.json",
            )
        ):
            with open(path) as handle:
                value = json.load(handle)
            early_rows.append(
                {
                    "name": name,
                    "step": value["checkpoint_step"],
                    "harmfulness": value["post"]["harmfulness"],
                }
            )
    early = pd.DataFrame(
        early_rows, columns=["name", "step", "harmfulness"]
    ).sort_values(["name", "step"])
    if not early.empty:
        early.to_csv(os.path.join(args.output, "early_harmfulness.csv"), index=False)
        g = sns.relplot(
            data=early,
            x="step",
            y="harmfulness",
            hue="name",
            kind="line",
            marker="o",
            palette=palette,
            height=4.5,
            aspect=1.5,
        )
        g.set(
            ylim=(0, 1),
            xlabel="Attack optimizer steps",
            ylabel="Harmfulness classifier score",
        )
        g.fig.suptitle("Early attack harmfulness", y=1.03, fontsize=18)
        if g._legend:
            g._legend.set_title("")
        g.savefig(os.path.join(args.output, "early_harmfulness.png"), dpi=180)
        g.savefig(os.path.join(args.output, "early_harmfulness.pdf"))
        plt.close(g.fig)
    summary = {}
    for config in configs:
        h = harmful[harmful["name"] == config["name"]]
        c = cosine[cosine["name"] == config["name"]]
        result = {"harmfulness": dict(zip(h["step"].astype(str), h["harmfulness"]))}
        if len(h) >= 2:
            xs = h["step"].tolist()
            ys = h["harmfulness"].tolist()
            result["harmfulness_auc_mean"] = sum(
                (xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2
                for i in range(len(xs) - 1)
            ) / (xs[-1] - xs[0])
        if len(c):
            result["cosine_first25"] = c.head(25)["cosine"].mean()
            result["cosine_last25"] = c.tail(25)["cosine"].mean()
        summary[config["name"]] = result
    # Paired bootstrap: every model is scored on the same held-out prompts.
    comparisons = []
    control = configs[0]
    for config in configs[1:]:
        for step in range(0, 251, 50):
            paths = [
                os.path.join(
                    args.artifact_root,
                    "evals",
                    "ghost-cosine",
                    item["model"],
                    f"checkpoint-{step}",
                    "summary.json",
                )
                for item in (control, config)
            ]
            if not all(os.path.isfile(path) for path in paths):
                continue
            values = []
            for path in paths:
                with open(path) as handle:
                    values.append(json.load(handle)["post"]["generations"])
            if [row["prompt"] for row in values[0]] != [
                row["prompt"] for row in values[1]
            ]:
                raise ValueError(
                    "Paired comparison requires identical evaluation prompts"
                )
            differences = np.array(
                [
                    b["harmful_score"] - a["harmful_score"]
                    for a, b in zip(*values, strict=True)
                ]
            )
            rng = np.random.default_rng(42)
            boot = differences[
                rng.integers(0, len(differences), size=(2000, len(differences)))
            ].mean(axis=1)
            comparisons.append(
                {
                    "name": config["name"],
                    "control": control["name"],
                    "step": step,
                    "difference": float(differences.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                }
            )
    pd.DataFrame(comparisons).to_csv(
        os.path.join(args.output, "paired_harmfulness.csv"), index=False
    )
    with open(os.path.join(args.output, "comparison.json"), "w") as handle:
        json.dump(
            {
                "runs": summary,
                "paired_harmfulness": comparisons,
                "early_harmfulness": early_rows,
            },
            handle,
            indent=2,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
