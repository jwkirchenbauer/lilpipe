"""Plot CIFAR-10 relearning trajectories with attack learning rates as rows."""

import argparse
import json
import os
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument(
    "--output",
    default="artifacts/figures/resnet18-cifar10-tar-relearning-trajectory.pdf",
)
args = parser.parse_args()
root = args.root
configs = [
    {"path": os.path.join(root, "random", "results.json"), "name": "Random init"}
]
configs += [
    {"path": os.path.join(root, folder, "results.json"), "name": f"TAR k={k}"}
    for k, folder in [(1, "tar-k1"), (8, "tar-k8"), (16, "tar"), (32, "tar-k32")]
]
configs += [
    {
        "path": os.path.join(root, f"sam-{variant}-rho0.01", "results.json"),
        "name": f"SAM ({variant}) ρ=0.01",
    }
    for variant in ["nll", "ent"]
]
rows = []
for lr, subdir in [(0.001, ""), (0.01, "attack-lr0.01")]:
    for config in configs:
        relative = os.path.relpath(config["path"], root)
        path = os.path.join(root, subdir, relative)
        if not os.path.exists(path):
            continue
        with open(path) as stream:
            history = json.load(stream)["history"]
        for record in history:
            if record["phase"] == "relearn":
                for metric in ["test_accuracy", "loss", "gradient_norm"]:
                    rows.append(
                        dict(
                            name=config["name"],
                            attack_lr=lr,
                            step=record["step"],
                            metric=metric,
                            value=record[metric],
                        )
                    )
data = pd.DataFrame(rows)
sns.set_theme(style="whitegrid")
tar_names = [f"TAR k={k}" for k in [1, 8, 16, 32]]
palette = dict(zip(tar_names, sns.color_palette("Blues", n_colors=5)[1:]))
palette["Random init"] = "black"
sam_names = ["SAM (nll) ρ=0.01", "SAM (ent) ρ=0.01"]
palette.update(dict(zip(sam_names, ["#d95f02", "#1b9e77"])))
plot = sns.relplot(
    data=data,
    x="step",
    y="value",
    hue="name",
    palette=palette,
    hue_order=["Random init"] + tar_names + sam_names,
    col="metric",
    row="attack_lr",
    row_order=[0.001, 0.01],
    kind="line",
    facet_kws={"sharey": False},
    height=3.6,
    aspect=1.15,
)
plot.set_axis_labels("SFT updates", "")
plot.set_titles(col_template="{col_name}", row_template="Attack LR={row_name}")
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
plot.savefig(args.output)
data.to_csv(os.path.join(root, "trajectories.csv"), index=False)
print(args.output)
