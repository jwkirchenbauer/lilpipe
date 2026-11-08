"""Plot CIFAR-10 relearning trajectories with attack learning rates as rows."""

import argparse
import json
import os
import sys
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root")
parser.add_argument("--fullbatch-root", help="Compare saved 512/1024 full-batch runs")
parser.add_argument(
    "--run",
    nargs=2,
    action="append",
    default=[],
    metavar=("NAME", "RESULTS"),
    help="Add a named results.json trajectory to the full-batch comparison",
)
parser.add_argument(
    "--metric",
    default="test_accuracy",
    choices=(
        "test_accuracy",
        "loss",
        "individual_pairwise_cosine",
        "diagnostic_group_cosine",
    ),
    help="Metric for full-batch/named-run relearning comparisons",
)
parser.add_argument(
    "--fullbatch-tar",
    action="store_true",
    help="Compare TAR versus no defense using Adam only",
)
parser.add_argument(
    "--output",
    default="artifacts/figures/resnet18-cifar10-tar-relearning-trajectory.pdf",
)
parser.add_argument(
    "--tar-defence-trace",
    action="store_true",
    help="Plot outer forget loss and cumulative inner attack loss",
)
parser.add_argument(
    "--bad-minima-root",
    help="Compare candidate basin methods and random initialization",
)
args = parser.parse_args()
if args.bad_minima_root:
    root = args.bad_minima_root
    configs = [
        {
            "path": os.path.join(root, "..", "cifar10-fullbatch512", "results.json"),
            "name": "Random init",
            "color": "black",
        },
        {
            "path": os.path.join(
                root, "..", "cifar10-fullbatch512-tar-k16-trace", "results.json"
            ),
            "name": "TAR k=16",
            "color": "#2171b5",
        },
    ]
    configs += [
        {
            "path": os.path.join(root, folder, "results.json"),
            "name": name,
            "color": color,
        }
        for folder, name, color in [
            ("target", "Loss target", "#e6ab02"),
            ("stationary01", "Stationary 0.01", "#66a61e"),
            ("stationary1", "Stationary 0.1", "#1b9e77"),
            ("basin10", "Escape 10", "#d95f02"),
            ("basin100", "Escape 100", "#a6761d"),
            ("margin", "Activation margin", "#e7298a"),
            ("margin-strong", "Strong activation margin", "#e31a1c"),
            ("margin-long", "Activation margin 500", "#b15928"),
            ("calibrated", "Learned inactive basin", "#984ea3"),
            ("projected", "ReLU margin projection", "#4daf4a"),
            ("dead-relu", "Dead ReLU control", "#7570b3"),
        ]
    ]
    rows = []
    for cfg in configs:
        with open(cfg["path"]) as stream:
            history = json.load(stream)["history"]
        for record in history:
            if record["phase"] == "relearn":
                rows.append(
                    {
                        "epochs": record["step"],
                        "accuracy": 100 * record["test_accuracy"],
                        "method": cfg["name"],
                    }
                )
    plot = sns.relplot(
        data=pd.DataFrame(rows),
        x="epochs",
        y="accuracy",
        hue="method",
        palette={cfg["name"]: cfg["color"] for cfg in configs},
        kind="line",
        height=4,
        aspect=1.8,
    )
    plot.axes[0, 0].set_title("Full batch Adam attacks on candidate basins")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot.savefig(args.output)
    sys.exit()

if args.tar_defence_trace:
    if not args.fullbatch_root:
        parser.error("--tar-defence-trace requires --fullbatch-root")
    configs = [
        {
            "path": os.path.join(
                args.fullbatch_root,
                f"cifar10-fullbatch{size}-tar-k16-trace{suffix}",
                "results.json",
            ),
            "samples": str(size),
            "name": "TAR k=16",
        }
        for size in [512]
        for lr, suffix in [("1e-3", "")]
    ]
    configs.append(
        {
            "path": os.path.join(
                args.fullbatch_root,
                "cifar10-fullbatch512-tar-k32-trace",
                "results.json",
            ),
            "samples": "512",
            "name": "TAR k=32",
        }
    )
    new_path = os.path.join(
        args.fullbatch_root,
        "cifar10-fullbatch512-tar-k32-trace-inner5e-3",
        "results.json",
    )
    if os.path.exists(new_path):
        configs.append(
            {"path": new_path, "samples": "512", "name": "TAR k=32, inner LR=5e-3"}
        )
    low_lr_path = os.path.join(
        args.fullbatch_root,
        "cifar10-fullbatch512-tar-k32-trace-inner5e-4",
        "results.json",
    )
    if os.path.exists(low_lr_path):
        configs.append(
            {"path": low_lr_path, "samples": "512", "name": "TAR k=32, inner LR=5e-4"}
        )
    full_path = os.path.join(
        args.fullbatch_root, "cifar10-fullbatch512-tar-full-k16", "results.json"
    )
    if os.path.exists(full_path):
        configs.append(
            {"path": full_path, "samples": "512", "name": "TAR k=16 (full gradient)"}
        )
    rows = []
    for config in configs:
        with open(config["path"]) as stream:
            history = json.load(stream)["history"]
        for record in history:
            if record["phase"] in ["defense", "inner"]:
                rows.append(
                    {
                        "step": (
                            record["step"]
                            if record["phase"] == "inner"
                            else record["step"] + 1
                        ),
                        "loss": (
                            record["loss"]
                            if record["phase"] == "inner"
                            else record["objective"]
                        ),
                        "Stage": (
                            "Inner attack"
                            if record["phase"] == "inner"
                            else "Outer defense"
                        ),
                        "Training images": config["samples"],
                        "Run": config["name"],
                    }
                )
    sns.set_theme(style="whitegrid")
    plot = sns.relplot(
        data=pd.DataFrame(rows),
        x="step",
        y="loss",
        hue="Run",
        palette={
            "TAR k=16": "#2171b5",
            "TAR k=16 (full gradient)": "#88419d",
            "TAR k=32": "#08306b",
            "TAR k=32, inner LR=5e-3": "#6baed6",
            "TAR k=32, inner LR=5e-4": "#4292c6",
        },
        col="Stage",
        col_order=["Outer defense", "Inner attack"],
        kind="line",
        facet_kws={"sharex": False, "sharey": False},
        height=3.5,
        aspect=1.4,
    )
    plot.set_titles("{col_name}")
    plot.axes[0, 0].set(
        xlabel="Defense update", ylabel="Forget loss (negative entropy)"
    )
    plot.axes[0, 1].set(xlabel="Cumulative inner Adam step", ylabel="Attack loss (NLL)")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot.savefig(args.output, bbox_inches="tight")
    print(args.output)
    sys.exit()

if args.fullbatch_root or args.run:
    root = args.fullbatch_root
    if root is None:
        configs = []
    elif args.fullbatch_tar:
        configs = [
            {
                "path": os.path.join(root, "cifar10-fullbatch512/results.json"),
                "name": "Base",
                "color": "black",
            },
            {
                "path": os.path.join(
                    root, "cifar10-fullbatch512-tar-k16-trace/results.json"
                ),
                "name": r"TAR k=16 ($\alpha_o=1e-3, \alpha_i=1e-3$)",
                "color": "#2171b5",
            },
            {
                "path": os.path.join(
                    root, "cifar10-fullbatch512-tar-k32-trace-inner5e-3/results.json"
                ),
                "name": r"TAR k=32 ($\alpha_o=1e-3, \alpha_i=5e-3$)",
                "color": "#08306b",
            },
        ]
    else:
        configs = [
            {
                "path": os.path.join(
                    root, f"cifar10-fullbatch{size}", subdir, "results.json"
                ),
                "name": f"{optimizer}, {size} samples",
                "color": color,
            }
            for size, color in [(512, "#2171b5"), (1024, "#08306b")]
            for optimizer, subdir in [("Adam", ""), ("SGD", "sgd")]
        ]

    colors = sns.color_palette("colorblind", n_colors=len(args.run) + 3)
    configs.extend(
        {"name": name, "path": path, "color": colors[index + 3]}
        for index, (name, path) in enumerate(args.run)
    )
    rows = []
    for cfg in configs:
        with open(cfg["path"]) as stream:
            history = json.load(stream)["history"]
        for record in history:
            if record["phase"] == "relearn" and args.metric in record:
                rows.append(
                    {
                        "epochs": record["step"],
                        "value": record[args.metric]
                        * (100 if args.metric == "test_accuracy" else 1),
                        "method": cfg["name"],
                    }
                )

    plot = sns.relplot(
        data=pd.DataFrame(rows),
        x="epochs",
        y="value",
        hue="method",
        palette={cfg["name"]: cfg["color"] for cfg in configs},
        kind="line",
        height=3,
        aspect=1,
    )
    plot.axes[0, 0].set_title("Full-batch attacks on ResNet-18")
    plot.set_axis_labels(
        "Relearning updates",
        {
            "test_accuracy": "Test accuracy (%)",
            "loss": "Training cross-entropy",
            "individual_pairwise_cosine": "Mean individual pairwise cosine",
            "diagnostic_group_cosine": "Group-gradient cosine (diagnostic batch)",
        }[args.metric],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot.savefig(args.output)
    sys.exit()
if not args.root:
    parser.error("Specify --root or --fullbatch-root")
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
