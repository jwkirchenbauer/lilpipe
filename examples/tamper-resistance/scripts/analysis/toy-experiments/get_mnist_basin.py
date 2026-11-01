"""Compare SFT and BaSFT on MNIST or CIFAR-10 with a GroupNorm ResNet-18."""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms


def norm(values):
    return sum(value.square().sum() for value in values).sqrt()


def perturbation(model, batch, radius, learning_rate, steps):
    """Random L2-sphere start, optionally followed by projected forget-loss descent."""
    center = dict(model.named_parameters())
    delta = {name: torch.randn_like(value) for name, value in center.items()}
    scale = radius / norm(delta.values()).clamp_min(1e-12)
    delta = {name: (value * scale).detach() for name, value in delta.items()}
    for _ in range(steps):
        point = {
            name: (value.detach() + delta[name]).requires_grad_()
            for name, value in center.items()
        }
        loss = F.cross_entropy(functional_call(model, point, (batch[0],)), batch[1])
        gradient = torch.autograd.grad(loss, tuple(point.values()))
        delta = {
            name: delta[name] - learning_rate * grad.detach()
            for name, grad in zip(center, gradient)
        }
        scale = (radius / norm(delta.values()).clamp_min(1e-12)).clamp(max=1)
        delta = {name: (value * scale).detach() for name, value in delta.items()}
    return delta


def basin_penalty(model, batch, center_loss, delta, curvature):
    """Differentiate both endpoint losses with respect to theta, holding delta fixed."""
    point = {name: value + delta[name] for name, value in model.named_parameters()}
    neighbor_loss = F.cross_entropy(
        functional_call(model, point, (batch[0],)), batch[1]
    )
    distance_squared = sum(value.square().sum() for value in delta.values())
    return F.relu(curvature * distance_squared + center_loss - neighbor_loss)


def batches(dataset, batch_size, device):
    generator = torch.Generator().manual_seed(0)
    while True:
        for images, labels in DataLoader(
            dataset, batch_size=batch_size, shuffle=True, generator=generator
        ):
            yield images.to(device), labels.to(device)


def evaluate(model, dataset, device, with_gradient=False):
    model.eval()
    if with_gradient:
        model.zero_grad(set_to_none=True)
    loss, correct = 0.0, 0
    with torch.set_grad_enabled(with_gradient):
        for images, labels in DataLoader(dataset, batch_size=256):
            logits = model(images.to(device))
            batch_loss = F.cross_entropy(logits, labels.to(device), reduction="sum")
            if with_gradient:
                (batch_loss / len(dataset)).backward()
            loss += batch_loss.item()
            correct += (logits.argmax(1).cpu() == labels).sum().item()
    metrics = {"loss": loss / len(dataset), "accuracy": correct / len(dataset)}
    if with_gradient:
        metrics["gradient_norm"] = norm(
            p.grad for p in model.parameters() if p.grad is not None
        ).item()
        model.zero_grad(set_to_none=True)
    return metrics


def fit(model, train, test, args, device, method, phase, train_eval=None):
    model.eval()  # GroupNorm has identical train/eval behavior and no running buffers.
    training = phase == "train"
    steps = args.steps if training else args.relearn_steps
    interval = args.log_every if training else args.relearn_log_every
    lr = args.learning_rate if training else args.relearn_learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stream = batches(train, args.batch_size, device)
    choices = list(itertools.product(args.radii, args.probe_lrs))
    history = []
    for step in range(steps + 1):
        if step % interval == 0 or step == steps:
            train_metrics = evaluate(
                model,
                train if train_eval is None else train_eval,
                device,
                with_gradient=True,
            )
            row = {
                "method": method,
                "phase": phase,
                "step": step,
                "loss": train_metrics["loss"],
                "gradient_norm": train_metrics["gradient_norm"],
                "test_accuracy": evaluate(model, test, device)["accuracy"],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step == steps:
            break
        batch = next(stream)
        center_loss = F.cross_entropy(model(batch[0]), batch[1])
        objective = center_loss
        if training and method.startswith("BaSFT"):
            penalty = center_loss.new_zeros(())
            if args.beta:
                radius, learning_rate = choices[step % len(choices)]
                for inner_steps in (0, args.probe_steps):
                    delta = perturbation(
                        model, batch, radius, learning_rate, inner_steps
                    )
                    penalty = (
                        penalty
                        + basin_penalty(
                            model, batch, center_loss, delta, args.curvature
                        )
                        / 2
                    )
            objective = (
                args.alpha * (center_loss - args.target_loss).square()
                + args.beta * penalty
            )
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    return history


def load_data(dataset):
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / dataset
    if dataset == "cifar10":
        train = datasets.CIFAR10(
            root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.ToTensor(),
                ]
            ),
        )
        test = datasets.CIFAR10(root, train=False, download=True)
        # Use fixed images for loss and gradient measurements.
        train_eval, test_eval = [
            TensorDataset(
                torch.as_tensor(data.data).permute(0, 3, 1, 2).float() / 255,
                torch.as_tensor(data.targets),
            )
            for data in (train, test)
        ]
        return train, train_eval, test_eval, 3
    train, test = [
        TensorDataset(
            (data.data.unsqueeze(1).float() / 255 - 0.1307) / 0.3081, data.targets
        )
        for data in [
            datasets.MNIST(root, train=split, download=True) for split in (True, False)
        ]
    ]
    return train, train, test, 1


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train, train_eval, test, channels = load_data(getattr(args, "dataset", "mnist"))
    initial = models.resnet18(
        weights=None,
        num_classes=10,
        norm_layer=lambda channels: nn.GroupNorm(8, channels),
    )
    initial.conv1 = nn.Conv2d(channels, 64, 3, padding=1, bias=False)
    initial.maxpool = nn.Identity()
    initial.to(device)
    result = {
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "training_examples": len(train),
        "test_examples": len(test),
        "history": [],
    }
    methods = [("SFT", None)] + [
        (f"BaSFT ($\\beta={beta}$)", float(beta))
        for beta in (args.betas or [str(args.beta)])
    ]
    results_file = args.output / "results.json"
    if args.resume and results_file.exists():
        previous = json.loads(results_file.read_text())
        for key, value in result["config"].items():
            if key not in ("output", "beta", "betas", "resume"):
                default = "mnist" if key == "dataset" else None
                if previous["config"].get(key, default) != value:
                    raise ValueError(f"Cannot reuse runs with different {key}")
        for row in previous["history"]:
            if row["method"] == "BaSFT":
                row["method"] = f"BaSFT ($\\beta={previous['config']['beta']}$)"
            if row["method"] in dict(methods):
                result["history"].append(row)
    for method, beta in methods:
        rows = [row for row in result["history"] if row["method"] == method]
        if all("gradient_norm" in row for row in rows) and all(
            any(row["phase"] == phase and row["step"] == steps for row in rows)
            for phase, steps in (("train", args.steps), ("relearn", args.relearn_steps))
        ):
            print(f"Reusing completed run: {method}", flush=True)
            continue
        result["history"] = [
            row for row in result["history"] if row["method"] != method
        ]
        run_args = argparse.Namespace(**vars(args))
        if beta is not None:
            run_args.beta = beta
        torch.manual_seed(args.seed + 1)
        model = copy.deepcopy(initial)
        for phase in ("train", "relearn"):
            result["history"].extend(
                fit(
                    model,
                    train,
                    test,
                    run_args,
                    device,
                    method,
                    phase,
                    train_eval=train_eval,
                )
            )
            slug = "SFT" if beta is None else f"BaSFT-beta-{beta}"
            torch.save(model.state_dict(), args.output / f"{slug}-{phase}.pt")
            results_file.write_text(json.dumps(result, indent=2))
    results_file.write_text(json.dumps(result, indent=2))
    print(f"Results: {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), default="mnist")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--relearn-steps", type=int, default=100)
    parser.add_argument("--relearn-log-every", type=int, default=10)
    parser.add_argument("--relearn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-loss", type=float, default=2.3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--betas", nargs="+", help="BaSFT beta values and legend labels"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Reuse completed compatible runs"
    )
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument(
        "--probe-lrs", type=float, nargs="+", default=[0.001, 0.01, 0.1]
    )
    parser.add_argument("--probe-steps", type=int, default=3)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            Path(os.environ.get("SCRATCH", ".")) / f"lilpipe/{args.dataset}-sft-basft"
        )
    if min(args.steps, args.relearn_steps, args.log_every, args.relearn_log_every) < 1:
        parser.error("Step counts and log intervals must be positive.")
    main(args)
