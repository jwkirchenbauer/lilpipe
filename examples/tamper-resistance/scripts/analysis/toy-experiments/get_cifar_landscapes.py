"""Evaluate CIFAR-10 loss landscapes around random and defended weights."""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models

from run_cifar_defence_and_attack import load_data, norm


def directions(parameters, seed):
    """Return two orthonormal global-L2 directions shared across checkpoints."""
    generator = torch.Generator(device=parameters[0].device).manual_seed(seed)
    first, second = [
        [
            torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=generator)
            for p in parameters
        ]
        for _ in range(2)
    ]
    length = norm(first)
    first = [d / length for d in first]
    projection = sum((a * b).sum() for a, b in zip(first, second))
    second = [b - projection * a for a, b in zip(first, second)]
    length = norm(second)
    second = [d / length for d in second]
    return first, second


@torch.no_grad()
def landscape(model, batches, axes, first, second):
    parameters = list(model.parameters())
    center = [p.clone() for p in parameters]
    losses = []
    try:
        for x in axes:
            row = []
            for y in axes:
                for p, c, a, b in zip(parameters, center, first, second):
                    p.copy_(c + float(x) * a + float(y) * b)
                total = sum(
                    nn.functional.cross_entropy(model(images), labels, reduction="sum")
                    for images, labels in batches
                )
                row.append((total / sum(len(labels) for _, labels in batches)).item())
            losses.append(row)
    finally:
        for p, c in zip(parameters, center):
            p.copy_(c)
    return losses


def main(args):
    device = torch.device("cuda")
    _, train, _, channels = load_data()
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[
        :1024
    ]
    batches = [
        (images.to(device), labels.to(device))
        for images, labels in DataLoader(
            Subset(train, indices.tolist()), batch_size=256
        )
    ]
    torch.manual_seed(0)  # Original experiment initialization seed.
    model = models.resnet18(
        weights=None, num_classes=10, norm_layer=lambda c: nn.GroupNorm(8, c)
    )
    model.conv1 = nn.Conv2d(channels, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.to(device).eval()
    first, second = directions(list(model.parameters()), args.seed)
    initial = {name: p.clone() for name, p in model.state_dict().items()}
    configs = [("Random initialization", None)]
    for k, folder in [(1, "tar-k1"), (8, "tar-k8"), (16, "tar"), (32, "tar-k32")]:
        configs.append((f"TAR k={k}, defended", f"{folder}/train.pt"))
    for variant in ["nll", "ent"]:
        configs.append(
            (f"SAM ({variant}), rho=0.01, defended", f"sam-{variant}-rho0.01/train.pt")
        )
    result = dict(
        seed=args.seed,
        subset_seed=0,
        subset_size=1024,
        directions="shared orthonormal random directions; global L2 norm 1",
        metric="mean training cross-entropy",
        radii=args.radii,
        points=args.points,
        models=[],
    )
    cached = {}
    if args.resume and os.path.exists(args.output + ".json"):
        with open(args.output + ".json") as stream:
            previous = json.load(stream)
        if any(
            previous.get(key) != value
            for key, value in result.items()
            if key != "models"
        ):
            raise ValueError("Cannot reuse grids with different landscape settings")
        cached = {record["name"]: record for record in previous["models"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for name, relative in configs:
        checkpoint = os.path.join(args.root, relative) if relative else None
        if name in cached and cached[name]["checkpoint"] == checkpoint:
            result["models"].append(cached[name])
            print(f"Reusing {name}", flush=True)
            continue
        state = (
            torch.load(checkpoint, map_location=device, weights_only=True)
            if checkpoint
            else initial
        )
        model.load_state_dict(state)
        grids = []
        for radius in args.radii:
            axes = np.linspace(-radius, radius, args.points)
            losses = landscape(model, batches, axes, first, second)
            grids.append(dict(radius=radius, axes=axes.tolist(), losses=losses))
            print(f"Finished {name}, radius {radius}", flush=True)
        result["models"].append(dict(name=name, checkpoint=checkpoint, grids=grids))
        with open(args.output + ".json", "w") as stream:
            json.dump(result, stream, indent=2)
    with open(args.output + ".json", "w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--radii", type=float, nargs="+", default=[1000.0, 5000.0])
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/figures/resnet18-cifar10-tar-loss-landscapes"
    )
    args = parser.parse_args()
    if (
        any(radius <= 0 for radius in args.radii)
        or args.points < 3
        or args.points % 2 != 1
    ):
        parser.error("Use a positive radius and an odd grid size of at least 3.")
    main(args)
