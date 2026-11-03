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


def hessian_vector(model, batches, vector):
    """Compute the full-subset NLL Hessian-vector product in memory-sized batches."""
    parameters = tuple(model.parameters())
    pieces = vector.split([p.numel() for p in parameters])
    vector_parts = [v.reshape_as(p) for v, p in zip(pieces, parameters)]
    total = sum(len(labels) for _, labels in batches)
    result = torch.zeros_like(vector)
    for images, labels in batches:
        loss = (
            nn.functional.cross_entropy(model(images), labels, reduction="sum") / total
        )
        gradient = torch.autograd.grad(loss, parameters, create_graph=True)
        product = sum((g * v).sum() for g, v in zip(gradient, vector_parts))
        hv = torch.autograd.grad(product, parameters)
        result += torch.cat([v.detach().flatten() for v in hv])
    return result


def curvature_directions(model, batches, seed, steps):
    """Find low-curvature Ritz directions; these estimates do not certify a minimum."""
    parameters = tuple(model.parameters())
    generator = torch.Generator(device=parameters[0].device).manual_seed(seed)
    size = sum(p.numel() for p in parameters)
    vector = torch.randn(size, device=parameters[0].device, generator=generator)
    vector /= vector.norm()
    basis, diagonal, off_diagonal = [], [], []
    for _ in range(steps):
        basis.append(vector)
        residual = hessian_vector(model, batches, vector)
        diagonal.append(torch.dot(vector, residual).item())
        # Full reorthogonalization avoids repeated copies of the same eigenvector.
        for _ in range(2):
            for q in basis:
                residual -= torch.dot(q, residual) * q
        length = residual.norm().item()
        if length < 1e-7:
            break
        off_diagonal.append(length)
        vector = residual / length
    matrix = torch.diag(torch.tensor(diagonal, dtype=torch.float64))
    if len(basis) > 1:
        off = torch.tensor(off_diagonal[: len(basis) - 1], dtype=torch.float64)
        matrix += torch.diag(off, 1) + torch.diag(off, -1)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    vectors = []
    for i in range(min(2, len(basis))):
        v = sum(float(c) * q for c, q in zip(eigenvectors[:, i], basis))
        vectors.append(v / v.norm())
    if len(vectors) < 2:
        v = torch.randn(size, device=vector.device, generator=generator)
        v -= torch.dot(v, vectors[0]) * vectors[0]
        vectors.append(v / v.norm())
    residual = (
        (
            hessian_vector(model, batches, vectors[0])
            - float(eigenvalues[0]) * vectors[0]
        )
        .norm()
        .item()
    )
    parts = [
        [
            v.reshape_as(p)
            for v, p in zip(vector.split([p.numel() for p in parameters]), parameters)
        ]
        for vector in vectors
    ]
    return parts, dict(
        min_ritz_eigenvalue=eigenvalues[0].item(),
        eigen_residual=residual,
        lanczos_steps=len(basis),
    )


@torch.no_grad()
def activation_metrics(model, batches):
    """Check whether every final-block ReLU is strictly inactive on the subset."""
    stats = {}
    maxima, positive, count = [], 0, 0

    def capture(module, inputs):
        # Called twice per BasicBlock; the last invocation is the output ReLU.
        value = inputs[0].detach()
        stats.update(
            maximum=value.max().item(),
            positive=(value > 0).sum().item(),
            count=value.numel(),
        )

    hook = model.layer4[-1].relu.register_forward_pre_hook(capture)
    try:
        for images, _ in batches:
            model(images)
            maxima.append(stats["maximum"])
            positive += stats["positive"]
            count += stats["count"]
    finally:
        hook.remove()
    return dict(
        final_relu_max_input=max(maxima), final_relu_active_fraction=positive / count
    )


def gradient_metrics(model, batches):
    parameters = tuple(model.parameters())
    gradient = [torch.zeros_like(p) for p in parameters]
    total = sum(len(labels) for _, labels in batches)
    value, correct = 0.0, 0
    for images, labels in batches:
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum") / total
        grads = torch.autograd.grad(loss, parameters)
        for g, part in zip(gradient, grads):
            g.add_(part.detach())
        value += loss.item()
        correct += (logits.argmax(-1) == labels).sum().item()
    return (
        dict(
            nll=value,
            train_accuracy=correct / total,
            gradient_norm=norm(gradient).item(),
        ),
        gradient,
    )


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
    if args.indices:
        with open(args.indices) as stream:
            indices = torch.tensor(json.load(stream)["training"])
    batches = [
        (images.to(device), labels.to(device))
        for images, labels in DataLoader(
            Subset(train, indices.tolist()), batch_size=args.batch_size
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
    if args.model:
        configs = [("Random initialization", None)] + args.model
    result = dict(
        seed=args.seed,
        subset_seed=0,
        subset_size=len(indices),
        training_indices=indices.tolist(),
        directions=args.directions,
        lanczos_steps=(
            args.lanczos_steps if args.directions in ["curvature", "descent"] else None
        ),
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
        metrics, gradient = gradient_metrics(model, batches)
        metrics.update(activation_metrics(model, batches))
        model_first, model_second = first, second
        if args.directions == "head":
            model_first = [torch.zeros_like(p) for p in model.parameters()]
            model_second = [torch.zeros_like(p) for p in model.parameters()]
            bias_index = list(dict(model.named_parameters())).index("fc.bias")
            model_first[bias_index][0] = 1 / 2**0.5
            model_first[bias_index][1] = -1 / 2**0.5
            model_second[bias_index][0] = 1 / 6**0.5
            model_second[bias_index][1] = 1 / 6**0.5
            model_second[bias_index][2] = -2 / 6**0.5
        elif args.directions != "random":
            (model_first, model_second), curvature = curvature_directions(
                model, batches, args.seed, args.lanczos_steps
            )
            metrics.update(curvature)
            if args.directions == "descent":
                length = norm(gradient)
                if length.item() > 1e-10:
                    model_first = [g / length for g in gradient]
                    projection = sum(
                        (a * b).sum() for a, b in zip(model_first, model_second)
                    )
                    model_second = [
                        b - projection * a for a, b in zip(model_first, model_second)
                    ]
                    length = norm(model_second)
                    model_second = [b / length for b in model_second]
        print(json.dumps(dict(model=name, **metrics)), flush=True)
        grids = []
        for radius in args.radii:
            axes = np.linspace(-radius, radius, args.points)
            losses = landscape(model, batches, axes, model_first, model_second)
            grids.append(dict(radius=radius, axes=axes.tolist(), losses=losses))
            print(f"Finished {name}, radius {radius}", flush=True)
        result["models"].append(
            dict(name=name, checkpoint=checkpoint, grids=grids, metrics=metrics)
        )
        with open(args.output + ".json", "w") as stream:
            json.dump(result, stream, indent=2)
    with open(args.output + ".json", "w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--model", action="append", nargs=2, metavar=("NAME", "CHECKPOINT")
    )
    parser.add_argument(
        "--indices", help="Saved indices.json defining the training subset"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--directions",
        choices=("random", "curvature", "descent", "head"),
        default="random",
    )
    parser.add_argument("--lanczos-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
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
