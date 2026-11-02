"""Run CIFAR-10 BaSFT, TAR, or SAM defense followed by ordinary SFT attacks."""

import argparse
import copy
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


def load_data():
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "cifar10"
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
    train_eval, test_eval = [
        TensorDataset(
            torch.as_tensor(data.data).permute(0, 3, 1, 2).float() / 255,
            torch.as_tensor(data.targets),
        )
        for data in [train, test]
    ]
    return train, train_eval, test_eval, 3


@torch.no_grad()
def evaluate(model, dataset, device):
    correct = 0
    for images, labels in DataLoader(dataset, batch_size=256):
        predictions = model(images.to(device)).argmax(1).cpu()
        correct += (predictions == labels).sum().item()
    return correct / len(dataset)


def defense_loss(logits, labels, objective):
    """Minimize negative NLL or negative entropy to suppress classification."""
    if objective == "nll":
        return -nn.functional.cross_entropy(logits, labels)
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return (log_probs.exp() * log_probs).sum(-1).mean()


def sam_loss(model, images, labels, objective, rho):
    """Evaluate the defense loss after global-L2 gradient ascent, holding delta fixed."""
    parameters = dict(model.named_parameters())
    loss = defense_loss(model(images), labels, objective)
    gradients = torch.autograd.grad(loss, tuple(parameters.values()))
    length = norm(gradients).clamp_min(1e-12)
    point = {
        name: parameter + (rho * gradient / length).detach()
        for (name, parameter), gradient in zip(parameters.items(), gradients)
    }
    return defense_loss(
        torch.func.functional_call(model, point, (images,)), labels, objective
    )


def tar_loss(model, attack_stream, defense_stream, steps, learning_rate):
    """Evaluate negative entropy at attacked weights with an identity outer derivative."""
    attacker = copy.deepcopy(model)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=learning_rate)
    for _ in range(steps):
        images, labels = next(attack_stream)
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(attacker(images), labels).backward()
        optimizer.step()
    point = {
        name: parameter + (attacked - parameter).detach()
        for (name, parameter), (_, attacked) in zip(
            model.named_parameters(), attacker.named_parameters()
        )
    }
    images, labels = next(defense_stream)
    logits = torch.func.functional_call(model, point, (images,))
    return defense_loss(logits, labels, "ent")


def relearn(model, train, test, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.relearn_learning_rate)
    stream = batches(train, args.batch_size, device)
    history = []
    for step in range(args.relearn_steps + 1):
        images, labels = next(stream)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        if step % args.log_every == 0 or step == args.relearn_steps:
            row = dict(
                method=args.method,
                phase="relearn",
                step=step,
                loss=loss.item(),
                gradient_norm=norm(p.grad for p in model.parameters()).item(),
                test_accuracy=evaluate(model, test, device),
            )
            history.append(row)
            print(json.dumps(row), flush=True)
        if step < args.relearn_steps:
            optimizer.step()
    return history


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    train, _, test, channels = load_data()
    indices = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))[
        :1024
    ]
    test = torch.utils.data.Subset(test, indices.tolist())
    torch.manual_seed(args.seed)
    model = models.resnet18(
        weights=None, num_classes=10, norm_layer=lambda c: nn.GroupNorm(8, c)
    )
    model.conv1 = nn.Conv2d(channels, 64, 3, padding=1, bias=False)
    model.maxpool = nn.Identity()
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        )
    model.to(device).eval()
    result = dict(config=vars(args), history=[])
    if args.method != "random" and not args.checkpoint:
        torch.manual_seed(args.seed + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        attack_stream = batches(train, args.batch_size, device)
        defense_stream = batches(train, args.batch_size, device)
        for step in range(args.steps):
            if args.method == "tar":
                loss = tar_loss(
                    model,
                    attack_stream,
                    defense_stream,
                    args.attack_steps,
                    args.attack_learning_rate,
                )
            elif args.method == "basft":
                images, labels = next(attack_stream)
                center_loss = nn.functional.cross_entropy(model(images), labels)
                penalty = center_loss.new_zeros(())
                choices = [
                    (radius, lr) for radius in args.radii for lr in args.probe_lrs
                ]
                radius, lr = choices[step % len(choices)]
                for inner_steps in [0, args.probe_steps]:
                    delta = perturbation(
                        model, (images, labels), radius, lr, inner_steps
                    )
                    penalty = (
                        penalty
                        + basin_penalty(
                            model, (images, labels), center_loss, delta, args.curvature
                        )
                        / 2
                    )
                loss = (
                    args.alpha * (center_loss - args.target_loss).square()
                    + args.beta * penalty
                )
            else:
                images, labels = next(attack_stream)
                loss = sam_loss(
                    model, images, labels, args.method.removeprefix("sam-"), args.rho
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step % 100 == 0 or step == args.steps - 1:
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite defense loss")
                row = dict(
                    method=args.method,
                    phase="defense",
                    step=step,
                    objective=loss.item(),
                )
                result["history"].append(row)
                print(json.dumps(row), flush=True)
            optimizer.step()
    torch.save(model.state_dict(), os.path.join(args.output, "train.pt"))
    torch.manual_seed(args.seed + 1)
    result["history"].extend(relearn(model, train, test, args, device))
    torch.save(model.state_dict(), os.path.join(args.output, "relearn.pt"))
    with open(os.path.join(args.output, "results.json"), "w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("random", "basft", "tar", "sam-nll", "sam-ent"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", help="Skip defense and relearn from these weights"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--relearn-steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--relearn-learning-rate", type=float, default=0.001)
    parser.add_argument("--attack-steps", type=int, default=16)
    parser.add_argument("--attack-learning-rate", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--target-loss", type=float, default=2.3)
    parser.add_argument("--curvature", type=float, default=0.1)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument(
        "--probe-lrs", type=float, nargs="+", default=[0.001, 0.01, 0.1]
    )
    parser.add_argument("--probe-steps", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.relearn_steps,
            args.log_every,
            args.batch_size,
            args.attack_steps,
        )
        < 1
    ):
        parser.error("Step counts, log interval, and batch size must be positive")
    if (
        min(
            args.learning_rate,
            args.relearn_learning_rate,
            args.attack_learning_rate,
            args.rho,
        )
        <= 0
    ):
        parser.error("Learning rates and rho must be positive")
    main(args)
