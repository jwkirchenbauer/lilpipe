"""Run CIFAR-10 BaSFT, TAR, or SAM defense followed by ordinary SFT attacks."""

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path

import torch
from torch import nn

from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, models, transforms


def norm(values):
    return sum(value.square().sum() for value in values).sqrt()


def perturbation(model, batch, radius, learning_rate, steps, random_start=True):
    """Random L2-sphere start, optionally followed by projected forget-loss descent."""
    center = dict(model.named_parameters())
    delta = {
        name: torch.randn_like(value) if random_start else torch.zeros_like(value)
        for name, value in center.items()
    }
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


def dead_relu_initialization(model, labels):
    """Construct a constant-feature local minimum; all parameters remain trainable."""
    probabilities = torch.bincount(labels, minlength=model.fc.out_features).float()
    probabilities /= probabilities.sum()
    if (probabilities == 0).any():
        raise ValueError("The dead-ReLU control requires every class in the subset")
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                module.weight.zero_()
                module.bias.fill_(-1)
        model.fc.weight.zero_()
        model.fc.bias.copy_(probabilities.log())


@torch.no_grad()
def project_inactive_margin(model, images, margin):
    """Shift the final normalization bias to make its ReLU inactive on every sample."""
    stats = {}

    def capture(module, inputs):
        stats["maximum"] = inputs[0].amax(dim=(0, 2, 3))

    block = model.layer4[-1]
    hook = block.relu.register_forward_pre_hook(capture)
    try:
        model(images)
    finally:
        hook.remove()
    shift = (stats["maximum"] + margin).clamp_min(0)
    block.bn2.bias.sub_(shift)
    return dict(margin=margin, maximum_bias_shift=shift.max().item())


@torch.no_grad()
def calibrate_constant_head(model, images, labels):
    """Calibrate class priors only after verifying a strictly inactive final ReLU."""
    stats = {}

    def capture(module, inputs):
        stats["max_input"] = inputs[0].max().item()

    hook = model.layer4[-1].relu.register_forward_pre_hook(capture)
    try:
        model(images)
    finally:
        hook.remove()
    if stats["max_input"] >= 0:
        raise ValueError(
            "Final features are not strictly inactive; cannot certify a constant-feature basin"
        )
    probabilities = torch.bincount(labels, minlength=model.fc.out_features).float()
    probabilities /= probabilities.sum()
    if (probabilities == 0).any():
        raise ValueError("Class-prior calibration requires every class in the subset")
    model.fc.bias.copy_(probabilities.log())
    return dict(
        final_relu_max_input=stats["max_input"],
        class_probabilities=probabilities.tolist(),
    )


def bad_minimum_loss(
    model,
    images,
    labels,
    target,
    gradient_weight,
    escape_weight,
    radius,
    probe_lr,
    probe_steps,
    curvature,
    feature_weight=0,
    feature_margin=0.1,
):
    """Target chance NLL, stationarity, and no nearby loss-decreasing escape."""
    preactivations = []
    if feature_weight:
        # BasicBlock reuses an in-place ReLU; clone its final input before mutation.
        hook = model.layer4[-1].relu.register_forward_pre_hook(
            lambda module, inputs: preactivations.append(inputs[0].clone())
        )
        try:
            logits = model(images)
        finally:
            hook.remove()
    else:
        logits = model(images)
    loss = F.cross_entropy(logits, labels)
    objective = (loss - target).square()
    if feature_weight:
        objective = (
            objective
            + feature_weight
            * F.relu(preactivations[-1] + feature_margin).square().mean()
        )
    if gradient_weight:
        gradient = torch.autograd.grad(
            loss, tuple(model.parameters()), create_graph=True
        )
        objective = objective + gradient_weight * sum(
            g.square().sum() for g in gradient
        )
    if escape_weight:
        penalty = loss.new_zeros(())
        for steps, random_start in [
            (0, True),
            (probe_steps, True),
            (probe_steps, False),
        ]:
            delta = perturbation(
                model,
                (images, labels),
                radius,
                probe_lr,
                steps,
                random_start=random_start,
            )
            penalty = (
                penalty
                + basin_penalty(model, (images, labels), loss, delta, curvature) / 3
            )
        objective = objective + escape_weight * penalty
    return objective


def tar_loss(
    model, attack_stream, defense_stream, steps, learning_rate, on_inner_loss=None
):
    """Evaluate negative entropy at attacked weights with an identity outer derivative."""
    attacker = copy.deepcopy(model)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=learning_rate)
    microbatch_size = None
    for inner_step in range(steps):
        images, labels = next(attack_stream)
        if microbatch_size is None:
            microbatch_size = len(labels)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                inner_loss = batch_backward(attacker, images, labels, microbatch_size)
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch_size == 1:
                    raise
                microbatch_size = max(1, microbatch_size // 2)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(
                    f"TAR inner OOM: retrying with microbatches of {microbatch_size}",
                    flush=True,
                )
        if on_inner_loss is not None:
            on_inner_loss(inner_step, inner_loss.item())
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


def differentiable_adam_step(model, names, step, learning_rate, *values):
    """One out-of-place Adam update, including differentiable moment states."""
    count = len(names)
    weights = values[:count]
    first = values[count : 2 * count]
    second = values[2 * count : 3 * count]
    images, labels = values[-2:]
    logits = functional_call(model, dict(zip(names, weights)), (images,))
    loss = F.cross_entropy(logits, labels)
    gradients = torch.autograd.grad(loss, weights, create_graph=True)
    first = tuple(0.9 * m + 0.1 * g for m, g in zip(first, gradients))
    second = tuple(0.999 * v + 0.001 * g.square() for v, g in zip(second, gradients))
    updated = []
    for weight, m, v in zip(weights, first, second):
        variance = v / (1 - 0.999**step)
        positive = variance > 0
        # Adam's composite update has a finite derivative at a zero gradient.
        # Avoid the spurious 0 * inf from sqrt's derivative at zero moments.
        root = torch.where(positive, variance, torch.ones_like(variance)).sqrt()
        denominator = torch.where(positive, root, torch.zeros_like(root)) + 1e-8
        updated.append(weight - learning_rate * (m / (1 - 0.9**step)) / denominator)
    return (*updated, *first, *second, loss)


def full_tar_loss(
    model,
    attack_stream,
    defense_stream,
    steps,
    learning_rate,
    on_inner_loss=None,
    checkpoint_steps=True,
):
    """Differentiate negative entropy through every inner Adam update."""
    names, weights = zip(*model.named_parameters())
    count = len(weights)
    first = tuple(torch.zeros_like(p) for p in weights)
    second = tuple(torch.zeros_like(p) for p in weights)
    for step in range(1, steps + 1):
        images, labels = next(attack_stream)
        # Bind the step now: checkpoint recomputation happens after this loop.
        update = partial(differentiable_adam_step, model, names, step, learning_rate)
        inputs = (*weights, *first, *second, images, labels)
        outputs = (
            checkpoint(update, *inputs, use_reentrant=False)
            if checkpoint_steps
            else update(*inputs)
        )
        weights = outputs[:count]
        first = outputs[count : 2 * count]
        second = outputs[2 * count : 3 * count]
        if on_inner_loss is not None:
            on_inner_loss(step - 1, outputs[-1].item())
    images, labels = next(defense_stream)
    logits = functional_call(model, dict(zip(names, weights)), (images,))
    return defense_loss(logits, labels, "ent")


def batch_backward(model, images, labels, microbatch_size):
    """Accumulate the gradient of mean NLL over the full effective batch."""
    total = images.new_zeros(())
    for start in range(0, len(labels), microbatch_size):
        end = start + microbatch_size
        loss = nn.functional.cross_entropy(
            model(images[start:end]), labels[start:end], reduction="sum"
        ) / len(labels)
        loss.backward()
        total += loss.detach()
    return total


def relearn(model, train, test, args, device):
    optimizer_class = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.relearn_learning_rate)
    stream = batches(train, args.batch_size, device)
    history = []
    microbatch_size = args.batch_size
    for step in range(args.relearn_steps + 1):
        images, labels = next(stream)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                loss = batch_backward(model, images, labels, microbatch_size)
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch_size == 1:
                    raise
                microbatch_size = max(1, microbatch_size // 2)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(
                    f"OOM: retrying full batch with microbatches of {microbatch_size}",
                    flush=True,
                )
        if step == 0 and device.type == "cuda":
            print(
                f"Peak GPU allocation after backward: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
                flush=True,
            )
        if step % args.log_every == 0 or step == args.relearn_steps:
            row = dict(
                method=args.method,
                phase="relearn",
                step=step,
                loss=loss.item(),
                gradient_norm=norm(p.grad for p in model.parameters()).item(),
                test_accuracy=evaluate(model, test, device),
                microbatch_size=microbatch_size,
            )
            history.append(row)
            print(json.dumps(row), flush=True)
        if step < args.relearn_steps:
            optimizer.step()
    return history


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    train, train_eval, test, channels = load_data()
    training_indices = None
    if args.train_samples:
        indices = torch.randperm(
            len(train_eval), generator=torch.Generator().manual_seed(0)
        )[: args.train_samples]
        # Fixed images: full-batch steps optimize the same empirical objective.
        training_indices = indices.tolist()
        train = torch.utils.data.Subset(train_eval, training_indices)
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
    initialization = args.checkpoint or args.defend_checkpoint
    if initialization:
        model.load_state_dict(
            torch.load(initialization, map_location="cpu", weights_only=True)
        )
    model.to(device).eval()
    result = dict(
        config=vars(args),
        training_indices=training_indices,
        test_indices=indices.tolist(),
        history=[],
    )
    with open(os.path.join(args.output, "indices.json"), "w") as stream:
        json.dump(
            dict(training=training_indices, test=indices.tolist()), stream, indent=2
        )
    if args.method.startswith("bad-"):
        labels = torch.tensor([train[i][1] for i in range(len(train))])
        probabilities = torch.bincount(labels, minlength=10).float() / len(labels)
        target = (
            -(probabilities[probabilities > 0] * probabilities[probabilities > 0].log())
            .sum()
            .item()
        )
        result["target_loss"] = target
    if args.method == "bad-dead-relu" and not args.checkpoint:
        dead_relu_initialization(model, labels.to(device))
    if args.method not in ["random", "bad-dead-relu"] and not args.checkpoint:
        torch.manual_seed(args.seed + 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        attack_stream = batches(train, args.batch_size, device)
        defense_stream = batches(train, args.batch_size, device)
        for step in range(args.steps):

            def log_inner(inner_step, loss):
                row = dict(
                    method=args.method,
                    phase="inner",
                    step=step * args.attack_steps + inner_step + 1,
                    defense_step=step,
                    inner_step=inner_step + 1,
                    loss=loss,
                )
                result["history"].append(row)
                print(json.dumps(row), flush=True)

            if args.method in ["tar", "tar-full"]:
                attack_loss = full_tar_loss if args.method == "tar-full" else tar_loss
                loss = attack_loss(
                    model,
                    attack_stream,
                    defense_stream,
                    args.attack_steps,
                    args.attack_learning_rate,
                    on_inner_loss=log_inner,
                )
            elif args.method.startswith("bad-"):
                images, labels = next(attack_stream)
                loss = bad_minimum_loss(
                    model,
                    images,
                    labels,
                    target,
                    args.gradient_weight if args.method != "bad-target" else 0,
                    args.beta if args.method == "bad-basin" else 0,
                    args.rho,
                    args.attack_learning_rate,
                    args.probe_steps,
                    args.curvature,
                    args.feature_weight if args.method == "bad-margin" else 0,
                    args.feature_margin,
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
            if step % args.log_every == 0 or step == args.steps - 1:
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite defense loss")
                row = dict(
                    method=args.method,
                    phase="defense",
                    step=step,
                    objective=loss.item(),
                    gradient_norm=norm(
                        p.grad.double()
                        for p in model.parameters()
                        if p.grad is not None
                    ).item(),
                )
                if args.method == "tar-full":
                    if not all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    ):
                        raise ValueError("Nonfinite full-TAR outer gradient")
                    if device.type == "cuda":
                        row["peak_gpu_gib"] = (
                            torch.cuda.max_memory_allocated() / 1024**3
                        )
                result["history"].append(row)
                print(json.dumps(row), flush=True)
            optimizer.step()
    if args.calibrate_constant_head:
        images, labels = next(iter(DataLoader(train, batch_size=len(train))))
        if args.project_feature_margin is not None:
            result["margin_projection"] = project_inactive_margin(
                model, images.to(device), args.project_feature_margin
            )
        result["calibration"] = calibrate_constant_head(
            model, images.to(device), labels.to(device)
        )
    torch.save(model.state_dict(), os.path.join(args.output, "train.pt"))
    with open(os.path.join(args.output, "results.json"), "w") as stream:
        json.dump(result, stream, indent=2)
    torch.manual_seed(args.seed + 1)
    result["history"].extend(relearn(model, train, test, args, device))
    torch.save(model.state_dict(), os.path.join(args.output, "relearn.pt"))
    with open(os.path.join(args.output, "results.json"), "w") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=(
            "random",
            "basft",
            "tar",
            "tar-full",
            "sam-nll",
            "sam-ent",
            "bad-target",
            "bad-stationary",
            "bad-basin",
            "bad-dead-relu",
            "bad-margin",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint", help="Skip defense and relearn from these weights"
    )
    parser.add_argument(
        "--defend-checkpoint",
        help="Start defense from these weights with a fresh optimizer",
    )
    parser.add_argument("--calibrate-constant-head", action="store_true")
    parser.add_argument("--project-feature-margin", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--relearn-steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--train-samples",
        type=int,
        help="Use this many fixed, unaugmented training images",
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--relearn-learning-rate", type=float, default=0.001)
    parser.add_argument("--attack-steps", type=int, default=16)
    parser.add_argument("--attack-learning-rate", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--feature-weight", type=float, default=10.0)
    parser.add_argument("--feature-margin", type=float, default=0.1)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
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
    if args.project_feature_margin is not None and (
        args.project_feature_margin <= 0 or not args.calibrate_constant_head
    ):
        parser.error(
            "A positive --project-feature-margin requires --calibrate-constant-head"
        )
    if args.checkpoint and args.defend_checkpoint:
        parser.error("Use either --checkpoint or --defend-checkpoint")
    if args.train_samples is not None and not 1 <= args.train_samples <= 50000:
        parser.error("--train-samples must be between 1 and 50000")
    main(args)
