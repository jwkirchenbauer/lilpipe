"""Run CIFAR-10 BaSFT, TAR, or SAM defense followed by ordinary SFT attacks."""

import argparse
import copy
from functools import partial
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn

from torch.func import functional_call
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.checkpoint import checkpoint
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from trainers.tr_gip import finite_difference_cosine, gradient_dot


def cosine_backward(model, images, labels, *, tau, rho, skip_threshold):
    """Assign only the finite-difference group-cosine gradient to parameters.

    Args:
        model: FP32 model, or FP64 for numerical validation.
        images: Input images, shape [B, C, H, W], with even B >= 2.
        labels: Class targets, shape [B].
        tau: Positive stabilization for group-gradient norms.
        rho: Positive parameter-space finite-difference perturbation length.
        skip_threshold: Skip the update below either group-gradient norm.

    Each call randomly partitions the batch into two equally sized groups.
    Cross-entropy defines their gradients but is not a defense objective term.
    The signed cosine objective permits opposing as well as orthogonal gradients.
    """
    if len(labels) < 2 or len(labels) % 2 or len(images) != len(labels):
        raise ValueError("Group cosine requires an even batch of at least two examples")
    order = torch.randperm(len(labels), device=images.device)
    a, b = order.chunk(2)
    evaluations = 0

    def group_loss(indices):
        nonlocal evaluations
        evaluations += 1
        return F.cross_entropy(model(images[indices]), labels[indices])

    result = finite_difference_cosine(
        model,
        lambda: group_loss(a),
        lambda: group_loss(b),
        tau=tau,
        rho=rho,
        skip_threshold=skip_threshold,
    )
    for parameter, gradient in zip(
        (p for p in model.parameters() if p.requires_grad),
        result.regularization,
        strict=True,
    ):
        parameter.grad = gradient
    return {
        "objective": result.cosine.item(),
        "group_cosine": result.cosine.item(),
        "group_gradient_norm_a": result.norm_a.item(),
        "group_gradient_norm_b": result.norm_b.item(),
        "regularization_gradient_norm": gradient_dot(
            result.regularization, result.regularization
        )
        .sqrt()
        .item(),
        "training_loss": ((result.loss_a + result.loss_b) / 2).item(),
        "skipped": result.skipped,
        "forget_gradient_evaluations": evaluations,
    }


def exact_cosine_backward(model, images, labels, *, tau, skip_threshold=0):
    """Assign the exact group-cosine derivative using second-order autograd.

    Args:
        model: FP32 model, or FP64 for numerical validation.
        images: Input images, shape [B, C, H, W], with even B >= 2.
        labels: Class targets, shape [B].
        tau: Positive stabilization for group-gradient norms.
        skip_threshold: Skip the update below either group-gradient norm.

    Randomly split the batch into equal disjoint groups. Cross-entropy defines
    their gradients; only the stabilized signed cosine is differentiated for
    the defense update. No parameter perturbation or explicit Hessian is used.
    """
    if len(labels) < 2 or len(labels) % 2 or len(images) != len(labels):
        raise ValueError("Group cosine requires an even batch of at least two examples")
    order = torch.randperm(len(labels), device=images.device)
    a, b = order.chunk(2)
    if not 0 < tau < float("inf") or not 0 <= skip_threshold < float("inf"):
        raise ValueError("Invalid cosine stabilization or skip threshold")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    losses = [F.cross_entropy(model(images[group]), labels[group]) for group in (a, b)]
    gradients = [
        tuple(
            torch.zeros_like(p) if g is None else g
            for p, g in zip(
                parameters,
                torch.autograd.grad(
                    loss, parameters, create_graph=True, allow_unused=True
                ),
                strict=True,
            )
        )
        for loss in losses
    ]
    ga, gb = gradients
    aa, bb = gradient_dot(ga, ga), gradient_dot(gb, gb)
    if not torch.isfinite(aa) or not torch.isfinite(bb):
        raise FloatingPointError("Group gradient norm is not finite")
    cosine = gradient_dot(ga, gb) / ((aa + tau**2).sqrt() * (bb + tau**2).sqrt())
    skipped = bool((aa.sqrt() < skip_threshold) | (bb.sqrt() < skip_threshold))
    regularization = (
        torch.autograd.grad(cosine, parameters, allow_unused=True)
        if not skipped
        else (None,) * len(parameters)
    )
    for p, g in zip(parameters, regularization, strict=True):
        p.grad = torch.zeros_like(p) if g is None else g.detach()
    return {
        "objective": cosine.item(),
        "group_cosine": cosine.item(),
        "group_gradient_norm_a": aa.sqrt().item(),
        "group_gradient_norm_b": bb.sqrt().item(),
        "regularization_gradient_norm": norm(p.grad for p in parameters).item(),
        "training_loss": ((losses[0] + losses[1]) / 2).item(),
        "skipped": skipped,
        "forget_gradient_evaluations": 2,
        "second_order_backward_calls": int(not skipped),
    }


def individual_cosine_backward(
    model,
    images,
    labels,
    *,
    method,
    tau,
    rho,
    sample_size=8,
    skip_threshold=0,
):
    """Estimate individual-gradient regularizers without storing a B-by-P matrix.

    Args:
        model: Batch-independent evaluation-mode model, FP32 (FP64 for tests).
        images: Inputs, shape [B, C, H, W], even B >= 2.
        labels: Class targets, shape [B].
        method: frozen-normalization, sampled-individual, or full-individual.
        tau: Positive stabilization for individual gradient norms.
        rho: Positive parameter-space perturbation length.
        sample_size: Number of sampled contributions; ignored by the full method.
        skip_threshold: Omit directions for gradients below this norm.

    Frozen normalization targets a cross-group inner product with detached
    inverse norms. The individual methods target the stabilized off-diagonal
    pairwise cosine, including derivatives of its diagonal correction.
    Functional parameter shifts and cloned buffers leave the model unchanged.
    """
    n = len(labels)
    if n < 2 or n % 2 or len(images) != n:
        raise ValueError("Individual regularization requires an even batch")
    if method not in {"frozen-normalization", "sampled-individual", "full-individual"}:
        raise ValueError(f"Unknown individual method: {method}")
    if (
        not 0 < tau < float("inf")
        or not 0 < rho < float("inf")
        or not 0 <= skip_threshold < float("inf")
    ):
        raise ValueError("Invalid individual regularizer stabilization or radius")
    if model.training:
        raise ValueError("Individual CIFAR methods require evaluation mode")
    if method == "sampled-individual" and not 1 <= sample_size <= n:
        raise ValueError("Sample size must be between one and batch size")
    named = {k: p for k, p in model.named_parameters() if p.requires_grad}
    parameters = tuple(named.values())
    if not parameters or parameters[0].dtype not in (torch.float32, torch.float64):
        raise ValueError("Individual regularizers require FP32, or FP64 for tests")
    buffers = {k: b.detach().clone() for k, b in model.named_buffers()}
    cuda_devices = sorted({p.device.index for p in parameters if p.is_cuda})
    evaluations = 0

    def evaluate(indices, *, weights=None, point=None, replay=None):
        nonlocal evaluations
        point = named if point is None else point
        rng = (
            torch.get_rng_state(),
            [torch.cuda.get_rng_state(d) for d in cuda_devices],
        )
        with torch.random.fork_rng(devices=cuda_devices, enabled=replay is not None):
            if replay is not None:
                torch.set_rng_state(replay[0])
                for device, state in zip(cuda_devices, replay[1], strict=True):
                    torch.cuda.set_rng_state(state, device)
            logits = functional_call(
                model,
                (point, {k: b.clone() for k, b in buffers.items()}),
                (images[indices],),
            )
            losses = F.cross_entropy(logits, labels[indices], reduction="none")
            loss = (losses if weights is None else losses * weights).mean()
            gs = torch.autograd.grad(loss, tuple(point.values()), allow_unused=True)
        evaluations += 1
        gs = tuple(
            torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(parameters, gs, strict=True)
        )
        return loss.detach(), gs, rng

    normalized_mean = [torch.zeros_like(p) for p in parameters]
    inverse_norms = images.new_empty(n)
    raw_norms = images.new_empty(n)
    base_rngs = []
    diagonal_mean = images.new_zeros(())
    loss_mean = images.new_zeros(())
    norm_started = time.perf_counter()
    for i in range(n):
        loss, gs, rng = evaluate([i])
        squared = gradient_dot(gs, gs)
        if not torch.isfinite(squared):
            raise FloatingPointError("Nonfinite individual gradient norm")
        inverse = (squared + tau**2).rsqrt()
        inverse_norms[i], raw_norms[i] = inverse, squared.sqrt()
        for total, g in zip(normalized_mean, gs, strict=True):
            total.add_(g * (inverse / n))
        diagonal_mean += squared * inverse.square() / n
        loss_mean += loss / n
        base_rngs.append(rng)
    norm_seconds = time.perf_counter() - norm_started
    pairwise = (n * gradient_dot(normalized_mean, normalized_mean) - diagonal_mean) / (
        n - 1
    )
    regularization = [torch.zeros_like(p) for p in parameters]
    skipped = 0

    def accumulate(base, direction, indices, factor, replay, weights=None):
        length = gradient_dot(direction, direction).sqrt()
        if not torch.isfinite(length):
            raise FloatingPointError("Nonfinite individual regularization direction")
        if length == 0:
            return
        point = {
            name: (p.detach() + rho * v / length).requires_grad_()
            for (name, p), v in zip(named.items(), direction, strict=True)
        }
        _, plus, _ = evaluate(indices, weights=weights, point=point, replay=replay)
        for r, gp, g in zip(regularization, plus, base, strict=True):
            r.add_((factor * length / rho) * (gp - g))

    if method == "frozen-normalization":
        a, b = torch.randperm(n, device=images.device).chunk(2)
        _, ha, rng_a = evaluate(a, weights=inverse_norms[a])
        _, hb, rng_b = evaluate(b, weights=inverse_norms[b])
        objective = gradient_dot(ha, hb)
        if bool((raw_norms < skip_threshold).any()):
            skipped = n
        else:
            accumulate(ha, hb, a, 1, rng_a, inverse_norms[a])
            accumulate(hb, ha, b, 1, rng_b, inverse_norms[b])
        contributions = 2
    else:
        indices = (
            list(range(n))
            if method == "full-individual"
            else torch.randperm(n)[:sample_size].tolist()
        )
        contributions = len(indices)
        objective = pairwise
        for i in indices:
            if raw_norms[i] < skip_threshold:
                skipped += 1
                continue
            _, gs, _ = evaluate([i], replay=base_rngs[i])
            inverse = inverse_norms[i]
            # d off-diagonal mean / d g_i includes the stabilized diagonal.
            u = tuple(g * inverse for g in gs)
            w = tuple(m - ui / n for m, ui in zip(normalized_mean, u, strict=True))
            projection = gradient_dot(u, w)
            direction = tuple(
                (wi - ui * projection) * inverse for wi, ui in zip(w, u, strict=True)
            )
            accumulate(
                gs, direction, [i], 2 * n / (contributions * (n - 1)), base_rngs[i]
            )
    for p, r in zip(parameters, regularization, strict=True):
        p.grad = r
    return {
        "objective": objective.item(),
        "individual_pairwise_cosine": pairwise.item(),
        "example_gradient_norm_mean": raw_norms.mean().item(),
        "example_gradient_norm_min": raw_norms.min().item(),
        "example_gradient_norm_max": raw_norms.max().item(),
        "regularization_gradient_norm": norm(regularization).item(),
        "training_loss": loss_mean.item(),
        "skipped": skipped
        == (n if method == "frozen-normalization" else contributions),
        "skipped_contributions": skipped,
        "sampled_contributions": contributions,
        "forget_gradient_evaluations": evaluations,
        "per_example_norm_seconds": norm_seconds,
    }


def positive_pair_cosine_backward(
    model, images, labels, *, tau, pair_count=32, skip_threshold=0, pairs=None
):
    """Differentiate the positive part of individual pairwise cosine.

    Args:
        model: Evaluation-mode, batch-independent classification model.
        images: Input images, shape [B, C, H, W], B >= 2.
        labels: Class targets, shape [B].
        tau: Positive gradient-norm stabilization.
        pair_count: Number of uniformly sampled distinct-example pairs.
        skip_threshold: Omit pairs below either raw gradient-norm threshold.
        pairs: Optional explicit indices, shape [K, 2], for exact validation.

    The target is mean(relu(cos(g_i, g_j))) over i != j. Opposing pairs
    contribute zero. Each sampled contribution uses exact second-order autograd;
    pairs are processed sequentially without retaining all per-example graphs.
    """
    n = len(labels)
    if n < 2 or len(images) != n or pair_count < 1:
        raise ValueError(
            "Positive-pair cosine requires examples and positive pair count"
        )
    if not 0 < tau < float("inf") or not 0 <= skip_threshold < float("inf"):
        raise ValueError("Invalid cosine stabilization or skip threshold")
    if model.training:
        raise ValueError("Positive-pair CIFAR regularization requires evaluation mode")
    if pairs is None:
        first = torch.randint(n, (pair_count,), device=images.device)
        second = torch.randint(n - 1, (pair_count,), device=images.device)
        second += second >= first
        pairs = torch.stack((first, second), dim=1)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs):
        raise ValueError("Pairs must have shape [K, 2]")
    if bool(
        (pairs < 0).any() | (pairs >= n).any() | (pairs[:, 0] == pairs[:, 1]).any()
    ):
        raise ValueError("Pairs must contain distinct valid example indices")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    accumulated = [torch.zeros_like(p) for p in parameters]
    objective = signed_cosine = training_loss = 0.0
    active = 0
    norms = []
    for i, j in pairs.tolist():
        losses = [
            F.cross_entropy(model(images[index : index + 1]), labels[index : index + 1])
            for index in (i, j)
        ]
        gradients = [
            tuple(
                torch.zeros_like(p) if g is None else g
                for p, g in zip(
                    parameters,
                    torch.autograd.grad(
                        loss, parameters, create_graph=True, allow_unused=True
                    ),
                    strict=True,
                )
            )
            for loss in losses
        ]
        a, b = gradients
        aa, bb = gradient_dot(a, a), gradient_dot(b, b)
        if not torch.isfinite(aa) or not torch.isfinite(bb):
            raise FloatingPointError("Nonfinite pair gradient norm")
        cosine = gradient_dot(a, b) / ((aa + tau**2).sqrt() * (bb + tau**2).sqrt())
        penalty = cosine.relu()
        objective += penalty.item() / len(pairs)
        signed_cosine += cosine.item() / len(pairs)
        training_loss += (losses[0].item() + losses[1].item()) / (2 * len(pairs))
        norms.extend((aa.detach().sqrt(), bb.detach().sqrt()))
        if cosine <= 0 or aa.sqrt() < skip_threshold or bb.sqrt() < skip_threshold:
            continue
        gs = torch.autograd.grad(penalty, parameters, allow_unused=True)
        for total, g in zip(accumulated, gs, strict=True):
            if g is not None:
                total.add_(g.detach() / len(pairs))
        active += 1
    for p, g in zip(parameters, accumulated, strict=True):
        p.grad = g
    norms = torch.stack(norms)
    return {
        "objective": objective,
        "sampled_pair_cosine_mean": signed_cosine,
        "sampled_pairs": len(pairs),
        "positive_pairs": active,
        "training_loss": training_loss,
        "example_gradient_norm_mean": norms.mean().item(),
        "regularization_gradient_norm": norm(accumulated).item(),
        "forget_gradient_evaluations": 2 * len(pairs),
        "second_order_backward_calls": active,
        "skipped": active == 0,
    }


def norm(values):
    return sum(value.square().sum() for value in values).sqrt()


def cosine_diagnostics(model, images, labels, *, tau):
    """Measure off-diagonal example cosine and group cosine with O(P) memory.

    Args:
        model: Evaluation-mode, batch-independent model with P trainable scalars.
        images: Fixed diagnostic inputs, shape [B, C, H, W], even B >= 2.
        labels: Targets, shape [B].
        tau: Positive stabilization, sqrt(squared gradient norm + tau squared).

    Gradients are materialized one example at a time. The off-diagonal mean is
    (||sum_i u_i|| squared - sum_i ||u_i|| squared) / (B * (B - 1)), including
    the actual stabilized diagonal. Parameter .grad fields are not modified.
    """
    n = len(labels)
    if n < 2 or n % 2 or len(images) != n:
        raise ValueError("Cosine diagnostics require an even batch of at least two")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    normalized_sum = [torch.zeros_like(p) for p in parameters]
    group_sums = [[torch.zeros_like(p) for p in parameters] for _ in range(2)]
    diagonal = images.new_zeros(())
    norms = []
    for i in range(n):
        loss = F.cross_entropy(model(images[i : i + 1]), labels[i : i + 1])
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        gradients = tuple(
            torch.zeros_like(p) if g is None else g
            for p, g in zip(parameters, gradients, strict=True)
        )
        squared = gradient_dot(gradients, gradients)
        stabilized = (squared + tau**2).sqrt()
        norms.append(squared.sqrt())
        diagonal += squared / (squared + tau**2)
        for total, group, g in zip(
            normalized_sum, group_sums[int(i >= n // 2)], gradients, strict=True
        ):
            total.add_(g / stabilized)
            group.add_(g / (n // 2))
    ga, gb = group_sums
    group_cosine = gradient_dot(ga, gb) / (
        (gradient_dot(ga, ga) + tau**2).sqrt() * (gradient_dot(gb, gb) + tau**2).sqrt()
    )
    values = torch.stack(norms)
    return {
        "individual_pairwise_cosine": (
            (gradient_dot(normalized_sum, normalized_sum) - diagonal) / (n * (n - 1))
        ).item(),
        "diagnostic_group_cosine": group_cosine.item(),
        "example_gradient_norm_mean": values.mean().item(),
        "example_gradient_norm_min": values.min().item(),
        "example_gradient_norm_max": values.max().item(),
        "diagnostic_samples": n,
    }


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
    microbatch_size = min(
        args.batch_size,
        getattr(args, "relearn_microbatch_size", None) or args.batch_size,
    )
    diagnostic_count = getattr(args, "cosine_diagnostic_samples", 0)
    if diagnostic_count:
        if diagnostic_count > len(train):
            raise ValueError("Diagnostic sample count exceeds training set")
        # Keep the same cohort and preserve the attack's RNG stream.
        with torch.random.fork_rng(devices=[]):
            diagnostic_images = torch.stack(
                [train[i][0] for i in range(diagnostic_count)]
            ).to(device)
            diagnostic_labels = torch.tensor(
                [train[i][1] for i in range(diagnostic_count)], device=device
            )
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
            if diagnostic_count and (
                step % args.cosine_diagnostic_every == 0 or step == args.relearn_steps
            ):
                start = time.perf_counter()
                row.update(
                    cosine_diagnostics(
                        model, diagnostic_images, diagnostic_labels, tau=args.cosine_tau
                    )
                )
                row["diagnostic_seconds"] = time.perf_counter() - start
            if device.type == "cuda":
                row["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            history.append(row)
            print(json.dumps(row), flush=True)
        if step < args.relearn_steps:
            optimizer.step()
    return history


def main(args):
    if getattr(args, "disable_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.method in [
        "gradient-cosine",
        "gradient-cosine-exact",
        "frozen-normalization",
        "sampled-individual",
        "full-individual",
    ] and (
        args.batch_size % 2
        or args.batch_size < 2
        or (args.train_samples and args.train_samples % args.batch_size)
    ):
        raise ValueError("Group cosine requires complete, even-sized batches")
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
        skipped_steps = 0
        for step in range(args.steps):
            step_started = time.perf_counter()

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

            cosine_metrics = None
            if args.method in [
                "gradient-cosine",
                "gradient-cosine-exact",
                "frozen-normalization",
                "sampled-individual",
                "full-individual",
                "positive-pair-cosine",
            ]:
                images, labels = next(attack_stream)
                optimizer.zero_grad(set_to_none=True)
                if args.method == "gradient-cosine-exact":
                    cosine_metrics = exact_cosine_backward(
                        model,
                        images,
                        labels,
                        tau=args.cosine_tau,
                        skip_threshold=args.cosine_skip_threshold,
                    )
                elif args.method == "gradient-cosine":
                    cosine_metrics = cosine_backward(
                        model,
                        images,
                        labels,
                        tau=args.cosine_tau,
                        rho=args.rho,
                        skip_threshold=args.cosine_skip_threshold,
                    )
                elif args.method == "positive-pair-cosine":
                    cosine_metrics = positive_pair_cosine_backward(
                        model,
                        images,
                        labels,
                        tau=args.cosine_tau,
                        pair_count=args.cosine_contributions,
                        skip_threshold=args.cosine_skip_threshold,
                    )
                else:
                    cosine_metrics = individual_cosine_backward(
                        model,
                        images,
                        labels,
                        method=args.method,
                        tau=args.cosine_tau,
                        rho=args.rho,
                        sample_size=args.cosine_contributions,
                        skip_threshold=args.cosine_skip_threshold,
                    )
                skipped_steps += cosine_metrics["skipped"]
                loss = images.new_tensor(cosine_metrics["objective"])
            elif args.method in ["tar", "tar-full"]:
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
            if cosine_metrics is None:
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
                if cosine_metrics is not None:
                    row["gradient_seconds"] = time.perf_counter() - step_started
                    row.update(cosine_metrics)
                    row["skipped_step_frequency"] = skipped_steps / (step + 1)
                if device.type == "cuda":
                    row["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
                    row["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                    row["peak_gpu_reserved_gib"] = (
                        torch.cuda.max_memory_reserved() / 1024**3
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
            # An Adam step with zero gradients can still move via momentum.
            if cosine_metrics is None or not cosine_metrics["skipped"]:
                if not all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise ValueError("Nonfinite defense gradient")
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
            "gradient-cosine",
            "gradient-cosine-exact",
            "frozen-normalization",
            "sampled-individual",
            "full-individual",
            "positive-pair-cosine",
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
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--relearn-steps", type=int, default=600)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--relearn-microbatch-size", type=int)
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
    parser.add_argument("--cosine-tau", type=float, default=1e-3)
    parser.add_argument("--cosine-contributions", type=int, default=8)
    parser.add_argument("--cosine-skip-threshold", type=float, default=1e-8)
    parser.add_argument("--cosine-diagnostic-samples", type=int, default=0)
    parser.add_argument("--cosine-diagnostic-every", type=int, default=10)
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
    if args.relearn_microbatch_size is not None and args.relearn_microbatch_size < 1:
        parser.error("Relearning microbatch size must be positive")
    if args.cosine_contributions < 1:
        parser.error("Cosine contribution count must be positive")
    if (
        args.cosine_diagnostic_samples < 0
        or args.cosine_diagnostic_samples % 2
        or args.cosine_diagnostic_every < 1
    ):
        parser.error(
            "Diagnostic sample count must be even and nonnegative, interval positive"
        )
    if not (
        0 < args.cosine_tau < float("inf")
        and 0 <= args.cosine_skip_threshold < float("inf")
    ):
        parser.error("Cosine tau must be positive and skip threshold nonnegative")
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
