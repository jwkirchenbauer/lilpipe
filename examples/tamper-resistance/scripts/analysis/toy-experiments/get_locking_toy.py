"""Reproduce the penalty-based locking toy of DLR-Lock (Sakamoto et al., 2026), Fig. 2.

A two-layer ReLU MLP is pretrained on MNIST, then fine-tuned from theta_0 with

    min_delta  MSE(f_theta0, f_theta0+delta) - lambda * Omega(theta0 + delta)

using SGD with gradient-norm clipping, for Omega = ||delta||^2 (distance) and
Omega = Tr(H_CE) (sharpness). Function-preserving weight-symmetry solutions
(W1 / a, W2 * a) are recorded as the reference points the trajectories miss.

With ``--canonical``, Omega is measured on the rebalanced network canon(theta),
so weight rescaling no longer raises it. ``--optimizer projected`` replaces the
penalty with constrained ascent: maximize Omega while the relative MSE on a
probe set stays below a tolerance epsilon.
"""

import argparse
import json
import os
import os.path as osp

import numpy as np
import torch
from torch.nn import functional as F

PENALTIES = ("distance", "hessian")


def load_mnist():
    import datasets

    data = datasets.load_dataset("ylecun/mnist").with_format("numpy")

    def split(name):
        images = torch.from_numpy(data[name][:]["image"]).float().flatten(1) / 255.0
        return (images - 0.1307) / 0.3081, torch.from_numpy(data[name][:]["label"])

    return split("train"), split("test")


def init_params(input_size, hidden_size, classes, generator):
    def linear(fan_in, fan_out):
        bound = fan_in**-0.5
        weight = (torch.rand(fan_out, fan_in, generator=generator) * 2 - 1) * bound
        bias = (torch.rand(fan_out, generator=generator) * 2 - 1) * bound
        return [weight, bias]

    return linear(input_size, hidden_size) + linear(hidden_size, classes)


def forward(params, inputs):
    w1, b1, w2, b2 = params
    return F.relu(inputs @ w1.T + b1) @ w2.T + b2


def hessian_trace(params, inputs):
    """Exact per-sample Tr(d^2 CE / d theta^2) for the two-layer ReLU MLP.

    Logits are linear in each layer's parameters inside a ReLU region, so the
    diagonal of the Hessian is the Gauss-Newton one: Tr(S K), with
    S = diag(p) - p p^T and the per-sample NTK
    K = (||h||^2 + 1) I + (||x||^2 + 1) W2 diag(m) W2^T. The trace does not
    depend on the label.
    """
    w1, b1, w2, b2 = params
    preactivation = inputs @ w1.T + b1
    hidden = F.relu(preactivation)
    mask = (preactivation > 0).to(inputs.dtype)
    probs = torch.softmax(hidden @ w2.T + b2, dim=-1)
    softmax_trace = 1 - (probs * probs).sum(-1)
    # w2[:, j]^T S w2[:, j] for every hidden unit j.
    unit_curvature = probs @ (w2 * w2) - (probs @ w2) ** 2
    first_layer = (unit_curvature * mask).sum(-1)
    return (hidden.pow(2).sum(-1) + 1) * softmax_trace + (
        inputs.pow(2).sum(-1) + 1
    ) * first_layer


def canonicalize(params, eps=1e-12):
    """Rebalance each hidden unit so its incoming and outgoing norms match.

    ReLU is positively homogeneous, so scaling unit j's incoming weights and
    bias by a_j and its outgoing weights by 1 / a_j keeps the function; the
    balanced representative removes the (W1 / a, a W2) symmetry.
    """
    w1, b1, w2, b2 = params
    incoming = torch.sqrt(w1.pow(2).sum(1) + b1.pow(2) + eps)
    outgoing = torch.sqrt(w2.pow(2).sum(0) + eps)
    scale = torch.sqrt(outgoing / incoming)
    return [w1 * scale[:, None], b1 * scale, w2 / scale, b2]


def flat_norm(tensors):
    return torch.sqrt(sum((tensor * tensor).sum() for tensor in tensors))


def pretrain(params, inputs, labels, epochs, batch_size, learning_rate, generator):
    params = [param.clone().requires_grad_() for param in params]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    for _ in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(inputs), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            F.cross_entropy(forward(params, inputs[batch]), labels[batch]).backward()
            optimizer.step()
    return [param.detach() for param in params]


class Reference:
    """Quantities of theta_0 on the fixed evaluation inputs."""

    def __init__(self, params, inputs, labels):
        self.params = params
        self.inputs = inputs
        self.labels = labels
        self.canonical_params = canonicalize(params)
        with torch.no_grad():
            self.logits = forward(params, inputs)
            self.trace = hessian_trace(params, inputs)
            self.norm = flat_norm(params).item()
            self.canonical_trace = hessian_trace(self.canonical_params, inputs)
            self.canonical_norm = flat_norm(self.canonical_params).item()
            self.accuracy = accuracy(params, inputs, labels)

    def measure(self, params):
        """Relative MSE, relative distance and relative Hessian trace (Fig. 2).

        ``relative_mse`` is the paper's geometric mean over examples;
        ``pooled_relative_mse`` is the ratio of sums the projected constraint uses.
        """
        with torch.no_grad():
            logits = forward(params, self.inputs)
            error = (logits - self.logits).pow(2).sum(-1)
            relative_mse = torch.log(
                error.clamp_min(1e-30) / self.logits.pow(2).sum(-1)
            )
            # A saturated softmax can underflow the trace to exactly zero.
            trace = hessian_trace(params, self.inputs).clamp_min(1e-30)
            relative_trace = torch.log(trace / self.trace)
            distance = flat_norm([p - p0 for p, p0 in zip(params, self.params)])
            canonical = canonicalize(params)
            canonical_trace = hessian_trace(canonical, self.inputs).clamp_min(1e-30)
            canonical_relative_trace = torch.log(canonical_trace / self.canonical_trace)
            canonical_distance = flat_norm(
                [p - p0 for p, p0 in zip(canonical, self.canonical_params)]
            )
        return {
            "relative_mse": torch.exp(relative_mse.mean()).item(),
            "pooled_relative_mse": (error.sum() / self.logits.pow(2).sum()).item(),
            "accuracy": (logits.argmax(-1) == self.labels).float().mean().item(),
            "relative_distance": distance.item() / self.norm,
            "relative_hessian_trace": torch.exp(relative_trace.mean()).item(),
            "canonical_relative_distance": canonical_distance.item()
            / self.canonical_norm,
            "canonical_relative_hessian_trace": torch.exp(
                canonical_relative_trace.mean()
            ).item(),
        }


def accuracy(params, inputs, labels):
    with torch.no_grad():
        return (forward(params, inputs).argmax(-1) == labels).float().mean().item()


def omega(params, reference, inputs, penalty, canonical):
    """||theta - theta_0||^2 or mean Tr(H_CE), optionally on canon(theta)."""
    anchor = reference.params
    if canonical:
        params, anchor = canonicalize(params), reference.canonical_params
    if penalty == "distance":
        return sum((p - p0).pow(2).sum() for p, p0 in zip(params, anchor))
    return hessian_trace(params, inputs).mean()


def objective(params, reference, inputs, penalty, coefficient, canonical=False):
    """MSE(f_theta0, f_theta) - lambda * Omega(theta) on one minibatch."""
    with torch.no_grad():
        target = forward(reference.params, inputs)
    mse = F.mse_loss(forward(params, inputs), target)
    return mse - coefficient * omega(params, reference, inputs, penalty, canonical)


def zeroth_order_gradient(params, loss_fn, directions, smoothing, generator):
    """Two-point SPSA estimate (MeZO), averaged over Gaussian directions.

    Uses only forward evaluations of ``loss_fn``; in expectation it is the
    gradient of the objective smoothed by a Gaussian of std ``smoothing``.
    """
    estimate = [torch.zeros_like(p) for p in params]
    with torch.no_grad():
        for _ in range(directions):
            noise = [torch.randn(p.shape, generator=generator) for p in params]
            for p, z in zip(params, noise):
                p.add_(z, alpha=smoothing)
            plus = loss_fn()
            for p, z in zip(params, noise):
                p.add_(z, alpha=-2 * smoothing)
            minus = loss_fn()
            for p, z in zip(params, noise):
                p.add_(z, alpha=smoothing)
            scale = (plus - minus) / (2 * smoothing * directions)
            for g, z in zip(estimate, noise):
                g.add_(z, alpha=scale.item())
    return estimate


def lock(reference, inputs, penalty, coefficient, learning_rate, args, seed):
    """SGD on MSE - lambda * Omega from a tiny perturbation of theta_0.

    Both objectives have zero gradient at theta_0 for the distance penalty, so
    every run starts from theta_0 plus Gaussian noise of std ``args.init_noise``.
    With ``args.optimizer == "zo"`` the gradient is replaced by a zeroth-order
    estimate; clipping and the SGD step are unchanged.
    """
    generator = torch.Generator().manual_seed(seed)
    params = [
        (p + args.init_noise * torch.randn(p.shape, generator=generator))
        .clone()
        .requires_grad_()
        for p in reference.params
    ]
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    trajectory = [{"step": 0, **reference.measure(params)}]
    status = "completed"
    for step in range(1, args.lock_steps + 1):
        batch = torch.randint(len(inputs), (args.batch_size,), generator=generator)
        batch_inputs = inputs[batch]

        def loss_fn():
            return objective(
                params, reference, batch_inputs, penalty, coefficient, args.canonical
            )

        optimizer.zero_grad()
        if args.optimizer == "zo":
            estimate = zeroth_order_gradient(
                params, loss_fn, args.zo_directions, args.zo_smoothing, generator
            )
            for p, g in zip(params, estimate):
                p.grad = g
        else:
            loss_fn().backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            status = "diverged"
            break
        optimizer.step()
        if step % args.log_every == 0 or step == args.lock_steps:
            point = {"step": step, **reference.measure(params)}
            if not all(np.isfinite(value) for value in point.values()):
                status = "diverged"
                break
            trajectory.append(point)
    return {
        "penalty": penalty,
        "lambda": coefficient,
        "learning_rate": learning_rate,
        "status": status,
        "trajectory": trajectory,
    }


def jacobian_products(params, inputs):
    """J v and J^T u for the logits of the two-layer MLP, written out by hand."""
    w1, b1, w2, _ = params
    preactivation = inputs @ w1.T + b1
    hidden = F.relu(preactivation)
    mask = (preactivation > 0).to(inputs.dtype)

    def jvp(vector):
        v1, vb1, v2, vb2 = vector
        return (mask * (inputs @ v1.T + vb1)) @ w2.T + hidden @ v2.T + vb2

    def vjp(cotangent):
        delta = (cotangent @ w2) * mask
        return [delta.T @ inputs, delta.sum(0), cotangent.T @ hidden, cotangent.sum(0)]

    return jvp, vjp


def gauss_newton_solve(params, inputs, rhs, damping, iterations):
    """Conjugate gradient for (J^T J / N + damping I) u = rhs.

    J is the Jacobian of the logits on ``inputs`` with respect to ``params``,
    so J^T J / N is the Gauss-Newton matrix of the MSE: its null space holds
    the directions that leave every probe output unchanged.
    """
    jvp, vjp = jacobian_products(params, inputs)

    def matvec(vector):
        return [g / len(inputs) + damping * v for g, v in zip(vjp(jvp(vector)), vector)]

    def dot(first, second):
        return sum((a * b).sum() for a, b in zip(first, second))

    solution = [torch.zeros_like(r) for r in rhs]
    residual = [r.clone() for r in rhs]
    direction = [r.clone() for r in rhs]
    residual_norm = dot(residual, residual)
    for _ in range(iterations):
        product = matvec(direction)
        alpha = residual_norm / dot(direction, product)
        solution = [s + alpha * d for s, d in zip(solution, direction)]
        residual = [r - alpha * p for r, p in zip(residual, product)]
        new_norm = dot(residual, residual)
        if new_norm.sqrt() < 1e-8 * dot(rhs, rhs).sqrt():
            break
        direction = [
            r + (new_norm / residual_norm) * d for r, d in zip(residual, direction)
        ]
        residual_norm = new_norm
    return solution


def lock_projected(reference, probe, penalty, tolerance, args, seed):
    """Maximize Omega subject to relative MSE on ``probe`` <= ``tolerance``.

    Each step moves a fixed parameter distance along (G + damping I)^-1 grad
    Omega, the ascent direction that changes the probe outputs least, then
    retracts toward f_theta0 with Gauss-Newton steps on the MSE. A step is
    accepted only if the probe constraint holds and Omega increased; otherwise
    the step size is halved. The run stalls when no step size is accepted.
    """
    generator = torch.Generator().manual_seed(seed)
    params = [
        p + args.init_noise * torch.randn(p.shape, generator=generator)
        for p in reference.params
    ]
    target = forward(reference.params, probe)
    target_norm = target.pow(2).sum()

    def probe_mse(tensors):
        with torch.no_grad():
            return (
                (forward(tensors, probe) - target).pow(2).sum() / target_norm
            ).item()

    def value(tensors):
        with torch.no_grad():
            return omega(tensors, reference, probe, penalty, args.canonical).item()

    def solve(tensors, rhs):
        return gauss_newton_solve(
            tensors, probe, rhs, args.gn_damping, args.cg_iterations
        )

    def retract(tensors):
        for _ in range(args.retraction_steps):
            residual = forward(tensors, probe) - target
            _, vjp = jacobian_products(tensors, probe)
            correction = solve(tensors, [g / len(probe) for g in vjp(residual)])
            tensors = [t - c for t, c in zip(tensors, correction)]
        return tensors

    step_size = args.projected_step
    trajectory = [{"step": 0, **reference.measure(params)}]
    status = "completed"
    for step in range(1, args.projected_steps + 1):
        current = value(params)
        leaves = [p.clone().requires_grad_() for p in params]
        gradient = torch.autograd.grad(
            omega(leaves, reference, probe, penalty, args.canonical), leaves
        )
        ascent = solve(params, list(gradient))
        ascent_norm = flat_norm(ascent)
        for _ in range(args.max_backtracks):
            candidate = retract(
                [p + step_size * a / ascent_norm for p, a in zip(params, ascent)]
            )
            if probe_mse(candidate) <= tolerance and value(candidate) > current:
                params = candidate
                step_size *= 1.5
                break
            step_size /= 2
        else:
            status = "stalled"
            trajectory.append({"step": step, **reference.measure(params)})
            break
        if step % args.projected_log_every == 0 or step == args.projected_steps:
            point = {"step": step, **reference.measure(params)}
            if not all(np.isfinite(v) for v in point.values()):
                status = "diverged"
                break
            trajectory.append(point)
    return {
        "penalty": penalty,
        "tolerance": tolerance,
        "status": status,
        "final_probe_mse": probe_mse(params),
        "trajectory": trajectory,
    }


def symmetry_solutions(reference, scales):
    """theta_a = (W1 / a, b1 / a, a W2, b2): the same function for any a > 0."""
    w1, b1, w2, b2 = reference.params
    return [
        {"scale": scale, **reference.measure([w1 / scale, b1 / scale, w2 * scale, b2])}
        for scale in scales
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/analysis/locking-toy.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--eval-examples", type=int, default=1000)
    parser.add_argument("--penalties", nargs="+", choices=PENALTIES, default=PENALTIES)
    # 29 log-spaced penalty coefficients and three learning rates, as in Fig. 2.
    parser.add_argument("--log10-lambda-min", type=float, default=-4.0)
    parser.add_argument("--log10-lambda-max", type=float, default=3.0)
    parser.add_argument("--num-lambdas", type=int, default=29)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1]
    )
    parser.add_argument(
        "--optimizer", choices=("sgd", "zo", "projected"), default="sgd"
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Measure Omega on the rebalanced network canon(theta).",
    )
    # Constrained ascent (--optimizer projected).
    parser.add_argument(
        "--tolerances", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2]
    )
    parser.add_argument("--probe-examples", type=int, default=8192)
    parser.add_argument("--projected-steps", type=int, default=100)
    parser.add_argument("--projected-step", type=float, default=0.1)
    parser.add_argument("--projected-log-every", type=int, default=5)
    parser.add_argument("--max-backtracks", type=int, default=12)
    parser.add_argument("--retraction-steps", type=int, default=2)
    parser.add_argument("--gn-damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=30)
    parser.add_argument("--zo-directions", type=int, default=1)
    parser.add_argument("--zo-smoothing", type=float, default=1e-3)
    parser.add_argument("--lock-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--init-noise", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--symmetry-scales", type=float, nargs="+", default=[2, 4, 8, 16, 32, 64]
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    (train_x, train_y), (test_x, test_y) = load_mnist()
    params = init_params(train_x.shape[1], args.hidden_size, 10, generator)
    params = pretrain(
        params,
        train_x,
        train_y,
        epochs=args.pretrain_epochs,
        batch_size=128,
        learning_rate=1e-3,
        generator=generator,
    )
    reference = Reference(
        params, test_x[: args.eval_examples], test_y[: args.eval_examples]
    )
    print(f"theta_0 test accuracy {reference.accuracy:.4f}")

    lambdas = np.logspace(
        args.log10_lambda_min, args.log10_lambda_max, args.num_lambdas
    ).tolist()
    runs = []
    for penalty in args.penalties:
        if args.optimizer == "projected":
            for index, tolerance in enumerate(args.tolerances):
                run = lock_projected(
                    reference,
                    train_x[: args.probe_examples],
                    penalty,
                    tolerance,
                    args,
                    seed=args.seed + index,
                )
                end = run["trajectory"][-1]
                print(
                    f"{penalty} projected eps={tolerance:.0e} {run['status']} "
                    f"after {end['step']} steps: probe mse {run['final_probe_mse']:.2e} "
                    f"mse {end['relative_mse']:.2e} "
                    f"canon dist {end['canonical_relative_distance']:.2e} "
                    f"canon trace {end['canonical_relative_hessian_trace']:.2e}"
                )
                runs.append(run)
            continue
        for learning_rate in args.learning_rates:
            for index, coefficient in enumerate(lambdas):
                run = lock(
                    reference,
                    train_x,
                    penalty,
                    coefficient,
                    learning_rate,
                    args,
                    seed=args.seed + index,
                )
                end = run["trajectory"][-1]
                print(
                    f"{penalty} lr={learning_rate:g} lambda={coefficient:.2e} "
                    f"{run['status']}: mse {end['relative_mse']:.2e} "
                    f"dist {end['relative_distance']:.2e} "
                    f"trace {end['relative_hessian_trace']:.2e} "
                    f"canon dist {end['canonical_relative_distance']:.2e} "
                    f"canon trace {end['canonical_relative_hessian_trace']:.2e}"
                )
                runs.append(run)

    results = {
        "config": vars(args),
        "base_accuracy": reference.accuracy,
        "lambdas": lambdas,
        "symmetry_solutions": symmetry_solutions(reference, args.symmetry_scales),
        "runs": runs,
    }
    os.makedirs(osp.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
