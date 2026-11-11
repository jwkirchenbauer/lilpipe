"""Short functional attacks for validating exact and first-order meta-gradients.

The loss callback has signature ``loss_fn(forward, batch)``. ``forward`` calls
the supplied model with the current functional parameters and private buffers.
No model parameters, existing ``.grad`` fields, or input optimizer state change.
Use deterministic forwards for finite differences (for example, ``model.eval()``).
The caller controls model mode and data placement, including model sharding.

First-order means an identity Jacobian through the *complete* optimizer update:
``p_next = p + (optimizer_update(p) - p).detach()``. Thus optimizer values match
the exact attack, while curvature, momentum, adaptive scaling, clipping, and
weight-decay derivatives are omitted. This convention is explicit because just
detaching the loss gradient does not give identity when weight decay is present.

The tiny FP64 CLI is intended as a launch_tuo.py payload for GPU checks. It has
no model-path, dataset, trainer, or cluster dependencies beyond PyTorch.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

TensorMap = Mapping[str, Tensor]
State = Mapping[str, Mapping[str, Tensor | int]]
LossFn = Callable[[Callable[..., Any], Any], Tensor]


@dataclass(frozen=True)
class OptimizerConfig:
    """One parameter group matching torch.optim.SGD or torch.optim.AdamW."""

    name: str = "sgd"
    lr: float = 0.05
    weight_decay: float = 0.0
    momentum: float = 0.0
    dampening: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.name not in {"sgd", "adamw"}:
            raise ValueError("name must be 'sgd' or 'adamw'")
        for name in ("lr", "weight_decay", "momentum", "dampening", "eps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.betas) != 2 or any(
            not math.isfinite(beta) or not 0 <= beta < 1 for beta in self.betas
        ):
            raise ValueError("betas must contain two values in [0, 1)")
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0):
            raise ValueError("Nesterov requires positive momentum and zero dampening")
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0
        ):
            raise ValueError("max_grad_norm must be finite and nonnegative")


@dataclass(frozen=True)
class UnrollResult:
    params: dict[str, Tensor]
    buffers: dict[str, Tensor]
    state: dict[str, dict[str, Tensor | int]]
    losses: tuple[Tensor, ...]
    # Includes initialization when requested. Empty by default to limit memory.
    trajectory: tuple[dict[str, Tensor], ...] = ()


@dataclass(frozen=True)
class MetaGradientResult:
    gradient: dict[str, Tensor]
    outer_loss: float
    inner_losses: tuple[float, ...]


def mapping_dot(left: TensorMap, right: TensorMap) -> Tensor:
    """Dot product with scalar transfers only, including sharded parameters."""
    if not left or left.keys() != right.keys():
        raise ValueError("nonempty tensor mappings must have matching keys")
    device = next(iter(left.values())).device
    result = None
    for name, value in left.items():
        if value.shape != right[name].shape or value.device != right[name].device:
            raise ValueError(f"dot-product shape/device mismatch for {name}")
        term = (value * right[name]).sum().to(device)
        result = term if result is None else result + term
    return result


def _sqrt_with_zero_subgradient(value: Tensor) -> Tensor:
    # sqrt(0)'s infinite derivative otherwise creates 0 * inf in Adam at a
    # zero gradient coordinate. The composed squared-moment derivative is zero.
    positive = value > 0
    root = torch.where(positive, value, torch.ones_like(value)).sqrt()
    return torch.where(positive, root, torch.zeros_like(value))


def global_grad_norm(gradients: Mapping[str, Tensor | None]) -> Tensor:
    """Global L2 norm; move only scalars between parameter devices."""
    present = [gradient for gradient in gradients.values() if gradient is not None]
    if not present:
        return torch.tensor(0.0)
    device = present[0].device
    squared = None
    for gradient in present:
        if gradient.is_sparse:
            raise ValueError("sparse gradients are not supported")
        term = torch.linalg.vector_norm(gradient).square().to(device)
        squared = term if squared is None else squared + term
    return _sqrt_with_zero_subgradient(squared)


def functional_optimizer_step(
    params: TensorMap,
    gradients: Mapping[str, Tensor | None],
    config: OptimizerConfig,
    state: State | None = None,
    *,
    first_order: bool = False,
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor | int]]]:
    """Pure SGD/AdamW step, including differentiable global L2 clipping.

    ``None`` gradients skip both decay and state advancement, as torch.optim
    does. AdamW state uses per-parameter integer ``step``, ``exp_avg``, and
    ``exp_avg_sq``; SGD state uses ``momentum_buffer``. Existing tensor state
    may carry an exact-unroll graph. First-order state is always detached.
    """
    if params.keys() != gradients.keys():
        raise ValueError("parameter and gradient mappings must have matching keys")
    state = {} if state is None else state
    if state.keys() - params.keys():
        raise ValueError("optimizer state contains unknown parameters")
    next_state = {name: dict(values) for name, values in state.items()}
    clip = None
    if config.max_grad_norm is not None:
        clip = (config.max_grad_norm / (global_grad_norm(gradients) + 1e-6)).clamp(
            max=1.0
        )
    updated = {}
    for name, parameter in params.items():
        gradient = gradients[name]
        if gradient is None:
            updated[name] = parameter
            continue
        if gradient.is_sparse:
            raise ValueError("sparse gradients are not supported")
        if gradient.shape != parameter.shape or gradient.device != parameter.device:
            raise ValueError(f"gradient shape/device does not match parameter {name}")
        if clip is not None:
            gradient = gradient * clip.to(gradient.device)
        parameter_state = dict(state.get(name, {}))
        if config.name == "sgd":
            if config.weight_decay:
                gradient = gradient + config.weight_decay * parameter
            if config.momentum:
                old_buffer = parameter_state.get("momentum_buffer")
                momentum_buffer = (
                    gradient.clone()
                    if old_buffer is None
                    else config.momentum * old_buffer
                    + (1.0 - config.dampening) * gradient
                )
                parameter_state["momentum_buffer"] = momentum_buffer
                gradient = (
                    gradient + config.momentum * momentum_buffer
                    if config.nesterov
                    else momentum_buffer
                )
            value = parameter - config.lr * gradient
        else:
            beta1, beta2 = config.betas
            step = int(parameter_state.get("step", 0)) + 1
            old_first = parameter_state.get("exp_avg")
            old_second = parameter_state.get("exp_avg_sq")
            first_moment = (
                beta1 * (torch.zeros_like(parameter) if old_first is None else old_first)
                + (1.0 - beta1) * gradient
            )
            second_moment = (
                beta2 * (torch.zeros_like(parameter) if old_second is None else old_second)
                + (1.0 - beta2) * gradient.square()
            )
            denominator = (
                _sqrt_with_zero_subgradient(second_moment) / math.sqrt(1.0 - beta2**step)
                + config.eps
            )
            value = parameter * (1.0 - config.lr * config.weight_decay)
            value = value - (config.lr / (1.0 - beta1**step)) * first_moment / denominator
            parameter_state.update(
                step=step, exp_avg=first_moment, exp_avg_sq=second_moment
            )
        updated[name] = (
            parameter + (value - parameter).detach() if first_order else value
        )
        if parameter_state:
            next_state[name] = parameter_state
    if first_order:
        next_state = {
            name: {
                key: value.detach() if isinstance(value, Tensor) else value
                for key, value in values.items()
            }
            for name, values in next_state.items()
        }
    return updated, next_state


def functional_forward(
    model: nn.Module, params: TensorMap, buffers: TensorMap
) -> Callable[..., Any]:
    """Bind a model to functional state without replacing its parameters."""

    def forward(*args: Any, **kwargs: Any) -> Any:
        return functional_call(model, (params, buffers), args, kwargs, strict=False)

    return forward


@torch.enable_grad()
def functional_unroll(
    model: nn.Module,
    initial_params: TensorMap,
    batches: Iterable[Any],
    loss_fn: LossFn,
    config: OptimizerConfig,
    *,
    first_order: bool = False,
    buffers: TensorMap | None = None,
    initial_state: State | None = None,
    store_trajectory: bool = False,
) -> UnrollResult:
    """Apply one optimizer step per batch, preserving the initialization graph.

    Pass only the parameters to adapt; omitted model parameters remain fixed.
    Buffers are cloned once and may evolve privately, as in an ordinary attack.
    Losses are detached scalars; intermediate parameter maps are retained only
    when ``store_trajectory=True`` (needed for trajectory outer objectives).
    """
    if not initial_params or any(
        not parameter.requires_grad for parameter in initial_params.values()
    ):
        raise ValueError("initial_params must be nonempty and require gradients")
    params = dict(initial_params)
    source_buffers = dict(model.named_buffers()) if buffers is None else buffers
    working_buffers = {name: value.clone() for name, value in source_buffers.items()}
    state = {
        name: dict(values) for name, values in (initial_state or {}).items()
    }
    trajectory = [dict(params)] if store_trajectory else []
    losses = []
    for batch in batches:
        loss = loss_fn(functional_forward(model, params, working_buffers), batch)
        if loss.ndim != 0:
            raise ValueError("loss_fn must return a scalar")
        gradients = torch.autograd.grad(
            loss,
            tuple(params.values()),
            create_graph=not first_order,
            allow_unused=True,
        )
        params, state = functional_optimizer_step(
            params,
            dict(zip(params, gradients)),
            config,
            state,
            first_order=first_order,
        )
        losses.append(loss.detach())
        if store_trajectory:
            trajectory.append(dict(params))
    return UnrollResult(params, working_buffers, state, tuple(losses), tuple(trajectory))


@torch.enable_grad()
def meta_gradient(
    model: nn.Module,
    initial_params: TensorMap,
    batches: Iterable[Any],
    inner_loss_fn: LossFn,
    outer_batch: Any,
    outer_loss_fn: LossFn,
    config: OptimizerConfig,
    *,
    first_order: bool = False,
    buffers: TensorMap | None = None,
    initial_state: State | None = None,
) -> MetaGradientResult:
    """Gradient of an explicitly supplied terminal outer loss at attack end."""
    result = functional_unroll(
        model,
        initial_params,
        batches,
        inner_loss_fn,
        config,
        first_order=first_order,
        buffers=buffers,
        initial_state=initial_state,
    )
    loss = outer_loss_fn(
        functional_forward(model, result.params, result.buffers), outer_batch
    )
    gradients = torch.autograd.grad(
        loss, tuple(initial_params.values()), allow_unused=True
    )
    gradient = {
        name: torch.zeros_like(parameter) if value is None else value.detach()
        for (name, parameter), value in zip(initial_params.items(), gradients)
    }
    return MetaGradientResult(
        gradient, loss.detach().item(), tuple(value.item() for value in result.losses)
    )


def gradient_diagnostic(
    model: nn.Module,
    initial_params: TensorMap,
    batches: Iterable[Any],
    inner_loss_fn: LossFn,
    outer_batch: Any,
    outer_loss_fn: LossFn,
    config: OptimizerConfig,
    *,
    buffers: TensorMap | None = None,
    initial_state: State | None = None,
    direction: TensorMap | None = None,
    fd_epsilons: Iterable[float] = (1e-3, 1e-4, 1e-5),
    seed: int = 1729,
) -> dict[str, Any]:
    """Compare exact/first-order gradients and central directional differences.

    Data are materialized once so every attack sees the same batches. Finite
    differences rerun the *actual* attack from perturbed initial parameters.
    A nonzero held-out outer loss avoids comparing a stationary inner loss.
    The caller must also ensure deterministic model forwards and loss callbacks.
    """
    batches = tuple(batches)
    if not batches:
        raise ValueError("at least one attack batch is required")
    common = dict(buffers=buffers, initial_state=initial_state)
    exact = meta_gradient(
        model,
        initial_params,
        batches,
        inner_loss_fn,
        outer_batch,
        outer_loss_fn,
        config,
        **common,
    )
    first = meta_gradient(
        model,
        initial_params,
        batches,
        inner_loss_fn,
        outer_batch,
        outer_loss_fn,
        config,
        first_order=True,
        **common,
    )
    if direction is None:
        generators = {
            value.device: torch.Generator(device=value.device).manual_seed(seed)
            for value in initial_params.values()
        }
        direction = {
            name: torch.randn(
                value.shape,
                device=value.device,
                dtype=value.dtype,
                generator=generators[value.device],
            )
            for name, value in initial_params.items()
        }
    if initial_params.keys() != direction.keys():
        raise ValueError("direction must have the same keys as initial_params")
    direction_norm = global_grad_norm(direction)
    if not torch.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("direction must have a finite positive norm")
    direction = {
        name: value / direction_norm.to(value.device) for name, value in direction.items()
    }
    exact_directional = mapping_dot(exact.gradient, direction).item()
    first_directional = mapping_dot(first.gradient, direction).item()
    norm_exact = global_grad_norm(exact.gradient).item()
    norm_first = global_grad_norm(first.gradient).item()
    cosine = (
        mapping_dot(exact.gradient, first.gradient).item() / (norm_exact * norm_first)
        if norm_exact and norm_first
        else None
    )
    differences = []
    for epsilon in fd_epsilons:
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("finite-difference epsilons must be positive and finite")
        losses = []
        for sign in (1, -1):
            shifted = {
                name: (value.detach() + sign * epsilon * direction[name])
                .detach()
                .requires_grad_()
                for name, value in initial_params.items()
            }
            # Numerical attack values do not need a higher-order graph.
            with torch.enable_grad():
                result = functional_unroll(
                    model,
                    shifted,
                    batches,
                    inner_loss_fn,
                    config,
                    first_order=True,
                    **common,
                )
                loss = outer_loss_fn(
                    functional_forward(model, result.params, result.buffers), outer_batch
                )
            losses.append(loss.detach().item())
        finite_difference = (losses[0] - losses[1]) / (2 * epsilon)
        differences.append(
            {
                "epsilon": epsilon,
                "finite_difference": finite_difference,
                "exact_absolute_error": abs(finite_difference - exact_directional),
                "first_order_absolute_error": abs(finite_difference - first_directional),
                "exact_relative_error": abs(finite_difference - exact_directional)
                / max(abs(finite_difference), abs(exact_directional), 1e-12),
            }
        )
    return {
        "optimizer": config.name,
        "steps": len(batches),
        "exact_outer_loss": exact.outer_loss,
        "first_order_outer_loss": first.outer_loss,
        "inner_losses": exact.inner_losses,
        "exact_gradient_norm": norm_exact,
        "first_order_gradient_norm": norm_first,
        "gradient_cosine": cosine,
        "exact_directional_derivative": exact_directional,
        "first_order_directional_derivative": first_directional,
        "finite_differences": differences,
    }


def tiny_problem(device: str = "cpu") -> tuple[nn.Module, tuple[Any, ...], Any, LossFn]:
    """A deterministic nonlinear FP64 problem with separate attack/outer data."""
    # Fork the CPU RNG so importing/calling this fixture does not reseed a run.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2)).double()
        attack = tuple(
            (torch.randn(4, 3, dtype=torch.float64), torch.randn(4, 2, dtype=torch.float64))
            for _ in range(3)
        )
        outer = (
            torch.randn(5, 3, dtype=torch.float64) + 0.4,
            torch.randn(5, 2, dtype=torch.float64) - 0.6,
        )
    model = model.to(device).eval()
    attack = tuple(tuple(value.to(device) for value in batch) for batch in attack)
    outer = tuple(value.to(device) for value in outer)

    def loss_fn(forward: Callable[..., Tensor], batch: Any) -> Tensor:
        return (forward(batch[0]) - batch[1]).square().mean()

    return model, attack, outer, loss_fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, batches, outer, loss_fn = tiny_problem(args.device)
    config = OptimizerConfig(
        name=args.optimizer,
        lr=args.lr,
        weight_decay=0.07,
        max_grad_norm=args.max_grad_norm,
    )
    result = gradient_diagnostic(
        model,
        dict(model.named_parameters()),
        batches[: args.steps],
        loss_fn,
        outer,
        loss_fn,
        config,
    )
    result.update(device=args.device, dtype="float64", torch_version=torch.__version__)
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
