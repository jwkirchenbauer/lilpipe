"""GIP: gradient cosine defense and passive gradient inner-product tracking.

Shared functions remain usable in PyTorch-only numerical tests. GD/NPO can add
finite-difference squared cosine between individual forget gradients. The
refusal trainer supports group cosine; relearning passively logs alignment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from data.seam import HARMFUL, REFUSAL

# Shared gradient inner-product calculations


@dataclass(frozen=True)
class GradientTerm:
    """A vector gradient or a sum of outer products, with an optional index axis."""

    left: torch.Tensor
    right: torch.Tensor | None = None
    indexed: bool = False


def gradient_terms(module, kind, inputs, delta, *, dtype=torch.float32):
    """Yield (parameter, term) pairs for a linear, Conv1D, or embedding use."""
    n = inputs.shape[0]
    d = delta.detach().to(dtype)
    if kind in {"linear", "conv1d"}:
        a = inputs.detach().to(dtype).reshape(n, -1, inputs.shape[-1])
        b = d.reshape(n, -1, d.shape[-1])
        left, right = (b, a) if kind == "linear" else (a, b)
        yield module.weight, GradientTerm(left, right)
        yield module.bias, GradientTerm(b.sum(dim=1))
    elif kind == "embedding":
        ids = inputs.detach().reshape(n, -1)
        b = d.reshape(n, -1, d.shape[-1])
        if module.padding_idx is not None:
            b = b * (ids != module.padding_idx).unsqueeze(-1)
        yield module.weight, GradientTerm(ids, b, indexed=True)
    else:
        raise ValueError(f"Unsupported term kind: {kind}")


def term_inner_product(a, b, i, j, *, chunk_size=32):
    """Contract two gradient terms for examples i and j in bounded chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if a.right is None or b.right is None:
        if a.right is not None or b.right is not None:
            raise NotImplementedError("Mixed vector/matrix parameter sharing")
        return torch.dot(a.left[i], b.left[j])
    total = a.right.new_zeros(())
    for t in range(0, a.right.shape[1], chunk_size):
        av = a.right[i, t : t + chunk_size]
        al = a.left[i, t : t + chunk_size]
        for s in range(0, b.right.shape[1], chunk_size):
            bv = b.right[j, s : s + chunk_size]
            bl = b.left[j, s : s + chunk_size]
            if a.indexed and b.indexed:
                left_dot = al[:, None] == bl[None, :]
            elif a.indexed:
                left_dot = bl[:, al].T
            elif b.indexed:
                left_dot = al[:, bl]
            else:
                left_dot = al @ bl.T
            total += (left_dot * (av @ bv.T)).sum()
    return total


def assemble_gram(
    terms: Iterable[tuple[nn.Parameter | None, GradientTerm]],
    *,
    batch_size,
    device,
    dtype=torch.float32,
    chunk_size=32,
):
    """Sum every ordered use pair for each shared trainable parameter."""
    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive")
    groups = defaultdict(list)
    for parameter, term in terms:
        if parameter is not None and parameter.requires_grad:
            groups[id(parameter)].append(term)
    result = torch.zeros((batch_size, batch_size), device=device, dtype=dtype)
    with torch.no_grad(), torch.autocast(result.device.type, enabled=False):
        for group in groups.values():
            for i in range(batch_size):
                for j in range(i, batch_size):
                    value = result.new_zeros(())
                    for a in group:
                        for b in group:
                            value += term_inner_product(
                                a, b, i, j, chunk_size=chunk_size
                            )
                    result[i, j] += value
                    if i != j:
                        result[j, i] += value
    return result


def norm_gradient_terms(module, inputs, delta, *, dtype, amp, amp_dtype):
    """Reproduce small norm parameter VJPs, including zero scales and autocast."""
    parameters = [p for p in module.parameters(recurse=False) if p.requires_grad]
    rows = [[] for _ in parameters]
    for i in range(inputs.shape[0]):
        with (
            torch.enable_grad(),
            torch.autocast(inputs.device.type, enabled=amp, dtype=amp_dtype),
        ):
            local = module(inputs[i : i + 1])
            gradients = torch.autograd.grad(
                local,
                parameters,
                grad_outputs=delta[i : i + 1],
                allow_unused=True,
            )
        for row, parameter, gradient in zip(rows, parameters, gradients):
            row.append(torch.zeros_like(parameter) if gradient is None else gradient)
    for parameter, row in zip(parameters, rows):
        yield parameter, GradientTerm(
            torch.stack(row).detach().to(dtype).reshape(inputs.shape[0], -1)
        )


def module_kind(module):
    """Recognize parameter owners whose per-example gradient formula is supported."""
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Embedding):
        if module.scale_grad_by_freq or module.max_norm is not None or module.sparse:
            raise NotImplementedError(
                "Embedding requires scale_grad_by_freq=False, max_norm=None, sparse=False"
            )
        return "embedding"
    cls = type(module)
    if cls.__module__ == "transformers.pytorch_utils" and cls.__name__ == "Conv1D":
        return "conv1d"
    if isinstance(module, (nn.LayerNorm, nn.RMSNorm)) or (
        cls.__module__.startswith("transformers.models.")
        and cls.__name__.endswith("RMSNorm")
    ):
        return "norm"
    raise NotImplementedError(f"Unsupported trainable module: {cls.__name__}")


def causal_lm_losses(logits, labels):
    """Mean shifted cross entropy per example; fully ignored examples return zero."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Expected logits [N,T,V], labels [N,T], and T >= 2")
    targets = labels[:, 1:].to(logits.device)
    token_losses = F.cross_entropy(
        logits[:, :-1].transpose(1, 2), targets, reduction="none", ignore_index=-100
    )
    return token_losses.sum(1) / (targets != -100).sum(1).clamp_min(1)


def term_gram(a: GradientTerm, b: GradientTerm) -> torch.Tensor:
    """Return all example-pair inner products without materializing gradients."""
    if a.right is None or b.right is None:
        if a.right is not None or b.right is not None:
            raise NotImplementedError("Mixed vector/matrix parameter sharing")
        return a.left @ b.left.T
    if a.indexed and b.indexed:
        left_dots = a.left[:, None, :, None] == b.left[None, :, None, :]
    elif a.indexed:
        batch_a = a.left.shape[0]
        batch_b, positions_b = b.left.shape[:2]
        left_dots = (
            b.left[None]
            .expand(batch_a, -1, -1, -1)
            .gather(-1, a.left[:, None, None, :].expand(-1, batch_b, positions_b, -1))
            .transpose(-1, -2)
        )
    elif b.indexed:
        return term_gram(b, a).T
    else:
        left_dots = torch.einsum("bti,csi->bcts", a.left, b.left)
    right_dots = torch.einsum("bto,cso->bcts", a.right, b.right)
    return torch.einsum("bcts,bcts->bc", left_dots.to(right_dots.dtype), right_dots)


def linear_weight_gram(z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Compute a linear weight's gradient Gram matrix.

    Args:
        z: Input activations. Shape [B, T, I].
        d: Output gradients (dL/dz_output). Shape [B, T, O].
    """
    ztz = torch.einsum("bti,csi->bcts", z, z)
    dtd = torch.einsum("bto,cso->bcts", d, d)
    return torch.einsum("bcts,bcts->bc", ztz, dtd)


def mean_gradient_cosine(
    gram: torch.Tensor, *, squared: bool = False
) -> torch.Tensor | None:
    """Average off-diagonal gradient cosines, excluding zero-norm gradients.

    Args:
        gram: Gradient inner products. Shape [B, B].
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be a square [B, B] tensor")
    norms = gram.diagonal().clamp_min(0).sqrt()
    valid = norms > 0
    pairs = valid[:, None] & valid[None, :]
    pairs.fill_diagonal_(False)
    if not pairs.any():
        return None
    denominators = norms[:, None] * norms[None, :]
    cosines = (gram[pairs] / denominators[pairs]).clamp(-1, 1)
    return cosines.square().mean() if squared else cosines.mean()


# Finite-difference cosine regularization


@dataclass(frozen=True)
class CosineGradients:
    loss_a: torch.Tensor
    loss_b: torch.Tensor
    gradient_a: tuple[torch.Tensor, ...]
    gradient_b: tuple[torch.Tensor, ...]
    regularization: tuple[torch.Tensor, ...]
    cosine: torch.Tensor
    norm_a: torch.Tensor
    norm_b: torch.Tensor
    skipped: bool
    evaluations: int


def gradient_dot(first, second):
    """Dot parameter-aligned gradients; only scalar totals cross devices."""
    partials = {}
    for a, b in zip(first, second, strict=True):
        value = (a * b).sum()
        partials[a.device] = partials.get(a.device, 0) + value
    if not partials:
        raise ValueError("gradient sequences must be nonempty")
    device = next(iter(partials))
    return torch.stack([value.to(device) for value in partials.values()]).sum()


def finite_difference_cosine(
    model: nn.Module,
    loss_a: Callable[[], torch.Tensor],
    loss_b: Callable[[], torch.Tensor],
    *,
    tau: float,
    rho: float,
    skip_threshold: float = 0.0,
    squared: bool = False,
    offload_originals: bool = False,
) -> CosineGradients:
    """Estimate a cosine or squared-cosine derivative with ordinary gradients.

    Loss callables return scalar means; single-example calls target individual
    pairwise cosine. FP32 parameters may span devices; FP64 is supported for
    validation. Each nonzero direction is normalized to perturbation length rho
    and its norm rescales the gradient difference. Squared cosine multiplies
    the estimate by the detached chain-rule coefficient 2*cosine.

    Base and perturbed evaluations replay matching torch RNG states. Parameters,
    buffers and existing gradients are restored, including on exceptions. Only
    scalar reductions cross GPUs. Optional CPU parameter backups reduce memory;
    directions are computed one parameter at a time rather than stored.
    """
    if not math.isfinite(tau) or tau <= 0 or not math.isfinite(rho) or rho <= 0:
        raise ValueError("tau and rho must be finite and positive")
    if not math.isfinite(skip_threshold) or skip_threshold < 0:
        raise ValueError("skip_threshold must be finite and nonnegative")
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    if not parameters:
        raise ValueError("model must have trainable parameters")
    if len({p.dtype for p in parameters}) != 1 or parameters[0].dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("finite differences require FP32 parameters (FP64 for tests)")
    originals = tuple(
        p.detach().to("cpu" if offload_originals else p.device).clone()
        for p in parameters
    )
    buffers = tuple(model.buffers())
    original_buffers = tuple(b.detach().clone() for b in buffers)
    cuda_devices = sorted({p.device.index for p in parameters if p.is_cuda})
    evaluations = 0

    def rng_state():
        return torch.get_rng_state(), [
            torch.cuda.get_rng_state(i) for i in cuda_devices
        ]

    def restore_rng(state):
        torch.set_rng_state(state[0])
        for device, value in zip(cuda_devices, state[1], strict=True):
            torch.cuda.set_rng_state(value, device)

    def restore_model():
        with torch.no_grad():
            for p, value in zip(parameters, originals, strict=True):
                p.copy_(value.to(p.device))
            for b, value in zip(buffers, original_buffers, strict=True):
                b.copy_(value)

    def evaluate(loss_fn):
        nonlocal evaluations
        evaluations += 1
        with (
            torch.enable_grad(),
            torch.autocast(parameters[0].device.type, enabled=False),
        ):
            loss = loss_fn()
            if loss.ndim != 0:
                raise ValueError("loss callables must return scalar means")
            gradients = torch.autograd.grad(
                loss, parameters, create_graph=False, allow_unused=True
            )
        gradients = tuple(
            torch.zeros_like(p) if g is None else g.detach()
            for p, g in zip(parameters, gradients, strict=True)
        )
        return loss.detach(), gradients

    rng_after_base = rng_state()
    try:
        rng_a = rng_state()
        value_a, g_a = evaluate(loss_a)
        restore_model()
        rng_b = rng_state()
        value_b, g_b = evaluate(loss_b)
        rng_after_base = rng_state()
        restore_model()
        norm_a = gradient_dot(g_a, g_a).sqrt()
        norm_b = gradient_dot(g_b, g_b).sqrt()
        if not torch.isfinite(norm_a) or not torch.isfinite(norm_b):
            raise FloatingPointError("gradient norm is not finite")
        a = (norm_a.square() + tau**2).sqrt()
        b = (norm_b.square() + tau**2).sqrt()
        cosine = gradient_dot(g_a, g_b) / (a * b)
        skipped = bool((norm_a < skip_threshold) | (norm_b < skip_threshold))
        regularization = [torch.zeros_like(p) for p in parameters]
        if not skipped and not (squared and cosine == 0):
            for base, other, loss_fn, rng, denominator in zip(
                (g_a, g_b),
                (g_b, g_a),
                (loss_a, loss_b),
                (rng_a, rng_b),
                (a.square(), b.square()),
                strict=True,
            ):

                def direction():
                    return (
                        other_g / (a * b).to(base_g.device)
                        - cosine.to(base_g.device)
                        * base_g
                        / denominator.to(base_g.device)
                        for base_g, other_g in zip(base, other, strict=True)
                    )

                length = gradient_dot(direction(), direction()).sqrt()
                if length == 0:
                    continue
                restore_model()
                with torch.no_grad():
                    for p, original, v in zip(
                        parameters, originals, direction(), strict=True
                    ):
                        p.copy_(original.to(p.device) + rho * v / length.to(p.device))
                restore_rng(rng)
                _, perturbed = evaluate(loss_fn)
                factor = length / rho
                if squared:
                    factor = factor * (2 * cosine)
                with torch.no_grad():
                    for r, plus, initial in zip(
                        regularization, perturbed, base, strict=True
                    ):
                        r.add_((plus - initial) * factor.to(r.device))
                del perturbed
        return CosineGradients(
            value_a,
            value_b,
            g_a,
            g_b,
            tuple(regularization),
            cosine,
            norm_a,
            norm_b,
            skipped,
            evaluations,
        )
    finally:
        restore_model()
        restore_rng(rng_after_base)


# Ghost tracking


class GhostCapture:
    """Capture one batch forward on an existing model; always remove hooks on exit.

    Use ``with GhostCapture(model) as capture:`` around the forward and call
    ``capture.gram(losses, retain_graph=True)`` before the ordinary backward.
    Each Gram call consumes its captures, including when it fails. Batch-coupled
    operations and direct parameter uses outside captured modules are unsupported.
    """

    def __init__(self, model, *, chunk_size=32, dtype=torch.float32):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.model = model
        self.chunk_size = chunk_size
        self.dtype = dtype
        self._records = []
        self._handles = []
        self._computing = False

    def __enter__(self):
        if self._handles:
            raise RuntimeError("Capture is already active")
        if getattr(self.model, "is_gradient_checkpointing", False):
            raise NotImplementedError(
                "Ghost capture does not support gradient checkpointing"
            )
        # Validate everything before installing any hooks.
        modules = [
            (module, module_kind(module))
            for module in self.model.modules()
            if any(p.requires_grad for p in module.parameters(recurse=False))
        ]
        try:
            for module, kind in modules:
                self._handles.append(
                    module.register_forward_hook(self._hook(kind), with_kwargs=True)
                )
        except BaseException:
            self.close()
            raise
        return self

    def _hook(self, kind):
        def capture(module, args, kwargs, output):
            if self._computing or not torch.is_grad_enabled():
                return
            inputs = (
                args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            )
            if not isinstance(inputs, torch.Tensor) or not isinstance(
                output, torch.Tensor
            ):
                raise ValueError("Supported modules require tensor inputs and outputs")
            self._records.append(
                (
                    module,
                    kind,
                    inputs.detach(),
                    output,
                    torch.is_autocast_enabled(inputs.device.type),
                    torch.get_autocast_dtype(inputs.device.type),
                )
            )

        return capture

    def gram(self, losses, *, retain_graph=False):
        """Return detached G[i,j] = <grad(losses[i]), grad(losses[j])>."""
        try:
            if not self._records:
                raise RuntimeError("Run a forward inside the capture context first")
            if losses.ndim != 1 or losses.numel() == 0 or not losses.requires_grad:
                raise ValueError("losses must be a differentiable, nonempty [N] tensor")
            n = losses.shape[0]
            for _, _, x, y, _, _ in self._records:
                if x.ndim == 0 or y.ndim == 0 or x.shape[0] != n or y.shape[0] != n:
                    raise ValueError(
                        "All supported modules must use batch-first tensors with the "
                        "loss batch size; expand broadcast position_ids to [N,T]"
                    )
            self._computing = True
            deltas = torch.autograd.grad(
                losses.sum(),
                [record[3] for record in self._records],
                retain_graph=retain_graph,
                allow_unused=True,
            )
            terms = []
            for (module, kind, x, _, amp, amp_dtype), delta in zip(
                self._records, deltas
            ):
                if delta is None:
                    continue
                if kind == "norm":
                    terms.extend(
                        norm_gradient_terms(
                            module,
                            x,
                            delta,
                            dtype=self.dtype,
                            amp=amp,
                            amp_dtype=amp_dtype,
                        )
                    )
                else:
                    terms.extend(
                        gradient_terms(module, kind, x, delta, dtype=self.dtype)
                    )
            return assemble_gram(
                terms,
                batch_size=n,
                device=losses.device,
                dtype=self.dtype,
                chunk_size=self.chunk_size,
            )
        finally:
            self._records.clear()
            self._computing = False

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._records.clear()

    def __exit__(self, *exc):
        self.close()


class GhostInnerProductsMixin:
    """Install gradient hooks without overriding training_step or compute_loss.

    Compose before HF Trainer or AxolotlTrainer. Each backward must correspond
    to one independent, batch-first microbatch. Logs the average microbatch
    off-diagonal gradient cosine per optimizer step. Common loss, accumulation,
    and AMP scales cancel. Full [B,B,T,S] intermediates are allocated.
    """

    ghost_forget_only = False

    def __init__(self, *args, ghost_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        if ghost_dtype not in (torch.float32, torch.float64):
            raise ValueError("ghost_dtype must be torch.float32 or torch.float64")
        if (
            self.accelerator.num_processes != 1
            or self.is_deepspeed_enabled
            or self.is_fsdp_enabled
            or (self.args.n_gpu > 1 and not self.is_model_parallel)
        ):
            raise NotImplementedError(
                "Ghost logging requires one process; multiple GPUs require model sharding"
            )
        if (
            getattr(self.model, "is_gradient_checkpointing", False)
            or self.args.gradient_checkpointing
        ):
            raise NotImplementedError(
                "Ghost logging does not support gradient checkpointing"
            )
        if getattr(self.args, "sample_packing", False):
            raise NotImplementedError("Ghost logging requires unpacked batches")
        self.ghost_dtype = ghost_dtype
        self.track_ghost = not self.ghost_forget_only
        self.gram = None
        self.previous_uses = {}
        self.microbatch_means = []
        self.microbatch_squared_means = []
        self.hook_handles = []
        self.output_handles = []
        self.computing_gram = False
        modules = [
            module
            for module in self.model.modules()
            if any(p.requires_grad for p in module.parameters(recurse=False))
        ]
        # Reject unsupported owners before installing any hooks.
        for module in modules:
            module_kind(module)
        try:
            for module in modules:
                self.hook_handles.append(
                    module.register_forward_hook(
                        self.on_module_forward, with_kwargs=True
                    )
                )
            self.add_callback(GhostLoggingCallback(self))
        except BaseException:
            self.close()
            raise

    def compute_forget_loss(self, *args, **kwargs):
        tracking = self.track_ghost
        # SAM revisits the forget loss at shifted weights. Measure its first,
        # unperturbed backward only, just like ordinary GD/NPO.
        self.track_ghost = self.gram is None
        try:
            return super().compute_forget_loss(*args, **kwargs)
        finally:
            self.track_ghost = tracking

    def on_module_forward(self, module, inputs, kwargs, output):
        if (
            not self.track_ghost
            or self.computing_gram
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return
        if not isinstance(output, torch.Tensor):
            raise ValueError("Ghost logging requires tensor module outputs")
        if not output.requires_grad:
            return
        z = inputs[0] if inputs else kwargs.get("input", kwargs.get("hidden_states"))
        if not isinstance(z, torch.Tensor):
            raise ValueError("Ghost logging requires tensor module inputs")
        z = z.detach()
        amp = torch.is_autocast_enabled(z.device.type)
        amp_dtype = torch.get_autocast_dtype(z.device.type)

        def on_module_backward(d):
            # d = dL/dz_output, where z_output is this module's output.
            handle.remove()
            self.computing_gram = True
            try:
                with torch.no_grad(), torch.autocast(z.device.type, enabled=False):
                    kind = module_kind(module)
                    if kind == "linear":
                        self.accumulate_gram_linear(module, z, d.detach())
                    elif kind == "embedding":
                        self.accumulate_gram_embedding(module, z, d.detach())
                    elif kind == "norm":
                        self.accumulate_gram_norm(module, z, d.detach(), amp, amp_dtype)
                    elif kind == "conv1d":
                        self.accumulate_gram_conv1d(module, z, d.detach())
            finally:
                self.computing_gram = False

        handle = output.register_hook(on_module_backward)
        self.output_handles.append(handle)

    def _accumulate_parameter(self, parameter, current, contribution=None):
        if parameter is None or not parameter.requires_grad:
            return
        contribution = (
            term_gram(current, current) if contribution is None else contribution
        ).to(self.args.device)
        if self.gram is None:
            self.gram = contribution
        else:
            if self.gram.shape != contribution.shape:
                raise ValueError(
                    "Modules must use the same batch size; expand broadcast position_ids to [B,T]"
                )
            self.gram.add_(contribution)
        previous = self.previous_uses.setdefault(id(parameter), [])
        for earlier in previous:
            cross = term_gram(current, earlier)
            self.gram.add_((cross + cross.T).to(self.gram.device))
        previous.append(current)

    def accumulate_gram_linear(self, module, z: torch.Tensor, d: torch.Tensor) -> None:
        """Accumulate a linear layer's weight and bias gradient Gram matrices.

        Args:
            module: Linear layer owning the parameters.
            z: Input activations. Shape [B, T, I].
            d: Output gradients (dL/dz_output). Shape [B, T, O].
        """
        batch = z.shape[0]
        z = z.to(self.ghost_dtype).reshape(batch, -1, z.shape[-1])
        d = d.to(self.ghost_dtype).reshape(batch, -1, d.shape[-1])
        if module.weight.requires_grad:
            self._accumulate_parameter(
                module.weight, GradientTerm(d, z), linear_weight_gram(z, d)
            )
        self._accumulate_parameter(module.bias, GradientTerm(d.sum(1)))

    def accumulate_gram_conv1d(self, module, z, d):
        batch = z.shape[0]
        z = z.to(self.ghost_dtype).reshape(batch, -1, z.shape[-1])
        d = d.to(self.ghost_dtype).reshape(batch, -1, d.shape[-1])
        self._accumulate_parameter(module.weight, GradientTerm(z, d))
        self._accumulate_parameter(module.bias, GradientTerm(d.sum(1)))

    def accumulate_gram_embedding(self, module, z, d):
        batch = z.shape[0]
        ids = z.reshape(batch, -1)
        d = d.to(self.ghost_dtype).reshape(batch, -1, d.shape[-1])
        if module.padding_idx is not None:
            d = d * (ids != module.padding_idx).unsqueeze(-1)
        self._accumulate_parameter(module.weight, GradientTerm(ids, d, indexed=True))

    def accumulate_gram_norm(self, module, z, d, amp, amp_dtype):
        for parameter, current in norm_gradient_terms(
            module, z, d, dtype=self.ghost_dtype, amp=amp, amp_dtype=amp_dtype
        ):
            self._accumulate_parameter(parameter, current)

    def finish_microbatch(self):
        if self.gram is not None:
            cosine = mean_gradient_cosine(self.gram)
            if cosine is not None:
                self.microbatch_means.append(cosine)
                self.microbatch_squared_means.append(
                    mean_gradient_cosine(self.gram, squared=True)
                )
        self.gram = None
        self.previous_uses.clear()
        for handle in self.output_handles:
            handle.remove()
        self.output_handles.clear()

    def log_step(self):
        if self.microbatch_means:
            value = torch.stack(self.microbatch_means).mean().item()
            squared = torch.stack(self.microbatch_squared_means).mean().item()
            self.log({"ghost_cosine_mean": value, "ghost_cosine_squared_mean": squared})
        self.microbatch_means.clear()
        self.microbatch_squared_means.clear()

    def close(self):
        for handle in self.hook_handles + self.output_handles:
            handle.remove()
        self.hook_handles.clear()
        self.output_handles.clear()
        self.previous_uses.clear()
        self.microbatch_means.clear()
        self.microbatch_squared_means.clear()
        self.gram = None


# Keep the shared functions usable without Trainer dependencies.
if importlib.util.find_spec("transformers") is not None:
    from transformers import TrainerCallback

    class GhostLoggingCallback(TrainerCallback):
        """Publish statistics after backward, leaving the training loop unchanged."""

        def __init__(self, trainer):
            self.trainer = trainer

        def on_substep_end(self, args, state, control, **kwargs):
            self.trainer.finish_microbatch()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.trainer.finish_microbatch()
            self.trainer.log_step()

        def on_train_end(self, args, state, control, **kwargs):
            try:
                directory = Path(args.output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                state.save_to_json(str(directory / "trainer_state.json"))
            finally:
                self.trainer.close()


# Defense batch sampling


class RefusalHarmfulSampler(torch.utils.data.Sampler[int]):
    """Shuffle harmful/refusal streams into equally sized batch halves."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 4 or batch_size % 4:
            raise ValueError("Defense batch size must be divisible by four")
        roles = dataset["seam_role"]
        self.groups = [
            [i for i, role in enumerate(roles) if role == value]
            for value in (HARMFUL, REFUSAL)
        ]
        if not all(self.groups):
            raise ValueError("Defense dataset requires harmful and refusal rows")
        self.per_group = batch_size // 2
        self.rows = min(map(len, self.groups)) // self.per_group * self.per_group
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        groups = [
            [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
            for group in self.groups
        ]
        for start in range(0, self.rows, self.per_group):
            for group in groups:
                yield from group[start : start + self.per_group]

    def __len__(self):
        return self.rows * 2


# Individual forget-example squared-cosine defense.


class ForgetGradientCosineMixin:
    """Add sampled individual-gradient squared cosine to the base defense.

    The ordinary GD/NPO backward remains authoritative. Additional evaluations
    use forget CE only, with optimizer gradients temporarily stored on CPU to
    bound memory when FP32 parameters span several GPUs.
    """

    def __init__(
        self,
        *args,
        cosine_coefficient=0.0,
        cosine_pairs=2,
        cosine_tau=1e-3,
        cosine_rho=1e-3,
        cosine_skip_threshold=1e-8,
        cosine_offload_originals=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not math.isfinite(cosine_coefficient) or cosine_coefficient < 0:
            raise ValueError("cosine_coefficient must be finite and nonnegative")
        if not isinstance(cosine_pairs, int) or cosine_pairs < 1:
            raise ValueError("cosine_pairs must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (cosine_tau, cosine_rho)):
            raise ValueError("cosine_tau and cosine_rho must be finite and positive")
        if not math.isfinite(cosine_skip_threshold) or cosine_skip_threshold < 0:
            raise ValueError("cosine_skip_threshold must be finite and nonnegative")
        if cosine_coefficient:
            if self.args.bf16 or self.args.fp16:
                raise ValueError("Squared gradient cosine requires FP32 without AMP")
            self.model.float()
        self.cosine_coefficient = cosine_coefficient
        self.cosine_pairs = cosine_pairs
        self.cosine_tau = cosine_tau
        self.cosine_rho = cosine_rho
        self.cosine_skip_threshold = cosine_skip_threshold
        self.cosine_offload_originals = cosine_offload_originals
        self._cosine_microbatch = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if not self.cosine_coefficient:
            return loss
        self.finish_microbatch()
        inputs = self._prepare_inputs(inputs)
        forget = torch.where(inputs["is_forget"].bool())[0]
        if len(forget) < 2:
            raise ValueError("Squared cosine requires at least two forget examples")
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        if any(p.dtype != torch.float32 for p in parameters):
            raise ValueError("Squared gradient cosine requires FP32 parameters")
        originals = tuple(
            None if p.grad is None else p.grad.detach().cpu().clone()
            for p in parameters
        )
        regularization = [torch.zeros_like(p, device="cpu") for p in parameters]
        for p in parameters:
            p.grad = None
        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        generator = torch.Generator().manual_seed(
            seed
            + self.state.global_step * self.args.gradient_accumulation_steps
            + self._cosine_microbatch
        )
        cuda_devices = sorted({p.device.index for p in parameters if p.is_cuda})
        previous_track = self.track_ghost
        self.track_ghost = False
        cosines, norms_a, norms_b = [], [], []
        skipped = evaluations = 0
        completed = False
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                for _ in range(self.cosine_pairs):
                    pair = forget[
                        torch.randperm(len(forget), generator=generator)[:2].to(
                            forget.device
                        )
                    ]
                    a, b = (
                        {
                            key: inputs[key][index : index + 1]
                            for key in ("input_ids", "attention_mask", "labels")
                        }
                        for index in pair.tolist()
                    )
                    result = finite_difference_cosine(
                        model,
                        lambda: causal_lm_losses(model(**a).logits, a["labels"]).mean(),
                        lambda: causal_lm_losses(model(**b).logits, b["labels"]).mean(),
                        tau=self.cosine_tau,
                        rho=self.cosine_rho,
                        skip_threshold=self.cosine_skip_threshold,
                        squared=True,
                        offload_originals=self.cosine_offload_originals,
                    )
                    for total, value in zip(
                        regularization, result.regularization, strict=True
                    ):
                        total.add_(value.cpu(), alpha=1 / self.cosine_pairs)
                    cosines.append(result.cosine.item())
                    norms_a.append(result.norm_a.item())
                    norms_b.append(result.norm_b.item())
                    skipped += int(result.skipped)
                    evaluations += result.evaluations
                    del result
            if any(not torch.isfinite(value).all() for value in regularization):
                raise FloatingPointError("Squared-cosine gradient is not finite")
            completed = True
        finally:
            self.track_ghost = previous_track
            divisor = self.current_gradient_accumulation_steps
            with torch.no_grad():
                for p, original, r in zip(
                    parameters, originals, regularization, strict=True
                ):
                    p.grad = None if original is None else original.to(p.device)
                    if completed and torch.count_nonzero(r):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.add_(
                            r.to(p.device), alpha=self.cosine_coefficient / divisor
                        )
        regularization_norm = gradient_dot(regularization, regularization).sqrt().item()
        base_gradients = [g for g in originals if g is not None]
        base_norm = (
            gradient_dot(base_gradients, base_gradients).sqrt().item()
            if base_gradients
            else 0.0
        )
        objective = sum(c * c for c in cosines) / self.cosine_pairs
        self.log(
            {
                "forget_pair_cosine": sum(cosines) / self.cosine_pairs,
                "forget_pair_cosine_squared": objective,
                "forget_pair_gradient_norm_a": sum(norms_a) / self.cosine_pairs,
                "forget_pair_gradient_norm_b": sum(norms_b) / self.cosine_pairs,
                "regularization_gradient_norm": regularization_norm,
                "regularization_skipped": skipped / self.cosine_pairs,
                "regularization_gradient_evaluations": evaluations,
                "base_update_gradient_norm": base_norm,
                "regularization_strength_ratio": self.cosine_coefficient
                * regularization_norm
                / (divisor * max(base_norm, 1e-30)),
            }
        )
        self._cosine_microbatch = (
            self._cosine_microbatch + 1
        ) % self.args.gradient_accumulation_steps
        return loss + self.cosine_coefficient * objective / divisor


# Defense and relearning trainers
if importlib.util.find_spec("axolotl") is not None:
    from axolotl.core.trainers.base import AxolotlTrainer
    from trainers.ul_gd import GradDiffTrainer
    from trainers.ul_npo import NPOTrainer
    from trainers.tr_sam import NPOSAMTrainer

    class GIPGradDiffTrainer(
        ForgetGradientCosineMixin, GhostInnerProductsMixin, GradDiffTrainer
    ):
        """GD with passive tracking and an optional individual squared-cosine penalty."""

        ghost_forget_only = True

    class GIPNPOTrainer(ForgetGradientCosineMixin, GhostInnerProductsMixin, NPOTrainer):
        """NPO with passive tracking and an optional individual squared-cosine penalty."""

        ghost_forget_only = True

    class GIPNPOSAMTrainer(GhostInnerProductsMixin, NPOSAMTrainer):
        """Track unperturbed forget gradients during unchanged NPO+SAM."""

        ghost_forget_only = True

    # Defense: retain loss plus finite-difference harmful-gradient cosine.
    class GIPTrainer(AxolotlTrainer):
        """FP32 refusal CE plus finite-difference cosine between harmful groups."""

        def __init__(
            self,
            *args,
            cosine_coefficient=0.01,
            cosine_tau=1e-3,
            cosine_rho=1e-3,
            cosine_skip_threshold=1e-8,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if not math.isfinite(cosine_coefficient) or cosine_coefficient < 0:
                raise ValueError("cosine_coefficient must be finite and nonnegative")
            if (
                not math.isfinite(cosine_tau)
                or cosine_tau <= 0
                or not math.isfinite(cosine_rho)
                or cosine_rho <= 0
            ):
                raise ValueError(
                    "cosine_tau and cosine_rho must be finite and positive"
                )
            if not math.isfinite(cosine_skip_threshold) or cosine_skip_threshold < 0:
                raise ValueError("cosine_skip_threshold must be finite and nonnegative")
            if self.args.bf16 or self.args.fp16:
                raise ValueError(
                    "Gradient cosine training initially requires FP32, without AMP"
                )
            if (
                self.accelerator.num_processes != 1
                or self.is_deepspeed_enabled
                or self.is_fsdp_enabled
                or self.args.n_gpu > 1
            ):
                raise NotImplementedError(
                    "Gradient cosine training requires one process and device"
                )
            if self.args.gradient_checkpointing or getattr(
                self.model, "is_gradient_checkpointing", False
            ):
                raise NotImplementedError(
                    "Gradient cosine training does not support checkpointing"
                )
            if getattr(self.args, "sample_packing", False):
                raise NotImplementedError(
                    "Gradient cosine training requires unpacked batches"
                )
            self.model.float()
            self.cosine_coefficient = cosine_coefficient
            self.cosine_tau = cosine_tau
            self.cosine_rho = cosine_rho
            self.cosine_skip_threshold = cosine_skip_threshold
            self.regularization_steps = 0
            self.skipped_steps = 0

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return RefusalHarmfulSampler(
                dataset,
                self._train_batch_size,
                (
                    self.args.data_seed
                    if self.args.data_seed is not None
                    else self.args.seed
                ),
            )

        def training_step(self, model, inputs, num_items_in_batch=None):
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()
            inputs = self._prepare_inputs(inputs)
            roles = inputs["seam_role"]
            harmful = torch.where(roles == HARMFUL)[0]
            refusal = torch.where(roles == REFUSAL)[0]
            if len(harmful) < 2 or len(harmful) % 2 or not len(refusal):
                raise ValueError(
                    "Each batch requires an even harmful group (>=2) and refusal rows"
                )
            parameters = tuple(p for p in model.parameters() if p.requires_grad)
            if any(p.dtype != torch.float32 for p in parameters):
                raise ValueError("All trainable parameters must remain FP32")
            fields = {
                key: value
                for key, value in inputs.items()
                if key not in {"seam_role", "source_index"}
            }
            retain_inputs = {key: value[refusal] for key, value in fields.items()}
            buffers = tuple(model.buffers())
            original_buffers = tuple(b.detach().clone() for b in buffers)
            try:
                with torch.autocast(parameters[0].device.type, enabled=False):
                    loss_retain = model(**retain_inputs).loss
                    gradients_retain = torch.autograd.grad(
                        loss_retain, parameters, create_graph=False, allow_unused=True
                    )
            finally:
                with torch.no_grad():
                    for b, original in zip(buffers, original_buffers, strict=True):
                        b.copy_(original)
            gradients = [
                torch.zeros_like(p) if g is None else g.detach()
                for p, g in zip(parameters, gradients_retain, strict=True)
            ]
            norm_retain = gradient_dot(gradients, gradients).sqrt()
            metrics = {
                "loss_retain": loss_retain.detach().item(),
                "retain_gradient_norm": norm_retain.item(),
            }
            objective = loss_retain.detach()
            if self.cosine_coefficient:
                order = harmful[torch.randperm(len(harmful), device=harmful.device)]
                group_a, group_b = order.chunk(2)
                inputs_a = {key: value[group_a] for key, value in fields.items()}
                inputs_b = {key: value[group_b] for key, value in fields.items()}
                result = finite_difference_cosine(
                    model,
                    lambda: causal_lm_losses(
                        model(**inputs_a).logits, inputs_a["labels"]
                    ).mean(),
                    lambda: causal_lm_losses(
                        model(**inputs_b).logits, inputs_b["labels"]
                    ).mean(),
                    tau=self.cosine_tau,
                    rho=self.cosine_rho,
                    skip_threshold=self.cosine_skip_threshold,
                )
                regularization_norm = gradient_dot(
                    result.regularization, result.regularization
                ).sqrt()
                strength_ratio = (
                    self.cosine_coefficient
                    * regularization_norm
                    / norm_retain.clamp_min(torch.finfo(norm_retain.dtype).tiny)
                )
                for gradient, regularization in zip(
                    gradients, result.regularization, strict=True
                ):
                    gradient.add_(regularization, alpha=self.cosine_coefficient)
                if not result.skipped:
                    objective = objective + self.cosine_coefficient * result.cosine
                self.regularization_steps += 1
                self.skipped_steps += int(result.skipped)
                metrics.update(
                    {
                        "gradient_cosine": result.cosine.item(),
                        "gradient_norm_a": result.norm_a.item(),
                        "gradient_norm_b": result.norm_b.item(),
                        "regularization_gradient_norm": regularization_norm.item(),
                        "regularization_strength_ratio": strength_ratio.item(),
                        "loss_harmful_a": result.loss_a.item(),
                        "loss_harmful_b": result.loss_b.item(),
                        "regularization_skipped": float(result.skipped),
                        "regularization_skip_frequency": self.skipped_steps
                        / self.regularization_steps,
                    }
                )
            divisor = self.current_gradient_accumulation_steps
            with torch.no_grad():
                for parameter, gradient in zip(parameters, gradients, strict=True):
                    if parameter.grad is None:
                        parameter.grad = gradient / divisor
                    else:
                        parameter.grad.add_(gradient / divisor)
            self.log(metrics)
            return objective / divisor

    # Relearning: ordinary training with passive ghost cosine logging.
    class GIPRelearnTrainer(GhostInnerProductsMixin, AxolotlTrainer):
        """Keep Axolotl training behavior and log ghost gradient inner products."""
