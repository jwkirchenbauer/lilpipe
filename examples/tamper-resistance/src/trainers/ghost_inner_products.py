"""Detached per-example gradient Gram matrices without full parameter gradients.

Losses and supported modules must be independent across the leading batch axis.
The returned Gram is suitable for diagnostics or choosing gradient weights, not
for a differentiable orthogonality penalty.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
