"""Observe ordinary Trainer backward passes and log gradient Gram statistics."""

import torch
from transformers import TrainerCallback

from trainers.ghost_inner_products import (
    GradientTerm,
    linear_weight_gram,
    module_kind,
    norm_gradient_terms,
    term_gram,
)


class GhostInnerProductsMixin:
    """Install gradient hooks without overriding training_step or compute_loss.

    Compose before HF Trainer or AxolotlTrainer. Each backward must correspond
    to one independent, batch-first microbatch. Logs the average microbatch Gram
    mean per optimizer step, retaining the loss and accumulation normalization
    while removing AMP loss scaling. Full [B,B,T,S] intermediates are allocated.
    """

    def __init__(self, *args, ghost_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        if ghost_dtype not in (torch.float32, torch.float64):
            raise ValueError("ghost_dtype must be torch.float32 or torch.float64")
        if (
            self.accelerator.num_processes != 1
            or self.is_deepspeed_enabled
            or self.is_fsdp_enabled
            or self.args.n_gpu > 1
        ):
            raise NotImplementedError(
                "Ghost logging requires single-process, single-device training"
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
        self.gram = None
        self.previous_uses = {}
        self.microbatch_means = []
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

    def on_module_forward(self, module, inputs, kwargs, output):
        if self.computing_gram or not module.training or not torch.is_grad_enabled():
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

        self.output_handles.append(output.register_hook(on_module_backward))

    def _accumulate_parameter(self, parameter, current, contribution=None):
        if parameter is None or not parameter.requires_grad:
            return
        contribution = (
            term_gram(current, current) if contribution is None else contribution
        )
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
            self.gram.add_(cross + cross.T)
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
            scaler = getattr(self.accelerator, "scaler", None)
            scale = scaler.get_scale() if scaler is not None else 1.0
            self.microbatch_means.append(self.gram.mean() / scale**2)
        self.gram = None
        self.previous_uses.clear()
        for handle in self.output_handles:
            handle.remove()
        self.output_handles.clear()

    def log_step(self):
        if self.microbatch_means:
            value = torch.stack(self.microbatch_means).mean().item()
            self.log({"ghost_gram_mean": value})
        self.microbatch_means.clear()

    def close(self):
        for handle in self.hook_handles + self.output_handles:
            handle.remove()
        self.hook_handles.clear()
        self.output_handles.clear()
        self.previous_uses.clear()
        self.microbatch_means.clear()
        self.gram = None


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
        self.trainer.close()
