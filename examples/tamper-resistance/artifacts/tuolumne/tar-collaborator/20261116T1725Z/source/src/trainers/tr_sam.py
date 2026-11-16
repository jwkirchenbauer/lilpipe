"""Sharpness-aware minimization (SAM) applied to unlearning objectives."""

import torch
from transformers import enable_full_determinism

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.ul_cb import CircuitBreakerTrainer
from trainers.ul_me import MaxEntTrainer
from trainers.ul_npo import NPOTrainer
from trainers.ul_rmu import RMUTrainer
from trainers.utils import (
    AuthorsMixedSourceSampler,
    MixedSourceSampler,
    OptmlMixedSourceSampler,
    global_l2_norm,
    gradient_dict_l2_norm,
)


def sam_perturbation(gradients, rho):
    """Return a radius-rho step from synchronized parameter gradients."""
    active_gradients = [
        gradient for gradient in gradients.values() if gradient is not None
    ]
    if not active_gradients:
        raise RuntimeError("SAM forget backward produced no parameter gradients")
    norm = global_l2_norm(active_gradients)
    if not norm.isfinite() or norm == 0:
        raise RuntimeError(f"SAM forget gradient has invalid norm: {norm}")
    return {
        name: (rho * gradient / norm.to(gradient.device)).detach()
        for name, gradient in gradients.items()
        if gradient is not None
    }


def optml_sam_perturbation(gradients, rho):
    """Reproduce OPTML's original-dtype, single-device norm calculation."""
    active_gradients = [
        gradient for gradient in gradients.values() if gradient is not None
    ]
    if not active_gradients:
        raise RuntimeError("SAM forget backward produced no parameter gradients")

    norm_device = active_gradients[0].device
    per_parameter_norms = [
        gradient.to(norm_device).norm(2) for gradient in active_gradients
    ]
    norm = torch.stack(per_parameter_norms).norm(2)
    if not norm.isfinite() or norm == 0:
        raise RuntimeError(f"SAM forget gradient has invalid norm: {norm}")
    return {
        name: (rho * gradient / norm.to(gradient.device)).detach()
        for name, gradient in gradients.items()
        if gradient is not None
    }


class GDSAMTrainer(AxolotlTrainer):
    """Apply a small SAM perturbation to gradient-difference unlearning."""

    def __init__(
        self,
        *args,
        rho=0.01,
        retain_coefficient=1.0,
        forget_coefficient=1.0,
        sampling="balanced",
        optml_parity=False,
        **kwargs,
    ):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.retain_coefficient = retain_coefficient
        self.forget_coefficient = forget_coefficient
        self.rho = rho
        self.sampling = sampling
        self.optml_parity = optml_parity
        if self.optml_parity:
            enable_full_determinism(self.args.seed)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self.sampling == "optml":
            return OptmlMixedSourceSampler(
                dataset,
                self.args.per_device_train_batch_size,
            )
        sampler_cls = (
            AuthorsMixedSourceSampler
            if self.sampling == "authors"
            else MixedSourceSampler
        )
        return sampler_cls(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("GradDiff requires forget and retain rows in every batch")

        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    def compute_forget_loss(self, model, inputs, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model(**forget_inputs)
        return -outputs.loss.float()

    def compute_retain_loss(self, model, inputs, **kwargs):
        _, retain_inputs = self.split_sources(inputs)
        outputs = model(**retain_inputs)
        return outputs.loss.float()

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        """Run FSDP-aware backward and remove its gradients from the model."""
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, grad_forget_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation_fn = (
                optml_sam_perturbation if self.optml_parity else sam_perturbation
            )
            perturbation = perturbation_fn(grad_forget_at_perturbation, self.rho)

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss_forget, grad_forget = self._backward_and_take_gradients(
                    lambda: self.forget_coefficient
                    * self.compute_forget_loss(model, inputs),
                    parameters,
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            loss_retain, grad_retain = self._backward_and_take_gradients(
                lambda: self.retain_coefficient
                * self.compute_retain_loss(model, inputs),
                parameters,
            )

            self.log(
                {
                    "loss_forget": (-loss_forget).detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": gradient_dict_l2_norm(grad_forget).item(),
                    "grad_norm_retain": gradient_dict_l2_norm(grad_retain).item(),
                    "grad_norm_forget_at_perturbation": gradient_dict_l2_norm(
                        grad_forget_at_perturbation
                    ).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (
                grad_forget[name],
                grad_retain[name],
            ):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )

        return (
            (loss_forget + loss_retain) / self.current_gradient_accumulation_steps
        ).detach()


class NPOSAMTrainer(NPOTrainer):
    """Apply SAM to NPO's forget objective."""

    def __init__(self, *args, rho=0.01, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        super().__init__(*args, **kwargs)
        self.rho = rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = self.theta
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, gradients_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation = sam_perturbation(gradients_at_perturbation, self.rho)
            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss_forget, gradients_forget = self._backward_and_take_gradients(
                    lambda: self.compute_forget_loss(model, inputs), parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            loss_retain, gradients_retain = self._backward_and_take_gradients(
                lambda: self.compute_retain_loss(model, inputs), parameters
            )
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": gradient_dict_l2_norm(gradients_forget).item(),
                    "grad_norm_retain": gradient_dict_l2_norm(gradients_retain).item(),
                    "grad_norm_forget_at_perturbation": gradient_dict_l2_norm(
                        gradients_at_perturbation
                    ).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (gradients_forget[name], gradients_retain[name]):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )
        return (
            (loss_forget + loss_retain) / self.current_gradient_accumulation_steps
        ).detach()


class RMUSAMTrainer(RMUTrainer):
    """Apply SAM to RMU's forget objective."""

    def __init__(self, *args, rho=0.01, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        super().__init__(*args, **kwargs)
        self.rho = rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = self.theta
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, gradients_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation = sam_perturbation(gradients_at_perturbation, self.rho)

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss_forget, gradients_forget = self._backward_and_take_gradients(
                    lambda: self.compute_forget_loss(model, inputs), parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            loss_retain, gradients_retain = self._backward_and_take_gradients(
                lambda: self.compute_retain_loss(model, inputs), parameters
            )
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": gradient_dict_l2_norm(gradients_forget).item(),
                    "grad_norm_retain": gradient_dict_l2_norm(gradients_retain).item(),
                    "grad_norm_forget_at_perturbation": gradient_dict_l2_norm(
                        gradients_at_perturbation
                    ).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (gradients_forget[name], gradients_retain[name]):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )

        return (
            (loss_forget + loss_retain) / self.current_gradient_accumulation_steps
        ).detach()


class MaxEntSAMTrainer(MaxEntTrainer):
    """Apply SAM to MaxEnt's forget objective."""

    def __init__(self, *args, rho=0.01, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        super().__init__(*args, **kwargs)
        self.rho = rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, gradients_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_forget_loss(model, inputs), parameters
            )
            perturbation = sam_perturbation(gradients_at_perturbation, self.rho)
            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss_forget, gradients_forget = self._backward_and_take_gradients(
                    lambda: self.compute_forget_loss(model, inputs), parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            loss_retain, gradients_retain = self._backward_and_take_gradients(
                lambda: self.compute_retain_loss(model, inputs), parameters
            )
            self.log(
                {
                    "entropy_forget": self._last_forget_entropy.item(),
                    "loss_retain": loss_retain.detach().item(),
                    "grad_norm_forget": gradient_dict_l2_norm(gradients_forget).item(),
                    "grad_norm_retain": gradient_dict_l2_norm(gradients_retain).item(),
                    "grad_norm_forget_at_perturbation": gradient_dict_l2_norm(
                        gradients_at_perturbation
                    ).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            for gradient in (gradients_forget[name], gradients_retain[name]):
                if gradient is not None:
                    parameter.grad = (
                        gradient
                        if parameter.grad is None
                        else parameter.grad.add_(gradient)
                    )
        return (
            (loss_forget + loss_retain) / self.current_gradient_accumulation_steps
        ).detach()


class CircuitBreakerSAMTrainer(CircuitBreakerTrainer):
    """Apply SAM to the complete circuit-breaker objective."""

    def __init__(self, *args, rho=0.01, **kwargs):
        if rho <= 0:
            raise ValueError("rho must be positive")
        super().__init__(*args, **kwargs)
        self.rho = rho

    def training_step(self, model, inputs, num_items_in_batch=None):
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._sam_training_step(model, self._prepare_inputs(inputs))

    def _backward_and_take_gradients(self, loss_fn, parameters):
        with self.compute_loss_context_manager():
            loss = loss_fn()
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return loss, gradients

    def _sam_training_step(self, model, inputs):
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated_gradients = {
            name: parameter.grad for name, parameter in parameters.items()
        }
        for parameter in parameters.values():
            parameter.grad = None

        try:
            _, gradients_at_perturbation = self._backward_and_take_gradients(
                lambda: self.compute_cb_loss(
                    model,
                    inputs,
                    advance_schedule=False,
                    log_metrics=False,
                )[0],
                parameters,
            )
            gradient_norm_at_perturbation = gradient_dict_l2_norm(
                gradients_at_perturbation
            )
            perturbation = (
                {}
                if gradient_norm_at_perturbation == 0
                else sam_perturbation(gradients_at_perturbation, self.rho)
            )

            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                loss, gradients = self._backward_and_take_gradients(
                    lambda: self.compute_cb_loss(model, inputs)[0], parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            self.log(
                {
                    "grad_norm": gradient_dict_l2_norm(gradients).item(),
                    "grad_norm_at_perturbation": (gradient_norm_at_perturbation.item()),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated_gradients[name]

        for name, parameter in parameters.items():
            gradient = gradients[name]
            if gradient is not None:
                parameter.grad = (
                    gradient
                    if parameter.grad is None
                    else parameter.grad.add_(gradient)
                )

        return (loss / self.current_gradient_accumulation_steps).detach()
