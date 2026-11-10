"""Booster alignment trainer with harmful-gradient finite differences."""

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import gradient_dict_l2_norm, global_l2_norm


class BoosterMixedSourceSampler(torch.utils.data.Sampler[int]):
    """Yield equal shuffled harmful and safe subsets without replacement."""

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Booster sampling requires an even batch size")
        sources = dataset["is_harmful"]
        self.harmful = [index for index, value in enumerate(sources) if value]
        self.safe = [index for index, value in enumerate(sources) if not value]
        if not self.harmful or not self.safe:
            raise ValueError("Booster sampling requires harmful and safe rows")
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        import random

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        harmful = self.harmful.copy()
        safe = self.safe.copy()
        rng.shuffle(harmful)
        rng.shuffle(safe)
        half_batch = self.batch_size // 2
        usable = min(len(harmful), len(safe)) // half_batch * half_batch
        for start in range(0, usable, half_batch):
            batch = (
                harmful[start : start + half_batch] + safe[start : start + half_batch]
            )
            rng.shuffle(batch)
            yield from batch

    def __len__(self):
        half_batch = self.batch_size // 2
        return 2 * (min(len(self.harmful), len(self.safe)) // half_batch * half_batch)


def negative_normalized_perturbation(gradients, radius):
    """Return ``-radius * gradient / ||gradient||`` for active gradients."""
    active = [gradient for gradient in gradients.values() if gradient is not None]
    if not active:
        raise RuntimeError("Booster harmful backward produced no parameter gradients")
    norm = global_l2_norm(active)
    if not norm.isfinite() or norm == 0:
        raise RuntimeError(f"Booster harmful gradient has invalid norm: {norm}")
    return {
        name: (-radius * gradient / norm.to(gradient.device)).detach()
        for name, gradient in gradients.items()
        if gradient is not None
    }


class BoosterTrainer(AxolotlTrainer):
    """Apply safe NLL plus a harmful-gradient finite-difference penalty."""

    def __init__(self, *args, coefficient=5.0, radius=0.1, **kwargs):
        if coefficient <= 0:
            raise ValueError("coefficient must be positive")
        if radius <= 0:
            raise ValueError("radius must be positive")
        super().__init__(*args, **kwargs)
        self.coefficient = coefficient
        self.radius = radius

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )
        return BoosterMixedSourceSampler(
            dataset, self.args.per_device_train_batch_size, seed
        )

    @staticmethod
    def split_sources(inputs):
        harmful = inputs["is_harmful"].bool()
        if not harmful.any() or harmful.all():
            raise ValueError("Booster requires safe and harmful rows in every batch")
        fields = ("input_ids", "attention_mask", "labels")
        return (
            {field: inputs[field][~harmful] for field in fields},
            {field: inputs[field][harmful] for field in fields},
        )

    def _backward_and_take_gradients(self, loss, parameters):
        self.accelerator.backward(loss / self.current_gradient_accumulation_steps)
        gradients = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None
        return gradients

    def training_step(self, model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._booster_step(model, self._prepare_inputs(inputs))

    def _booster_step(self, model, inputs):
        model.train()
        safe_inputs, harmful_inputs = self.split_sources(inputs)
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        accumulated = {name: parameter.grad for name, parameter in parameters.items()}
        for parameter in parameters.values():
            parameter.grad = None

        try:
            with self.compute_loss_context_manager():
                harmful_loss = model(**harmful_inputs).loss.float()
            harmful_grad = self._backward_and_take_gradients(harmful_loss, parameters)
            perturbation = negative_normalized_perturbation(harmful_grad, self.radius)
            try:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].add_(step)
                with self.compute_loss_context_manager():
                    perturbed_harmful_loss = model(**harmful_inputs).loss.float()
                perturbed_grad = self._backward_and_take_gradients(
                    perturbed_harmful_loss, parameters
                )
            finally:
                with torch.no_grad():
                    for name, step in perturbation.items():
                        parameters[name].sub_(step)

            with self.compute_loss_context_manager():
                safe_loss = model(**safe_inputs).loss.float()
            safe_grad = self._backward_and_take_gradients(safe_loss, parameters)
            self.log(
                {
                    "loss_safe": safe_loss.detach().item(),
                    "loss_harmful": harmful_loss.detach().item(),
                    "loss_harmful_perturbed": perturbed_harmful_loss.detach().item(),
                    "grad_norm_safe": gradient_dict_l2_norm(safe_grad).item(),
                    "grad_norm_harmful": gradient_dict_l2_norm(harmful_grad).item(),
                    "grad_norm_harmful_perturbed": gradient_dict_l2_norm(
                        perturbed_grad
                    ).item(),
                    "perturbation_norm": global_l2_norm(perturbation.values()).item(),
                }
            )
        finally:
            for name, parameter in parameters.items():
                parameter.grad = accumulated[name]

        for name, parameter in parameters.items():
            terms = (safe_grad[name], harmful_grad[name], perturbed_grad[name])
            scales = (1.0, self.coefficient, -self.coefficient)
            for gradient, scale in zip(terms, scales, strict=True):
                if gradient is not None:
                    update = gradient.mul(scale)
                    parameter.grad = (
                        update
                        if parameter.grad is None
                        else parameter.grad.add_(update)
                    )

        objective = safe_loss + self.coefficient * (
            harmful_loss - perturbed_harmful_loss
        )
        return (objective / self.current_gradient_accumulation_steps).detach()
