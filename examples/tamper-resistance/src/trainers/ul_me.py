"""Maximum-entropy unlearning with balanced forget and retain batches."""

import math

import torch

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import (
    BalancedSourceSamplerv2,
    MetadataCollator,
    global_l2_norm,
)


class MaxEntTrainer(AxolotlTrainer):
    """Minimize retain cross-entropy while maximizing forget entropy."""

    def __init__(
        self,
        *args,
        forget_coefficient=1.0,
        retain_coefficient=1.0,
        sampling="balanced",
        log_gradient_norms=False,
        defence_forget_batch_size=1,
        defence_retain_batch_size=1,
        **kwargs,
    ):
        if sampling not in {"balanced", "authors", "optml"}:
            raise ValueError("sampling must be 'balanced', 'authors', or 'optml'")
        super().__init__(*args, **kwargs)
        self.forget_coefficient = forget_coefficient
        self.retain_coefficient = retain_coefficient
        self.sampling = sampling
        self.log_gradient_norms = log_gradient_norms

        self.defence_forget_batch_size = defence_forget_batch_size
        self.defence_retain_batch_size = defence_retain_batch_size

        self.metadata_to_idx = {
            "defence::forget": 0,
            "defence::retain": 1,
            "attack::forget": 2,
        }
        self.metadata_to_batch_size = {
            "defence::forget": self.defence_forget_batch_size,
            "defence::retain": self.defence_retain_batch_size,
        }
        self.data_collator = MetadataCollator(self.data_collator, self.metadata_to_idx)

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        return BalancedSourceSamplerv2(
            dataset,
            metadata_to_idx=self.metadata_to_idx,
            metadata_to_batch_size=self.metadata_to_batch_size,
        )

    def compute_loss_term(
        self,
        model,
        inputs,
        coefficient,
        return_grads=False,
        return_outputs=False,
        model_kwargs=None,
    ):
        outputs = model(**inputs, **(model_kwargs or {}))
        loss = coefficient * outputs.loss.float()

        if return_grads:
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if return_outputs:
                return loss, grads, outputs
            return loss, grads
        if return_outputs:
            return loss, outputs
        return loss

    def compute_forget_loss(
        self,
        model,
        inputs,
        return_grads=False,
        return_outputs=False,
        model_kwargs=None,
    ):
        outputs = model(**inputs, **(model_kwargs or {}))
        log_probs = torch.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
        probs = torch.exp(log_probs)
        negative_entropy_per_token = torch.sum(probs * log_probs, dim=-1)
        valid = inputs["labels"][:, 1:] != -100
        if not valid.any():
            raise ValueError("MaxEnt requires a supervised token in every forget batch")
        entropy = -torch.mean(negative_entropy_per_token[valid])
        loss = -self.forget_coefficient * entropy
        self._last_forget_entropy = entropy.detach()

        if return_grads:
            parameters = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            grads = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            if return_outputs:
                return loss, grads, outputs
            return loss, grads
        if return_outputs:
            return loss, outputs
        return loss

    def compute_retain_loss(
        self, model, inputs, return_grads=False, return_outputs=False
    ):
        return self.compute_loss_term(
            model,
            inputs,
            self.retain_coefficient,
            return_grads,
            return_outputs,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["metadata_ids"] == self.metadata_to_idx["defence::forget"]
        retain_mask = inputs["metadata_ids"] == self.metadata_to_idx["defence::retain"]
        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        loss_forget, forget_outputs = self.compute_forget_loss(
            model, forget_inputs, return_outputs=True
        )
        loss_retain = self.compute_retain_loss(model, retain_inputs)
        loss = loss_forget + loss_retain

        if model.training and torch.is_grad_enabled():
            metrics = {
                "entropy_forget": self._last_forget_entropy.item(),
                "entropy_ceiling": math.log(forget_outputs.logits.shape[-1]),
                "loss_retain": loss_retain.detach().item(),
            }
            if self.log_gradient_norms:
                parameters = tuple(
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                grad_forget = torch.autograd.grad(
                    loss_forget,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                grad_retain = torch.autograd.grad(
                    loss_retain,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                metrics.update(
                    {
                        "grad_norm_forget": global_l2_norm(
                            gradient for gradient in grad_forget if gradient is not None
                        ).item(),
                        "grad_norm_retain": global_l2_norm(
                            gradient for gradient in grad_retain if gradient is not None
                        ).item(),
                    }
                )
            self.log(metrics)
        return (loss, forget_outputs) if return_outputs else loss
