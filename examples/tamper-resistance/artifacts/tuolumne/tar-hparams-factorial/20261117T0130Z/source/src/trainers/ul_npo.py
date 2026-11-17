"""Negative Preference Optimization for unlearning."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import MixedSourceSampler, load_reference_parameters


def sequence_cross_entropy(logits, labels):
    """Return each sequence's mean next-token cross-entropy."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_labels, reduction="none"
    )
    valid = shifted_labels != -100
    return (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class NPOTrainer(AxolotlTrainer):
    """Optimize NPO against the initial model while retaining benign data."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        beta=0.0225,
        retain_coefficient=1.0,
        **kwargs,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if retain_coefficient < 0:
            raise ValueError("retain_coefficient must be non-negative")
        super().__init__(*args, **kwargs)
        self.reference_model_name_or_path = reference_model_name_or_path
        self.beta = beta
        self.retain_coefficient = retain_coefficient
        self._theta_ref = None
        self._npo_logged_step = None

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @property
    def theta(self):
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def theta_ref(self):
        theta = self.theta
        if self._theta_ref is None:
            self._theta_ref = load_reference_parameters(
                theta, self.reference_model_name_or_path
            )
        if theta.keys() != self._theta_ref.keys():
            raise RuntimeError(
                "trainable parameters changed after loading the reference"
            )
        return self._theta_ref

    @contextmanager
    def reference_parameters(self):
        theta = self.theta
        theta_current = {name: parameter.data for name, parameter in theta.items()}
        try:
            for name, parameter in theta.items():
                parameter.data = self.theta_ref[name]
            yield
        finally:
            for name, parameter in theta.items():
                parameter.data = theta_current[name]

    @staticmethod
    def split_sources(inputs):
        forget_mask = inputs["is_forget"].bool()
        retain_mask = ~forget_mask
        if not forget_mask.any() or not retain_mask.any():
            raise ValueError("NPO requires forget and retain rows in every batch")
        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        retain_inputs = {field: inputs[field][retain_mask] for field in fields}
        return forget_inputs, retain_inputs

    def compute_forget_loss(
        self, model, inputs, return_outputs=False, model_kwargs=None
    ):
        forget_inputs, _ = self.split_sources(inputs)
        outputs = model(**forget_inputs, **(model_kwargs or {}))
        current_ce = sequence_cross_entropy(outputs.logits, forget_inputs["labels"])
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad(), self.reference_parameters():
                reference_outputs = model(**forget_inputs)
                reference_ce = sequence_cross_entropy(
                    reference_outputs.logits, forget_inputs["labels"]
                )
        finally:
            model.train(was_training)
        loss = (
            -(2 / self.beta)
            * F.logsigmoid(self.beta * (current_ce - reference_ce)).mean()
        )
        return (loss, outputs) if return_outputs else loss

    def compute_retain_loss(self, model, inputs):
        _, retain_inputs = self.split_sources(inputs)
        return self.retain_coefficient * model(**retain_inputs).loss.float()

    def compute_npo_loss(self, model, inputs, log_metrics=True):
        loss_forget, outputs = self.compute_forget_loss(
            model, inputs, return_outputs=True
        )
        loss_retain = self.compute_retain_loss(model, inputs)
        loss = loss_forget + loss_retain

        step = self.state.global_step
        if log_metrics and step != self._npo_logged_step:
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                }
            )
            self._npo_logged_step = step
        return loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = self.compute_npo_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss
