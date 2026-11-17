"""Suppressing Spurious Unlearning Neurons for Robust Unlearning (SSIUU)."""

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from trainers.ul_gd import GradDiffTrainer


class SSIUUTrainer(GradDiffTrainer):
    """Add the paper's attribution-drift regularizer to gradient difference."""

    def __init__(
        self,
        *args,
        ssiuu_coefficient=1e-3,
        probability_normalization="none",
        ssiuu_save_on_cpu=False,
        **kwargs,
    ):
        if ssiuu_coefficient < 0:
            raise ValueError("ssiuu_coefficient must be non-negative")
        if probability_normalization not in {"none", "geometric_mean"}:
            raise ValueError(
                "probability_normalization must be 'none' or 'geometric_mean'"
            )
        super().__init__(*args, **kwargs)
        self.ssiuu_coefficient = ssiuu_coefficient
        self.probability_normalization = probability_normalization
        self.ssiuu_save_on_cpu = ssiuu_save_on_cpu
        self._previous_attributions = None
        self._ssiuu_logged_step = None

    @staticmethod
    def forget_inputs(inputs):
        forget_mask = inputs["is_forget"].bool()
        if not forget_mask.any():
            raise ValueError("SSIUU requires forget rows in every batch")
        fields = ("input_ids", "attention_mask", "labels")
        return {field: inputs[field][forget_mask] for field in fields}

    @staticmethod
    def sequence_probability(logits, labels, normalization="none"):
        """Sum P(y|x) over a batch of causal-LM prompt/answer examples."""
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:].to(shift_logits.device)
        target_mask = shift_labels.ne(-100)
        if not target_mask.any():
            raise ValueError("SSIUU requires at least one labelled forget token")

        safe_labels = shift_labels.masked_fill(~target_mask, 0)
        token_log_probabilities = (
            F.log_softmax(shift_logits, dim=-1)
            .gather(dim=-1, index=safe_labels.unsqueeze(-1))
            .squeeze(-1)
        )
        sequence_log_probabilities = (token_log_probabilities * target_mask).sum(dim=-1)
        if normalization == "geometric_mean":
            sequence_log_probabilities = sequence_log_probabilities / target_mask.sum(
                dim=-1
            ).clamp_min(1)
        elif normalization != "none":
            raise ValueError("normalization must be 'none' or 'geometric_mean'")
        return sequence_log_probabilities.exp().sum()

    def compute_attributions(self, model, inputs, create_graph=None):
        """Compute batch-aggregated parameter attributions from Equation 12."""
        if create_graph is None:
            create_graph = getattr(self, "_attribution_create_graph", True)
        parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("SSIUU requires at least one trainable parameter")

        forget_inputs = self.forget_inputs(inputs)
        outputs = model(**forget_inputs, use_cache=False)
        probability = self.sequence_probability(
            outputs.logits,
            forget_inputs["labels"],
            getattr(self, "probability_normalization", "none"),
        )
        probability_gradients = torch.autograd.grad(
            probability,
            parameters,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        if create_graph:
            attributions = tuple(
                None if gradient is None else parameter * gradient
                for parameter, gradient in zip(parameters, probability_gradients)
            )
        else:
            # The first iteration has no SSIUU penalty. Build its reference
            # directly on CPU and release the unnecessary higher-order graph.
            attributions = tuple(
                (
                    None
                    if gradient is None
                    else parameter.detach().cpu() * gradient.detach().cpu()
                )
                for parameter, gradient in zip(parameters, probability_gradients)
            )
        return probability, attributions

    def compute_ssiuu_loss(self, model, inputs):
        """Penalize step-to-step drift at currently negative attributions."""
        offload = (
            getattr(self, "ssiuu_save_on_cpu", False)
            and next(model.parameters()).is_cuda
        )
        graph_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if offload
            else nullcontext()
        )
        with graph_context:
            previous = self._previous_attributions
            if previous is None:
                self._attribution_create_graph = False
                try:
                    probability, attributions = self.compute_attributions(model, inputs)
                finally:
                    self._attribution_create_graph = True
            else:
                probability, attributions = self.compute_attributions(model, inputs)
            self._last_ssiuu_probability = probability.detach()

            if previous is None:
                loss = probability.new_zeros(())
            else:
                terms = [
                    (current[current < 0] - old.to(current.device)[current < 0])
                    .square()
                    .sum()
                    for current, old in zip(attributions, previous)
                    if current is not None and old is not None
                ]
                loss = sum(terms, probability.new_zeros(()))

        # Algorithm 1 stores A(t) as the constant reference for iteration t + 1.
        self._previous_attributions = tuple(
            None if attribution is None else attribution.detach().to("cpu", copy=True)
            for attribution in attributions
        )
        return self.ssiuu_coefficient * loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        del kwargs
        loss_forget, outputs = self.compute_forget_loss(
            model, inputs, return_outputs=True
        )
        loss_gd = loss_forget + self.compute_retain_loss(model, inputs)
        if model.training and torch.is_grad_enabled():
            loss_ssiuu = self.compute_ssiuu_loss(model, inputs)
        else:
            loss_ssiuu = loss_gd.new_zeros(())
        loss = loss_gd + loss_ssiuu

        if model.training and torch.is_grad_enabled():
            step = self.state.global_step
            if step != self._ssiuu_logged_step:
                self.log(
                    {
                        "loss_ssiuu": loss_ssiuu.detach().item(),
                        "ssiuu_sequence_probability": self._last_ssiuu_probability.item(),
                    }
                )
                self._ssiuu_logged_step = step
        return (loss, outputs) if return_outputs else loss


# Backwards-compatible name used by early experiment configurations.
SSIUUGDTrainer = SSIUUTrainer
