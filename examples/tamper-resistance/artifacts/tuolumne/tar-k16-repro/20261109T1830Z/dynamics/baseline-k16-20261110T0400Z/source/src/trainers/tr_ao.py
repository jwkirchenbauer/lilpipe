"""Activation orthogonality layered on unlearning trainers."""

import torch

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_npo import NPOTrainer
from trainers.ul_rmu import RMUTrainer

FORGET_HIDDEN_STATES = {"output_hidden_states": True, "use_cache": False}


def mean_pairwise_cosine(hidden_states, attention_mask):
    """Average off-diagonal cosine of masked-mean activations across layers."""
    lengths = attention_mask.sum(1).clamp_min(1).to(hidden_states[0].dtype)
    off_diagonal_cosines = []
    for layer_hidden_states in hidden_states:
        pooled = (layer_hidden_states * attention_mask.unsqueeze(-1)).sum(
            1
        ) / lengths.unsqueeze(-1)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        pairwise_cosine = pooled @ pooled.T
        off_diagonal = ~torch.eye(
            pooled.shape[0], dtype=torch.bool, device=pooled.device
        )
        off_diagonal_cosines.append(pairwise_cosine[off_diagonal].mean())
    return torch.stack(off_diagonal_cosines).mean()


def validate_target_layers(target_layers, num_layers):
    """Return validated zero-based transformer block indices."""
    target_layers = tuple(target_layers)
    if (
        not target_layers
        or len(set(target_layers)) != len(target_layers)
        or any(layer < 0 or layer >= num_layers for layer in target_layers)
    ):
        raise ValueError(f"target_layers must be unique indices in 0-{num_layers - 1}")
    return target_layers


class AOBaseTrainer:
    """Add a forget-set activation-orthogonality term to an unlearning trainer.

    Mix in before an unlearning trainer whose ``compute_forget_loss`` accepts
    ``model_kwargs`` and whose ``compute_loss`` returns the forget-pass outputs.
    Every argument except the AO ones is forwarded to that trainer.
    """

    def __init__(
        self,
        *args,
        activation_orthogonality_coefficient=1.0,
        target_layers=None,
        **kwargs,
    ):
        if activation_orthogonality_coefficient < 0:
            raise ValueError(
                "activation_orthogonality_coefficient must be non-negative"
            )
        super().__init__(*args, **kwargs)
        self.activation_orthogonality_coefficient = activation_orthogonality_coefficient
        if target_layers is None:
            target_layers = range(self.model.config.num_hidden_layers)
        self.target_layers = validate_target_layers(
            target_layers, self.model.config.num_hidden_layers
        )
        self._ao_logged_step = None

    def compute_activation_orthogonality_loss(self, hidden_states, attention_mask):
        selected = tuple(hidden_states[layer + 1] for layer in self.target_layers)
        return self.activation_orthogonality_coefficient * mean_pairwise_cosine(
            selected, attention_mask
        )

    def compute_forget_loss(self, model, inputs, *args, **kwargs):
        return super().compute_forget_loss(
            model, inputs, *args, model_kwargs=FORGET_HIDDEN_STATES, **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_mask = inputs["is_forget"].bool()
        if int(forget_mask.sum()) < 2:
            raise ValueError("AO requires at least two forget rows per batch")

        loss_unlearn, forget_outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        loss_activation_orthogonality = self.compute_activation_orthogonality_loss(
            forget_outputs.hidden_states, inputs["attention_mask"][forget_mask]
        )
        loss = loss_unlearn + loss_activation_orthogonality

        step = self.state.global_step
        if step != self._ao_logged_step:
            self.log(
                {
                    "loss_activation_orthogonality": (
                        loss_activation_orthogonality.detach().item()
                    )
                }
            )
            self._ao_logged_step = step
        return (loss, forget_outputs) if return_outputs else loss


class AOGDTrainer(AOBaseTrainer, GradDiffTrainer):
    """Add forget-set activation orthogonality to gradient difference."""


class AONPOTrainer(AOBaseTrainer, NPOTrainer):
    """Add forget-set activation orthogonality to NPO."""


class AORMUTrainer(AOBaseTrainer, RMUTrainer):
    """Add forget-set activation orthogonality to RMU."""

    # RMU reads its forget activations from the hidden states it already requests.
    compute_forget_loss = RMUTrainer.compute_forget_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        forget_inputs, _ = self.split_sources(inputs)
        if forget_inputs["input_ids"].shape[0] < 2:
            raise ValueError("RMU+AO requires at least two forget rows per batch")

        hidden_states = model(
            **forget_inputs,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        loss_forget = self.compute_forget_loss_from_hidden_states(
            hidden_states, forget_inputs
        )
        loss_retain = self.compute_retain_loss(model, inputs)
        loss_activation_orthogonality = self.compute_activation_orthogonality_loss(
            hidden_states, forget_inputs["attention_mask"]
        )
        loss = loss_forget + loss_retain + loss_activation_orthogonality

        step = self.state.global_step
        if step != self._rmu_logged_step:
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "loss_activation_orthogonality": (
                        loss_activation_orthogonality.detach().item()
                    ),
                }
            )
            self._rmu_logged_step = step

        outputs = {
            "loss_forget": loss_forget,
            "loss_retain": loss_retain,
            "loss_activation_orthogonality": loss_activation_orthogonality,
        }
        return (loss, outputs) if return_outputs else loss
