"""Circuit-breaker unlearning with frozen reference activations."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from axolotl.core.trainers.base import AxolotlTrainer

from trainers.utils import MixedSourceSampler, load_reference_parameters

TARGET_LAYERS = (5, 10, 15, 20, 25, 30)


def masked_mean(values, mask):
    """Average token values over non-padding positions."""
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def orthogonalization_loss(activations, attention_mask):
    """Penalize positive cosine between different forget sequences."""
    attention_mask = attention_mask.to(
        device=activations.device, dtype=activations.dtype
    )
    lengths = attention_mask.sum(dim=1).clamp_min(1)
    pooled = (activations * attention_mask[..., None]).sum(dim=1)
    pooled = F.normalize((pooled / lengths[:, None]).float(), dim=-1)
    similarities = pooled @ pooled.T
    off_diagonal = ~torch.eye(
        activations.shape[0], dtype=torch.bool, device=activations.device
    )
    positive_pairs = F.relu(similarities[off_diagonal])
    if positive_pairs.numel() == 0:
        return activations.float().sum() * 0
    return positive_pairs.mean() * lengths.float().mean()


class CircuitBreakerTrainer(AxolotlTrainer):
    """Retain benign activations and reroute forget activations."""

    def __init__(
        self,
        *args,
        reference_model_name_or_path,
        retain_coefficient=10.0,
        reroute_coefficient=23.0,
        orthogonalization_coefficient=5.0,
        target_layers=TARGET_LAYERS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reference_model_name_or_path = reference_model_name_or_path
        self.retain_coefficient = retain_coefficient
        self.reroute_coefficient = reroute_coefficient
        self.orthogonalization_coefficient = orthogonalization_coefficient
        self.target_layers = tuple(target_layers)
        self.microstep = 0
        self.total_microsteps = (
            self.args.max_steps * self.args.gradient_accumulation_steps
        )
        self._theta_ref = None

        num_layers = self.model.config.num_hidden_layers
        if len(set(self.target_layers)) != len(self.target_layers) or any(
            layer < 0 or layer >= num_layers for layer in self.target_layers
        ):
            raise ValueError(
                f"target_layers must be unique indices in 0-{num_layers - 1}"
            )

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return MixedSourceSampler(
            dataset,
            self.args.per_device_train_batch_size,
            self.args.data_seed if self.args.data_seed is not None else self.args.seed,
        )

    @property
    def theta(self):
        return dict(self.model.named_parameters())

    @property
    def theta_ref(self):
        theta = self.theta
        if self._theta_ref is None:
            self._theta_ref = load_reference_parameters(
                theta, self.reference_model_name_or_path
            )
        if theta.keys() != self._theta_ref.keys():
            raise RuntimeError("model parameters changed after loading the reference")
        return self._theta_ref

    @contextmanager
    def reference_parameters(self):
        """Temporarily evaluate the model at its frozen reference weights."""
        theta = self.theta
        theta_ref = self.theta_ref
        theta_current = {name: parameter.data for name, parameter in theta.items()}
        try:
            for name, parameter in theta.items():
                parameter.data = theta_ref[name]
            yield
        finally:
            for name, parameter in theta.items():
                parameter.data = theta_current[name]

    def coefficients(self):
        progress = min(max(self.microstep, 0) / max(self.total_microsteps - 1, 1), 1.0)
        return (
            self.retain_coefficient * (0.1 + 0.9 * progress),
            self.reroute_coefficient * (1.0 - 0.25 * progress),
            self.orthogonalization_coefficient * progress,
        )

    def selected_activations(self, model, input_ids, attention_mask):
        hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        return tuple(hidden_states[layer + 1] for layer in self.target_layers)

    def reference_activations(self, model, input_ids, attention_mask):
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad(), self.reference_parameters():
                return tuple(
                    activation.detach()
                    for activation in self.selected_activations(
                        model, input_ids, attention_mask
                    )
                )
        finally:
            model.train(was_training)

    @staticmethod
    def mean_layer_losses(losses):
        device = losses[0].device
        return torch.stack(tuple(loss.to(device) for loss in losses)).mean()

    def compute_cb_loss(
        self,
        model,
        inputs,
        *,
        advance_schedule=True,
        log_metrics=True,
    ):
        sample_mask_forget = inputs["is_forget"].bool()
        sample_mask_retain = ~sample_mask_forget
        if not sample_mask_forget.any() or not sample_mask_retain.any():
            raise ValueError(
                "CircuitBreaker requires forget and retain rows in every batch"
            )

        x_retain = inputs["input_ids"][sample_mask_retain]
        attn_mask_retain = inputs["attention_mask"][sample_mask_retain]
        x_forget = inputs["input_ids"][sample_mask_forget]
        attn_mask_forget = inputs["attention_mask"][sample_mask_forget]

        z_retain_ref = self.reference_activations(model, x_retain, attn_mask_retain)
        z_forget_ref = self.reference_activations(model, x_forget, attn_mask_forget)
        z_retain = self.selected_activations(model, x_retain, attn_mask_retain)
        z_forget = self.selected_activations(model, x_forget, attn_mask_forget)

        loss_retain = self.mean_layer_losses(
            tuple(
                masked_mean(
                    (current.float() - reference.float()).norm(dim=-1),
                    attn_mask_retain,
                )
                for current, reference in zip(z_retain, z_retain_ref, strict=True)
            )
        )
        loss_reroute = self.mean_layer_losses(
            tuple(
                masked_mean(
                    F.relu(
                        F.cosine_similarity(current.float(), reference.float(), dim=-1)
                    ),
                    attn_mask_forget,
                )
                for current, reference in zip(z_forget, z_forget_ref, strict=True)
            )
        )
        loss_orthogonalization = self.mean_layer_losses(
            tuple(
                orthogonalization_loss(activation, attn_mask_forget)
                for activation in z_forget
            )
        )

        retain_coefficient, reroute_coefficient, orthogonalization_coefficient = (
            self.coefficients()
        )
        loss = (
            retain_coefficient * loss_retain
            + reroute_coefficient * loss_reroute
            + orthogonalization_coefficient * loss_orthogonalization
        )

        if log_metrics:
            self.log(
                {
                    "loss_retain": loss_retain.detach().item(),
                    "loss_reroute": loss_reroute.detach().item(),
                    "loss_orthogonalization": loss_orthogonalization.detach().item(),
                    "retain_coefficient": retain_coefficient,
                    "reroute_coefficient": reroute_coefficient,
                    "orthogonalization_coefficient": orthogonalization_coefficient,
                }
            )
        if advance_schedule:
            self.microstep += 1
        return loss, {"retain": z_retain, "forget": z_forget}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = self.compute_cb_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss
