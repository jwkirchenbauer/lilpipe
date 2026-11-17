"""Relearning after a one-time Gaussian parameter perturbation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from axolotl.core.trainers.base import AxolotlTrainer

from trainers.rl_bf16 import BF16Trainer
from trainers.tr_seam import SEAMRelearnTrainer


def add_gaussian_parameter_noise(model, sigma, seed):
    """Apply one reproducible N(0, sigma^2) delta to every floating parameter."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("gaussian_sigma must be a finite non-negative number")

    generators = {}
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.is_floating_point():
                continue
            if parameter.device not in generators:
                generators[parameter.device] = torch.Generator(
                    device=parameter.device
                ).manual_seed(seed + len(generators))
            noise = torch.randn(
                parameter.shape,
                generator=generators[parameter.device],
                dtype=torch.float32,
                device=parameter.device,
            )
            parameter.copy_((parameter.float() + sigma * noise).to(parameter.dtype))


class GaussianNoiseMixin:
    def __init__(self, *args, gaussian_sigma=0.0, gaussian_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        add_gaussian_parameter_noise(self.model, gaussian_sigma, gaussian_seed)
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_seed = gaussian_seed
        self._save_perturbed_checkpoint = True

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Save after Trainer has started, so checkpoint-0 is not mistaken for a
        # checkpoint from which the new attack should automatically resume.
        if self._save_perturbed_checkpoint:
            self.save_model(str(Path(self.args.output_dir) / "checkpoint-0"))
            self._save_perturbed_checkpoint = False
        return super().training_step(model, inputs, num_items_in_batch)


class GaussianTrainer(GaussianNoiseMixin, AxolotlTrainer):
    """Perturb once, then relearn with Axolotl's default dtype policy."""


class GaussianBF16Trainer(GaussianNoiseMixin, BF16Trainer):
    """Perturb once, then relearn with every parameter forced to BF16."""


class GaussianSEAMRelearnTrainer(GaussianNoiseMixin, SEAMRelearnTrainer):
    """Perturb once, then relearn while tracking SEAM gradient geometry."""
