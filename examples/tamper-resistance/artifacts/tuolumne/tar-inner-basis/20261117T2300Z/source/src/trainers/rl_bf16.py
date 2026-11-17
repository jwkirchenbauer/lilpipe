"""Ordinary BF16 relearning trainer."""

from __future__ import annotations

import torch

from axolotl.core.trainers.base import AxolotlTrainer


def use_bfloat16_parameters(model):
    """Undo Axolotl's embedding/norm upcast to keep the whole model in BF16."""
    model.bfloat16()
    if hasattr(model, "config"):
        model.config.dtype = torch.bfloat16


class BF16Trainer(AxolotlTrainer):
    """Axolotl trainer that keeps every floating model tensor in BF16."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        use_bfloat16_parameters(self.model)
