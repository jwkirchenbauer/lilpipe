"""Axolotl trainer with passive ghost Gram logging."""

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.ghost_trainer import GhostInnerProductsMixin


class GhostAxolotlTrainer(GhostInnerProductsMixin, AxolotlTrainer):
    """Keep Axolotl training behavior and log ghost gradient inner products."""
