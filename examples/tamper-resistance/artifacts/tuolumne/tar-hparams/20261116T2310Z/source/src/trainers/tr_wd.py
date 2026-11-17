"""Weight-distance regularization layered on unlearning trainers."""

import torch
import torch.distributed as dist

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_me import MaxEntTrainer
from trainers.ul_npo import NPOTrainer
from trainers.ul_rmu import RMUTrainer
from trainers.utils import load_reference_parameters

try:
    from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
except ImportError:  # pragma: no cover - older torch without FSDP2
    DTensor = ()
    Replicate = ()
    distribute_tensor = None


def global_rms_distance(parameters, anchors, eps=1e-8):
    """Return the authors' normalized full-model RMS parameter distance."""
    pairs = tuple(zip(parameters, anchors, strict=True))
    if not pairs:
        raise ValueError("global_rms_distance requires at least one parameter")

    uses_dtensor = any(isinstance(parameter, DTensor) for parameter, _ in pairs)
    if uses_dtensor and not dist.is_initialized():
        raise RuntimeError(
            "DTensor weight distance requires distributed initialization"
        )

    device = pairs[0][0].device
    value_squared = torch.zeros((), device=device, dtype=torch.float32)
    gradient_squared = torch.zeros((), device=device, dtype=torch.float32)
    element_count = sum(parameter.numel() for parameter, _ in pairs)
    world_size = dist.get_world_size() if uses_dtensor else 1

    for parameter, anchor in pairs:
        if isinstance(parameter, DTensor):
            delta = parameter.to_local().float() - anchor.to_local().float()
            is_replicated = all(
                isinstance(placement, Replicate) for placement in parameter.placements
            )
            value_divisor = world_size if is_replicated else 1
        else:
            delta = parameter.float() - anchor.float()
            value_divisor = 1
        squared = delta.square().sum()
        value_squared = value_squared + squared.to(device) / value_divisor
        gradient_squared = gradient_squared + squared.to(device)

    global_squared = value_squared.detach().clone()
    if uses_dtensor:
        dist.all_reduce(global_squared, op=dist.ReduceOp.SUM)
    distance = (global_squared / element_count + eps).sqrt()
    local_gradient = (gradient_squared - gradient_squared.detach()) / (
        2 * distance * element_count
    )
    return distance + local_gradient


class WDBaseTrainer:
    """Add an RMS weight-distance term to an unlearning trainer.

    Mix in before an unlearning trainer. The distance is measured from the
    anchor model; every other argument is forwarded to the unlearning trainer.
    """

    def __init__(
        self,
        *args,
        anchor_model_name_or_path,
        coefficient_weight_dist=1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_model_name_or_path = anchor_model_name_or_path
        self.coefficient_weight_dist = coefficient_weight_dist
        self._theta_anchor = None
        self._weight_distance_logged_step = None

    @property
    def theta(self):
        return {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    @property
    def theta_anchor(self):
        theta = self.theta
        if self._theta_anchor is None:
            self._theta_anchor = load_reference_parameters(
                theta, self.anchor_model_name_or_path
            )
        if theta.keys() != self._theta_anchor.keys():
            raise RuntimeError(
                "trainable parameters changed after weight-distance snapshot"
            )
        return self._theta_anchor

    def compute_weight_dist_loss(self):
        return -self.coefficient_weight_dist * global_rms_distance(
            self.theta.values(), self.theta_anchor.values()
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_unlearn, outputs = super().compute_loss(
            model, inputs, return_outputs=True, **kwargs
        )
        loss_weight_dist = self.compute_weight_dist_loss()
        loss = loss_unlearn + loss_weight_dist

        step = self.state.global_step
        if step != self._weight_distance_logged_step:
            self.log({"loss_weight_dist": loss_weight_dist.detach().item()})
            self._weight_distance_logged_step = step
        return (loss, outputs) if return_outputs else loss


class WDGDTrainer(WDBaseTrainer, GradDiffTrainer):
    """Add weight distance to gradient difference."""


class WDMaxEntTrainer(WDBaseTrainer, MaxEntTrainer):
    """Add weight distance to MaxEnt."""


class WDNPOTrainer(WDBaseTrainer, NPOTrainer):
    """Add weight distance to NPO."""


class WDRMUTrainer(WDBaseTrainer, RMUTrainer):
    """Add weight distance to RMU."""
