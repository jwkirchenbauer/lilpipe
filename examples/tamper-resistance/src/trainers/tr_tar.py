"""Tamper-resistant unlearning (TAR) with a pluggable simulated relearning attack."""

import copy
import gc
from collections.abc import Mapping
from functools import partial
from types import MethodType

import torch
import wandb
from torch.func import functional_call

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_me import MaxEntTrainer
from trainers.utils import BalancedSourceSamplerv2

ROLE_FIELDS = ("input_ids", "attention_mask", "labels")


def adamw_attack_v1(
    model,
    parameters,
    inputs,
    steps=8,
    learning_rate=2e-5,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    weight_decay=0.0,
    on_loss_backward=None,
):
    """Run AdamW relearning steps on forget rows and return the parameter delta.

    ``on_loss_backward(inner_step, loss)`` is called after each backward pass.
    """
    forget_mask = inputs["is_forget"].bool()
    if not forget_mask.any():
        raise ValueError("TAR attack requires forget rows in every batch")
    forget_inputs = {field: inputs[field][forget_mask] for field in ROLE_FIELDS}

    initial = {
        name: parameter.detach().clone() for name, parameter in parameters.items()
    }
    current = {
        name: value.clone().requires_grad_(True) for name, value in initial.items()
    }
    optimizer = torch.optim.AdamW(
        current.values(),
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        weight_decay=weight_decay,
    )
    for inner_step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = functional_call(model, current, (), forget_inputs).loss
        loss.backward(inputs=list(current.values()))
        if on_loss_backward is not None:
            on_loss_backward(inner_step, loss.detach())
        if all(value.grad is None for value in current.values()):
            raise ValueError(
                "attack loss has no gradient through the trainable parameters"
            )
        optimizer.step()

    return {name: (current[name] - initial[name]).detach() for name in current}


def adamw_attack(
    model,
    inputs,
    attack_forget_batch_size_per_iteration=1,
    attack_epochs=1,
    attack_learning_rate=2e-5,
    on_loss_backward=None,
):
    """Run AdamW epochs over already-selected attack rows, retaining partial batches."""
    if attack_forget_batch_size_per_iteration < 1 or attack_epochs < 0:
        raise ValueError("attack batch size must be positive and epochs nonnegative")
    n_samples = len(inputs["input_ids"])
    if not n_samples:
        raise ValueError("TAR attack requires at least one attack row")
    optimizer = torch.optim.AdamW(model.parameters(), lr=attack_learning_rate)
    inner_step = 0
    for _ in range(attack_epochs):
        for start in range(0, n_samples, attack_forget_batch_size_per_iteration):
            optimizer.zero_grad(set_to_none=True)
            loss = model(
                **{
                    field: inputs[field][
                        start : start + attack_forget_batch_size_per_iteration
                    ]
                    for field in ROLE_FIELDS
                }
            ).loss
            loss.backward()
            if on_loss_backward is not None:
                on_loss_backward(inner_step, loss.detach())
            optimizer.step()
            inner_step += 1
    return model


def validate_delta(parameters, delta):
    """Check that an attack delta is a detached, parameter-shaped mapping."""
    if not isinstance(delta, Mapping):
        raise TypeError("sample_attack_fn must return a parameter-delta mapping")
    unknown = set(delta) - set(parameters)
    if unknown:
        raise ValueError(f"attack delta contains unknown parameters: {sorted(unknown)}")
    for name, change in delta.items():
        parameter = parameters[name]
        if not isinstance(change, torch.Tensor):
            raise TypeError(f"attack delta for {name!r} is not a tensor")
        for field in ("shape", "device", "dtype"):
            actual, expected = getattr(change, field), getattr(parameter, field)
            if actual != expected:
                raise ValueError(
                    f"attack delta for {name!r} has {field} {actual}, "
                    f"expected {expected}"
                )
        if change.requires_grad:
            raise ValueError(f"attack delta for {name!r} must be detached")


def dict_binary_op(dct1, dct2, binop):
    dct_out = dict()
    for key in dct1.keys():
        dct_out[key] = binop(dct1[key], dct2[key])
    return dct_out


class TARBaseTrainer:
    """Evaluate the forget loss after a simulated relearning attack.

    Mix in before an unlearning trainer whose ``compute_forget_loss`` and
    ``compute_retain_loss`` accept a model-like callable and return a loss.
    ``sample_attack_fn(model, parameters, inputs, *, on_loss_backward)`` returns a
    detached delta for the trainable parameters; the forget loss is taken at
    ``theta + delta``. Without one, ``adamw_attack`` is configured from
    the ``attack_*`` arguments.
    """

    def __init__(
        self,
        *args,
        sample_attack_fn=None,
        defence_forget_batch_size=1,
        defence_retain_batch_size=1,
        attack_forget_batch_size=1,
        attack_forget_batch_size_per_iteration=1,
        attack_epochs=1,
        attack_learning_rate,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.defence_forget_batch_size = defence_forget_batch_size
        self.defence_retain_batch_size = defence_retain_batch_size
        self.attack_forget_batch_size = attack_forget_batch_size
        self.attack_forget_batch_size_per_iteration = (
            attack_forget_batch_size_per_iteration
        )
        self.attack_epochs = attack_epochs
        self.attack_learning_rate = attack_learning_rate
        self.inner_step_total = 0
        self._inner_loss_metric_defined = False
        if sample_attack_fn is None:
            sample_attack_fn = partial(
                adamw_attack,
                attack_forget_batch_size_per_iteration=attack_forget_batch_size_per_iteration,
                attack_epochs=attack_epochs,
                attack_learning_rate=attack_learning_rate,
            )
        self.sample_attack_fn = sample_attack_fn

        self.metadata_to_idx = {
            "defence::forget": 0,
            "defence::retain": 1,
            "attack::forget": 2,
        }
        self.metadata_to_batch_size = {
            "defence::forget": self.defence_forget_batch_size,
            "defence::retain": self.defence_retain_batch_size,
            "attack::forget": self.attack_forget_batch_size,
        }

    def _get_train_sampler(self, dataset=None):
        dataset = self.train_dataset if dataset is None else dataset
        sampler = BalancedSourceSamplerv2(
            dataset,
            metadata_to_idx=self.metadata_to_idx,
            metadata_to_batch_size=self.metadata_to_batch_size,
        )
        if sampler.batch_size != self.args.per_device_train_batch_size:
            raise ValueError(
                "micro_batch_size must equal the sum of per-source batch sizes"
            )
        return sampler

    def _log_inner_loss(self, inner_step, loss):
        """Log attack losses on one curve against cumulative inner steps."""
        if (
            not self._inner_loss_metric_defined
            and wandb.run is not None
            and self.is_world_process_zero()
        ):
            wandb.define_metric(
                "train/inner_loss", step_metric="train/inner_step_total"
            )
            self._inner_loss_metric_defined = True
        self.log({"inner_loss": loss.item(), "inner_step_total": self.inner_step_total})
        self.inner_step_total += 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        defence_forget_mask = (
            inputs["metadata_ids"] == self.metadata_to_idx["defence::forget"]
        )
        defence_retain_mask = (
            inputs["metadata_ids"] == self.metadata_to_idx["defence::retain"]
        )
        attack_forget_mask = (
            inputs["metadata_ids"] == self.metadata_to_idx["attack::forget"]
        )
        inputs_defence_forget = {
            field: inputs[field][defence_forget_mask] for field in ROLE_FIELDS
        }
        inputs_defence_retain = {
            field: inputs[field][defence_retain_mask] for field in ROLE_FIELDS
        }
        inputs_attack_forget = {
            field: inputs[field][attack_forget_mask] for field in ROLE_FIELDS
        }
        theta = {name: param for name, param in model.named_parameters()}
        # make new model
        model_copy = copy.deepcopy(model)
        # deepcopy keeps accelerate's top-level forward wrapper bound to `model`
        model_copy.forward = MethodType(type(model_copy).forward, model_copy)
        with self.accelerator.autocast():  # the rebind skips accelerate's autocast
            model_attacked = self.sample_attack_fn(
                model_copy, inputs_attack_forget, on_loss_backward=self._log_inner_loss
            )
        theta_attacked = dict_binary_op(
            theta,
            dict(model_attacked.named_parameters()),
            lambda w, w_att: w + (w_att - w).detach(),
        )
        del model_copy, model_attacked
        gc.collect()  # accelerate hooks form reference cycles
        loss_retain = self.compute_retain_loss(model, inputs_defence_retain)
        loss_forget = self.compute_forget_loss(
            lambda **x: functional_call(model, theta_attacked, (), x),
            inputs_defence_forget,
        )
        loss = loss_forget + loss_retain

        if model.training and torch.is_grad_enabled():
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                }
            )
        return (loss, None) if return_outputs else loss


class TARGDTrainer(TARBaseTrainer, GradDiffTrainer):
    """TAR with gradient-difference losses."""


class TARMETrainer(TARBaseTrainer, MaxEntTrainer):
    """TAR with MaxEnt losses."""
