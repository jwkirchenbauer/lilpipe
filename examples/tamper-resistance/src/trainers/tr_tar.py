"""Tamper-resistant unlearning (TAR) with a pluggable simulated relearning attack."""

import copy
import gc
import json
import os
from collections import Counter
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


def log_device_assignment(model, phase, *, accelerator=None, inputs=None):
    """Report actual placement and check the configured layer map when debugging."""
    parameter_devices = Counter()
    for parameter in model.parameters():
        parameter_devices[str(parameter.device)] += parameter.numel()
    device_map = getattr(model, "hf_device_map", {})
    for name, target in device_map.items():
        expected = torch.device(f"cuda:{target}" if isinstance(target, int) else target)
        module = model.get_submodule(name) if name else model
        actual = {parameter.device for parameter in module.parameters()}
        if actual and actual != {expected}:
            raise RuntimeError(
                f"{phase}: module {name!r} is on {actual}, expected {expected}"
            )
    report = {
        "phase": phase,
        "pid": os.getpid(),
        "ranks": {
            key: os.environ.get(key)
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "SLURM_PROCID")
        },
        "visible_devices": torch.cuda.device_count(),
        "current_device": (
            torch.cuda.current_device() if torch.cuda.is_available() else None
        ),
        "parameter_counts_by_device": dict(parameter_devices),
        "hf_device_map": {key: str(value) for key, value in device_map.items()},
    }
    if accelerator is not None:
        report["accelerator"] = {
            "device": str(accelerator.device),
            "process_index": accelerator.process_index,
            "num_processes": accelerator.num_processes,
            "distributed_type": str(accelerator.distributed_type),
        }
    if inputs is not None:
        report["batch"] = {
            key: {"device": str(value.device), "shape": list(value.shape)}
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
    print("TAR device check: " + json.dumps(report, sort_keys=True), flush=True)


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
    on_step_end=None,
    attack_max_steps=None,
    attack_optimizer_mode="legacy",
    attack_learning_rates=None,
    attack_batches=None,
    attack_parameter_groups=None,
    on_inner_step=None,
):
    """Run AdamW on selected rows, calling on_step_end with the completed step count."""
    if attack_forget_batch_size_per_iteration < 1 or attack_epochs < 0:
        raise ValueError("attack batch size must be positive and epochs nonnegative")
    n_samples = len(inputs["input_ids"])
    if not n_samples:
        raise ValueError("TAR attack requires at least one attack row")
    if attack_max_steps is not None and attack_max_steps < 1:
        raise ValueError("attack_max_steps must be positive when provided")
    if attack_optimizer_mode not in {"legacy", "matched"}:
        raise ValueError("attack_optimizer_mode must be 'legacy' or 'matched'")
    batches = attack_batches
    if batches is None:
        batches = [
            {
                field: inputs[field][
                    start : start + attack_forget_batch_size_per_iteration
                ]
                for field in ROLE_FIELDS
            }
            for start in range(0, n_samples, attack_forget_batch_size_per_iteration)
        ]
    if attack_epochs < 0:
        raise ValueError("attack_epochs must be nonnegative")
    available_steps = len(batches) * attack_epochs
    step_limit = (
        available_steps
        if attack_max_steps is None
        else min(available_steps, attack_max_steps)
    )
    if attack_learning_rates is not None and len(attack_learning_rates) < step_limit:
        raise ValueError(
            "Not enough archived learning rates for requested attack steps"
        )
    optimizer_kwargs = {"lr": attack_learning_rate}
    if attack_optimizer_mode == "matched":
        optimizer_kwargs.update(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    if attack_parameter_groups is None:
        optimizer_parameters = model.parameters()
    else:
        named_parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        assigned = [name for group in attack_parameter_groups for name in group]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(
            named_parameters
        ):
            raise ValueError(
                "attack_parameter_groups must partition all trainable parameter names"
            )
        optimizer_parameters = [
            {"params": [named_parameters[name] for name in group]}
            for group in attack_parameter_groups
        ]
    optimizer = torch.optim.AdamW(optimizer_parameters, **optimizer_kwargs)
    inner_step = 0
    for _ in range(attack_epochs):
        for batch in batches:
            if inner_step >= step_limit:
                break
            learning_rate = (
                attack_learning_rates[inner_step]
                if attack_learning_rates is not None
                else attack_learning_rate
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            if on_loss_backward is not None:
                on_loss_backward(inner_step, loss.detach())
            grad_norm = None
            if attack_optimizer_mode == "matched":
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                )
            optimizer.step()
            if attack_optimizer_mode == "matched":
                # TAR's caller keeps one autocast context open across the loop.
                # Cached FP32 parameter casts otherwise retain pre-update values.
                # A genuine Trainer ends that context after every forward.
                torch.clear_autocast_cache()
            inner_step += 1
            if on_inner_step is not None:
                on_inner_step(
                    inner_step,
                    batch,
                    float(loss.detach()),
                    float(learning_rate),
                    grad_norm,
                )
            if on_step_end is not None:
                on_step_end(inner_step)
        if inner_step >= step_limit:
            break
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
        attack_max_steps=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._device_debug = os.environ.get("LILPIPE_DEVICE_DEBUG") == "1"
        self._device_debug_batch_logged = False
        if self._device_debug:
            log_device_assignment(
                self.model, "trainer_init", accelerator=self.accelerator
            )
        self.defence_forget_batch_size = defence_forget_batch_size
        self.defence_retain_batch_size = defence_retain_batch_size
        self.attack_forget_batch_size = attack_forget_batch_size
        self.attack_forget_batch_size_per_iteration = (
            attack_forget_batch_size_per_iteration
        )
        self.attack_epochs = attack_epochs
        self.attack_learning_rate = attack_learning_rate
        self.attack_max_steps = attack_max_steps
        self.inner_step_total = 0
        self._inner_loss_metric_defined = False
        if sample_attack_fn is None:
            sample_attack_fn = partial(
                adamw_attack,
                attack_forget_batch_size_per_iteration=(
                    attack_forget_batch_size_per_iteration
                ),
                attack_epochs=attack_epochs,
                attack_learning_rate=attack_learning_rate,
                attack_max_steps=attack_max_steps,
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
        if self._device_debug and self.inner_step_total == 0:
            print(
                f"TAR first inner backward completed; loss device: {loss.device}",
                flush=True,
            )
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
        debug_batch = self._device_debug and not self._device_debug_batch_logged
        if debug_batch:
            log_device_assignment(
                model, "first_batch", accelerator=self.accelerator, inputs=inputs
            )
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
        if debug_batch:
            log_device_assignment(
                model_copy, "copied_attack_model", inputs=inputs_attack_forget
            )
            self._device_debug_batch_logged = True
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
            self.log(self.loss_metrics(loss_forget, loss_retain))
        return (loss, None) if return_outputs else loss

    def loss_metrics(self, loss_forget, loss_retain):
        return {
            "loss_forget": loss_forget.detach().item(),
            "loss_retain": loss_retain.detach().item(),
        }


class TARGDTrainer(TARBaseTrainer, GradDiffTrainer):
    """TAR with gradient-difference losses."""


class TARMETrainer(TARBaseTrainer, MaxEntTrainer):
    """TAR with MaxEnt losses."""

    def loss_metrics(self, loss_forget, loss_retain):
        return {
            "loss_forget": -self._last_forget_entropy.item(),
            "loss_retain": self._last_retain_loss.item(),
        }
