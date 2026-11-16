"""Layer-sharded implementation of the paper's SEAM update."""

from __future__ import annotations

import contextlib
from functools import partial

import torch
from torch.utils.data import Sampler
from torch.utils.checkpoint import checkpoint

from trainers.rl_bf16 import BF16Trainer

# Utilities


def model_to_params(model, only_trainable=False):
    """Return the model's parameters keyed by name, optionally only trainable ones."""
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not only_trainable or parameter.requires_grad
    }


def gradient_dict_dot(first, second):
    """Dot two name-to-tensor mappings that may span several devices."""
    partials = {}
    for name in first.keys() & second.keys():
        device = first[name].device
        value = (first[name] * second[name]).sum()
        partials[device] = value if device not in partials else partials[device] + value
    if not partials:
        raise RuntimeError("gradient mappings have no common tensors")
    device = next(iter(partials))
    values = iter(partials.values())
    result = next(values).to(device)
    for value in values:
        result = result + value.to(device)
    return result


def named_gradients(loss, parameters, divisor=1):
    """Differentiate a scalar into a complete name-to-gradient mapping."""
    values = torch.autograd.grad(
        loss / divisor, tuple(parameters.values()), allow_unused=True
    )
    return {
        name: value.detach() if value is not None else torch.zeros_like(parameter)
        for (name, parameter), value in zip(parameters.items(), values, strict=True)
    }


def accumulate_gradients(parameters, gradients, coefficient=1.0):
    """Add a name-to-gradient mapping into the corresponding parameters."""
    with torch.no_grad():
        for name, parameter in parameters.items():
            gradient = gradients[name].mul(coefficient)
            if parameter.grad is None:
                parameter.grad = gradient
            else:
                parameter.grad.add_(gradient)


@contextlib.contextmanager
def shifted_parameters(parameters, shift):
    """Temporarily add a name-to-tensor shift and restore the exact parameters."""
    originals = {
        name: parameter.detach().clone() for name, parameter in parameters.items()
    }
    try:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.add_(shift[name])
        yield
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(originals[name])


# The implementation below follows this update directly. The outer method must
# be training_step, rather than compute_loss, so Trainer does not backward twice.
#
# def training_step(model, inputs):
#     loss_hh = accumulate_hh_grad(model, inputs, coefficient=-coeff_hh)
#     loss_hb = accumulate_hb_grad(model, inputs, coefficient=+coeff_hb)
#     loss_sd = accumulate_sd_grad(model, inputs, coefficient=+coeff_sd)
#     return (-coeff_hh * loss_hh + coeff_hb * loss_hb + coeff_sd * loss_sd)
#
# def accumulate_hb_grad(model, inputs):
#     loss_hb = compute_hb_loss(model, inputs)
#     loss_hb.backward()  # accumulates directly into each sharded parameter.grad
#
# def accumulate_sd_grad(model, inputs):
#     g_hh = autograd.grad(compute_harmful_loss(model, inputs), parameters)
#     g_bb = autograd.grad(compute_benign_loss(model, inputs), parameters)
#     norm_hh = global_norm(g_hh)  # only scalar partials cross GPUs
#     norm_bb = global_norm(g_bb)
#     g_sd = finite_difference(g_hh, g_bb, norm_hh, norm_bb)
#     for name, parameter in model.named_parameters():
#         parameter.grad += g_sd[name]  # parameter and gradient share a GPU


def hidden_cross_entropy(hidden, labels, norm, head):
    """Completion-only CE for a hidden state without dense prompt logits."""
    labels = labels[..., 1:].to(hidden.device)
    live = labels.ne(-100)
    selected = hidden[..., :-1, :][live].to(head.weight.device)
    logits = head(norm(selected).to(head.weight.dtype))
    return torch.nn.functional.cross_entropy(
        logits, labels[live].to(logits.device).long()
    )


class BalancedSampler(Sampler):
    """Visit concatenated datasets one row from each stream at a time.

    ``BalancedSampler(2, 3)`` treats the concatenated index ranges as
    ``[0, 1]`` and ``[2, 3, 4]``. It yields ``0, 2, 1, 3``: one index from
    each dataset in turn, stopping when the shortest dataset is exhausted.
    """

    def __init__(self, *lengths):
        if not lengths:
            raise ValueError("BalancedSampler requires at least one dataset")
        self.lengths = lengths
        self.offsets = []
        offset = 0
        for length in lengths:
            self.offsets.append(offset)
            offset += length

    def __iter__(self):
        for index in range(min(self.lengths)):
            for offset in self.offsets:
                yield offset + index

    def __len__(self):
        return min(self.lengths) * len(self.lengths)


class SEAMTrainer(BF16Trainer):
    """Accumulate UL, UP, and finite-difference SD gradients in place."""

    def __init__(
        self,
        *args,
        coeff_hh=1.0,
        coeff_hb=1.0,
        coeff_sd=0.01,
        epsilon=1e-3,
        n_hh=8_000,
        n_hb=8_000,
        n_bb=8_000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # input validation
        assert all(
            c >= 0 for c in [coeff_hh, coeff_hb, coeff_sd]
        ), "Coefficients must be nonnegative"
        self.coeff_hh, self.coeff_hb, self.coeff_sd = coeff_hh, coeff_hb, coeff_sd
        self.epsilon = epsilon
        self.n_hh, self.n_hb, self.n_bb = n_hh, n_hb, n_bb
        # Leave each hidden state on the GPU that owns its transformer layer.
        if hasattr(self.model, "_hf_hook"):
            self.model._hf_hook.io_same_device = False

    def _get_train_sampler(self, train_dataset=None):
        """Draw equally from the contiguous HH, HB, and BB dataset streams."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if self._train_batch_size < 3 or self._train_batch_size % 3:
            raise ValueError("SEAM sampling requires a batch size divisible by three")
        if len(dataset) < self.n_hh + self.n_hb + self.n_bb:
            raise ValueError("SEAM dataset is shorter than its configured streams")
        return BalancedSampler(self.n_hh, self.n_hb, self.n_bb)

    @staticmethod
    def compute_ce(model, batch, layerwise=False):
        """Compute completion-only CE from the masks already stored in labels."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                **batch,
                output_hidden_states=layerwise,
                use_cache=False,
            )

            if not layerwise:
                return outputs.loss

            norm, head = model.model.norm, model.get_output_embeddings()
            hidden_loss = partial(
                hidden_cross_entropy,
                labels=batch["labels"],
                norm=norm,
                head=head,
            )
            losses = [outputs.loss]
            losses.extend(
                checkpoint(
                    hidden_loss,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                for hidden in outputs.hidden_states
            )
            return torch.log(torch.stack(losses).sum() / len(outputs.hidden_states) + 1)

    def accumulate_hh_grad(
        self, model, harmful, coefficient=1.0, compute_loss_only=False
    ):
        """Backward the layer-wise harmful unlearning term into parameter.grad."""
        divisor = self.current_gradient_accumulation_steps
        loss = self.compute_ce(model, harmful, layerwise=True)
        if compute_loss_only:
            return coefficient * loss.detach()

        # Compute the gradient without accumulating it into parameter.grad.
        parameters = model_to_params(model, only_trainable=True)
        g = named_gradients(loss, parameters, divisor)
        norm = gradient_dict_dot(g, g).sqrt() * abs(coefficient)

        # Accumulate the gradient for the eventual optimizer step.
        accumulate_gradients(parameters, g, coefficient)
        return coefficient * loss.detach(), norm.detach()

    def accumulate_hb_grad(
        self, model, refusal, coefficient=1.0, compute_loss_only=False
    ):
        """Backward the refusal-preservation term into parameter.grad."""
        divisor = self.current_gradient_accumulation_steps
        loss = self.compute_ce(model, refusal)
        if compute_loss_only:
            return coefficient * loss.detach()

        # Compute the gradient without accumulating it into parameter.grad.
        parameters = model_to_params(model, only_trainable=True)
        g = named_gradients(loss, parameters, divisor)
        norm = gradient_dict_dot(g, g).sqrt() * abs(coefficient)

        # Accumulate the gradient for the eventual optimizer step.
        accumulate_gradients(parameters, g, coefficient)
        return coefficient * loss.detach(), norm.detach()

    def accumulate_sd_grad(
        self, model, harmful, benign, coefficient=1.0, compute_loss_only=False
    ):
        """Accumulate weighted SD gradients; return raw cosine and weighted norm."""
        parameters = model_to_params(model, only_trainable=True)
        divisor = self.current_gradient_accumulation_steps

        loss_hh = self.compute_ce(model, harmful)
        g_hh = named_gradients(loss_hh, parameters, divisor)
        loss_bb = self.compute_ce(model, benign)
        g_bb = named_gradients(loss_bb, parameters, divisor)
        norm_hh = gradient_dict_dot(g_hh, g_hh).sqrt()
        norm_bb = gradient_dict_dot(g_bb, g_bb).sqrt()
        cosine = gradient_dict_dot(g_hh, g_bb) / (norm_hh * norm_bb + 1e-8)
        if compute_loss_only:
            return cosine.detach()

        d_hh = {
            name: self.epsilon
            * (
                g_bb[name] / norm_bb.to(parameter.device)
                - cosine.to(parameter.device)
                * g_hh[name]
                / norm_hh.to(parameter.device)
            )
            for name, parameter in parameters.items()
        }
        d_bb = {
            name: self.epsilon
            * (
                g_hh[name] / norm_hh.to(parameter.device)
                - cosine.to(parameter.device)
                * g_bb[name]
                / norm_bb.to(parameter.device)
            )
            for name, parameter in parameters.items()
        }
        with shifted_parameters(parameters, d_hh):
            loss_hh_shifted = self.compute_ce(model, harmful)
            g_hh_shifted = named_gradients(loss_hh_shifted, parameters, divisor)
        with shifted_parameters(parameters, d_bb):
            loss_bb_shifted = self.compute_ce(model, benign)
            g_bb_shifted = named_gradients(loss_bb_shifted, parameters, divisor)

        g_sd = {
            name: (g_hh_shifted[name] - g_hh[name]) / norm_hh.to(parameter.device)
            + (g_bb_shifted[name] - g_bb[name]) / norm_bb.to(parameter.device)
            for name, parameter in parameters.items()
        }
        norm_sd = gradient_dict_dot(g_sd, g_sd).sqrt() * abs(coefficient / self.epsilon)
        accumulate_gradients(parameters, g_sd, coefficient / self.epsilon)
        return cosine.detach(), norm_sd.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        inp_hh = {name: value[0::3] for name, value in inputs.items()}
        inp_hb = {name: value[1::3] for name, value in inputs.items()}
        inp_bb = {name: value[2::3] for name, value in inputs.items()}
        divisor = self.current_gradient_accumulation_steps

        configs = [
            {
                "name": "hh",
                "accum_fn": self.accumulate_hh_grad,
                "kwargs": {
                    "model": model,
                    "harmful": inp_hh,
                    "coefficient": -self.coeff_hh,
                },
            },
            {
                "name": "hb",
                "accum_fn": self.accumulate_hb_grad,
                "kwargs": {
                    "model": model,
                    "refusal": inp_hb,
                    "coefficient": self.coeff_hb,
                },
            },
            {
                "name": "sd",
                "accum_fn": self.accumulate_sd_grad,
                "kwargs": {
                    "model": model,
                    "harmful": inp_hh,
                    "benign": inp_bb,
                    "coefficient": self.coeff_sd,
                },
            },
        ]

        log_dict = {}
        losses = []
        for cfg in configs:
            loss, norm = cfg["accum_fn"](**cfg["kwargs"])
            log_dict[f"loss_{cfg['name']}"] = loss.item()
            log_dict[f"norm_{cfg['name']}"] = norm.item()
            losses.append(self.coeff_sd * loss if cfg["name"] == "sd" else loss)

        self.log(log_dict)

        # Trainer uses this only for logging; all gradients are already accumulated.
        loss = sum(value.to(losses[0].device) for value in losses)
        return (loss / divisor).to(self.args.device)


class SEAMRelearnTrainer(SEAMTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        inputs = self._prepare_inputs(inputs)
        inp_hh = {name: value[0::3] for name, value in inputs.items()}
        inp_bb = {name: value[2::3] for name, value in inputs.items()}

        divisor = self.current_gradient_accumulation_steps
        loss_hh = self.compute_ce(model, inp_hh)
        loss_sd = self.accumulate_sd_grad(model, inp_hh, inp_bb, compute_loss_only=True)
        self.accelerator.backward(+loss_hh / divisor)
        self.log(
            {
                "loss_hh": loss_hh.item(),
                "loss_sd": loss_sd.item(),
                "cosine_similarity": loss_sd.item(),
            }
        )
        return (loss_hh / divisor).to(self.args.device)
