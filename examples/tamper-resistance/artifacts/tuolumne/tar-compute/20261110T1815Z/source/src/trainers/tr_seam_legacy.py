"""Layer-sharded implementation of the paper's SEAM update."""

from __future__ import annotations

import contextlib
import math
import random
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from axolotl.core.trainers.base import AxolotlTrainer

from data.seam import BENIGN, HARMFUL, REFUSAL

# Utilities


def gradient_dict_dot(first, second, fp32=False, reduction_seed=None):
    """Dot two name-to-tensor mappings that may span several devices."""
    names = first.keys() & second.keys()
    if reduction_seed is not None:
        names = sorted(names)
        random.Random(reduction_seed).shuffle(names)
    partials = {}
    for name in names:
        device = first[name].device
        left = first[name].float() if fp32 else first[name]
        right = second[name].float() if fp32 else second[name]
        value = (left * right).sum()
        partials[device] = value if device not in partials else partials[device] + value
    if not partials:
        raise RuntimeError("gradient mappings have no common tensors")
    device = next(iter(partials))
    values = iter(partials.values())
    result = next(values).to(device)
    for value in values:
        result = result + value.to(device)
    return result


def use_bfloat16_parameters(model):
    """Undo Axolotl's embedding/norm upcast to match the authors' BF16 model."""
    model.bfloat16()
    if hasattr(model, "config"):
        model.config.dtype = torch.bfloat16


def named_gradients(loss, parameters, divisor=1):
    """Differentiate a scalar into a complete name-to-gradient mapping."""
    values = torch.autograd.grad(
        loss / divisor, tuple(parameters.values()), allow_unused=True
    )
    return {
        name: value.detach() if value is not None else torch.zeros_like(parameter)
        for (name, parameter), value in zip(parameters.items(), values, strict=True)
    }


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
#     loss_ul = accumulate_ul_grad(model, inputs)
#     loss_up = accumulate_up_grad(model, inputs)
#     accumulate_sd_grad(model, inputs)
#     return (loss_ul + loss_up).detach()
#
# def accumulate_up_grad(model, inputs):
#     loss_up = compute_up_loss(model, inputs)
#     loss_up.backward()  # accumulates directly into each sharded parameter.grad
#
# def accumulate_sd_grad(model, inputs):
#     g_hh = autograd.grad(compute_harmful_loss(model, inputs), parameters)
#     g_bb = autograd.grad(compute_benign_loss(model, inputs), parameters)
#     norm_hh = global_norm(g_hh)  # only scalar partials cross GPUs
#     norm_bb = global_norm(g_bb)
#     g_sd = finite_difference(g_hh, g_bb, norm_hh, norm_bb)
#     for name, parameter in model.named_parameters():
#         parameter.grad += g_sd[name]  # parameter and gradient share a GPU


def masked_cross_entropy(logits, labels, mask):
    """Authors' masked next-token CE, including masked-logit constants."""
    labels = labels[..., 1:].to(logits.device)
    valid = mask[..., 1:].to(logits.device).bool() & labels.ne(0) & labels.ne(-100)
    live = valid & mask[..., :-1].to(logits.device).bool()
    loss = torch.nn.functional.cross_entropy(
        logits[..., :-1, :][live], labels[live].long(), reduction="sum"
    )
    loss += (valid.sum() - live.sum()) * math.log(logits.shape[-1])
    return loss / valid.sum()


def hidden_cross_entropy(hidden, labels, mask, norm, head):
    """Masked CE for a sharded hidden state without dense vocabulary logits."""
    labels = labels[..., 1:].to(hidden.device)
    valid = mask[..., 1:].to(hidden.device).bool() & labels.ne(0) & labels.ne(-100)
    live = valid & mask[..., :-1].to(hidden.device).bool()
    selected = hidden[..., :-1, :][live].to(head.weight.device)
    logits = head(norm(selected).to(head.weight.dtype))
    loss = torch.nn.functional.cross_entropy(
        logits, labels[live].to(logits.device).long(), reduction="sum"
    )
    loss += (valid.sum() - live.sum()).to(logits.device) * math.log(logits.shape[-1])
    return loss / valid.sum().to(logits.device)


class BF16Trainer(AxolotlTrainer):
    """Axolotl trainer that keeps every floating model tensor in BF16."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        use_bfloat16_parameters(self.model)


class SEAMAlignmentTrainer(BF16Trainer):
    """Accumulate UL, UP, and finite-difference SD gradients in place."""

    def __init__(
        self,
        *args,
        alpha=1.0,
        beta=0.01,
        epsilon=1e-3,
        fp32_accumulation=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha, self.beta, self.epsilon = alpha, beta, epsilon
        self.fp32_accumulation = fp32_accumulation
        # Leave each hidden state on the GPU that owns its transformer layer.
        if hasattr(self.model, "_hf_hook"):
            self.model._hf_hook.io_same_device = False

    def _get_train_sampler(self, train_dataset=None):
        """Order each batch as paired harmful, refusal, and benign rows."""
        dataset = self.train_dataset if train_dataset is None else train_dataset
        batch_size = self._train_batch_size
        if batch_size < 3 or batch_size % 3:
            raise ValueError("SEAM sampling requires a batch size divisible by three")

        roles, sources = dataset["seam_role"], dataset["source_index"]
        harmful = {
            source: index
            for index, (role, source) in enumerate(zip(roles, sources, strict=True))
            if role == HARMFUL
        }
        refusal = {
            source: index
            for index, (role, source) in enumerate(zip(roles, sources, strict=True))
            if role == REFUSAL
        }
        benign = [index for index, role in enumerate(roles) if role == BENIGN]
        if harmful.keys() != refusal.keys() or not benign:
            raise ValueError("SEAM requires paired harmful/refusal and benign rows")

        role_batch_size = batch_size // 3
        role_size = min(len(harmful), len(benign)) // role_batch_size * role_batch_size
        harmful_sources = list(harmful)[:role_size]
        order = []
        for start in range(0, role_size, role_batch_size):
            selected = harmful_sources[start : start + role_batch_size]
            order.extend(harmful[source] for source in selected)
            order.extend(refusal[source] for source in selected)
            order.extend(benign[start : start + role_batch_size])
        return order

    @staticmethod
    def compute_ce(model, batch, mask=None, layerwise=False):
        """Compute ordinary, masked, or layer-wise masked next-token CE."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=layerwise,
                use_cache=False,
            )

            if mask is None:
                logits = outputs.logits[..., :-1, :].contiguous()
                labels = batch["labels"][..., 1:].to(logits.device).long()
                return torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.reshape(-1)
                )

            losses = [masked_cross_entropy(outputs.logits, batch["labels"], mask)]
            if not layerwise:
                return losses[0]

            norm, head = model.model.norm, model.get_output_embeddings()

            hidden_loss = partial(
                hidden_cross_entropy,
                labels=batch["labels"],
                mask=mask,
                norm=norm,
                head=head,
            )
            losses.extend(
                checkpoint(
                    hidden_loss,
                    hidden,
                    use_reentrant=True,
                    preserve_rng_state=False,
                )
                for hidden in outputs.hidden_states
            )
            return -torch.log(
                torch.stack(losses).sum() / len(outputs.hidden_states) + 1
            )

    def accumulate_ul_grad(self, model, harmful, mask):
        """Backward the layer-wise harmful unlearning term into parameter.grad."""
        divisor = self.current_gradient_accumulation_steps
        loss_ul = self.compute_ce(model, harmful, mask, layerwise=True)
        self.accelerator.backward(loss_ul / divisor)
        return loss_ul.detach()

    def accumulate_up_grad(self, model, refusal, mask):
        """Backward the refusal-preservation term into parameter.grad."""
        divisor = self.current_gradient_accumulation_steps
        loss_up = self.compute_ce(model, refusal, mask)
        self.accelerator.backward(self.alpha * loss_up / divisor)
        return loss_up.detach()

    def accumulate_sd_grad(self, model, harmful, benign, mask, compute_loss_only=False):
        """Accumulate beta-scaled SD gradients; return raw cosine and gradient norm."""
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        divisor = self.current_gradient_accumulation_steps
        dot = partial(
            gradient_dict_dot,
            fp32=self.fp32_accumulation,
            reduction_seed=self.args.seed if self.fp32_accumulation else None,
        )

        loss_hh = self.compute_ce(model, harmful, mask)
        g_hh = named_gradients(loss_hh, parameters, divisor)
        loss_bb = self.compute_ce(model, benign)
        g_bb = named_gradients(loss_bb, parameters, divisor)
        norm_hh = dot(g_hh, g_hh).sqrt()
        norm_bb = dot(g_bb, g_bb).sqrt()
        cosine = dot(g_hh, g_bb) / (norm_hh * norm_bb + 1e-8)
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
            loss_hh_shifted = self.compute_ce(model, harmful, mask)
            g_hh_shifted = named_gradients(loss_hh_shifted, parameters, divisor)
        with shifted_parameters(parameters, d_bb):
            loss_bb_shifted = self.compute_ce(model, benign)
            g_bb_shifted = named_gradients(loss_bb_shifted, parameters, divisor)

        g_sd = {
            name: (g_hh_shifted[name] - g_hh[name]) / norm_hh.to(parameter.device)
            + (g_bb_shifted[name] - g_bb[name]) / norm_bb.to(parameter.device)
            for name, parameter in parameters.items()
        }
        norm_sd = dot(g_sd, g_sd).sqrt() * abs(self.beta / self.epsilon)
        with torch.no_grad():
            for name, parameter in parameters.items():
                # parameter, parameter.grad, and g_sd[name] are on the same GPU.
                sd_grad = g_sd[name].mul(self.beta / self.epsilon)
                if parameter.grad is None:
                    parameter.grad = sd_grad
                else:
                    parameter.grad.add_(sd_grad)
        return cosine.detach(), norm_sd.detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        inputs = self._prepare_inputs(inputs)
        roles = inputs.pop("seam_role")
        source_indices = inputs.pop("source_index")

        harmful_rows = roles == HARMFUL
        refusal_rows = roles == REFUSAL
        benign_rows = roles == BENIGN
        harmful = {name: value[harmful_rows] for name, value in inputs.items()}
        refusal = {name: value[refusal_rows] for name, value in inputs.items()}
        benign = {name: value[benign_rows] for name, value in inputs.items()}
        harmful_sources = source_indices[roles == HARMFUL]
        refusal_sources = source_indices[roles == REFUSAL]
        if not torch.equal(harmful_sources, refusal_sources):
            raise ValueError("SEAM harmful/refusal rows are not paired")

        mask = harmful["input_ids"] != refusal["input_ids"]
        divisor = self.current_gradient_accumulation_steps

        loss_ul = self.accumulate_ul_grad(model, harmful, mask)
        loss_up = self.accumulate_up_grad(model, refusal, mask)
        loss_sd, norm_sd = self.accumulate_sd_grad(model, harmful, benign, mask)
        self.log(
            {
                "loss_ul": loss_ul.item(),
                "loss_up": (self.alpha * loss_up).item(),
                "loss_sd": loss_sd.item(),
                "norm_sd": norm_sd.item(),
            }
        )

        # Trainer uses this only for logging; all gradients are already accumulated.
        return ((loss_ul + self.alpha * loss_up.to(loss_ul.device)) / divisor).to(
            self.args.device
        )


class SEAMRelearnTrainer(SEAMAlignmentTrainer):
    """Optimize harmful CE while logging the legacy SD cosine."""

    def training_step(self, model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        inputs = self._prepare_inputs(inputs)
        roles = inputs.pop("seam_role")
        source_indices = inputs.pop("source_index")
        harmful = {name: value[roles == HARMFUL] for name, value in inputs.items()}
        refusal = {name: value[roles == REFUSAL] for name, value in inputs.items()}
        benign = {name: value[roles == BENIGN] for name, value in inputs.items()}
        if not torch.equal(
            source_indices[roles == HARMFUL], source_indices[roles == REFUSAL]
        ):
            raise ValueError("SEAM harmful/refusal rows are not paired")
        mask = harmful["input_ids"] != refusal["input_ids"]

        divisor = self.current_gradient_accumulation_steps
        loss_up = self.compute_ce(model, harmful, mask)
        loss_sd = self.accumulate_sd_grad(
            model, harmful, benign, mask, compute_loss_only=True
        )
        self.accelerator.backward(loss_up / divisor)
        self.log({"loss_up": loss_up.item(), "loss_sd": loss_sd.item()})
        return (loss_up.detach() / divisor).to(self.args.device)
