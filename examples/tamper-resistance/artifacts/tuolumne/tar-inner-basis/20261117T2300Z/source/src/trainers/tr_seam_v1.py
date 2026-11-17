"""First SEAM implementation retained for comparison and debugging."""

# Target design for a cleaner, layer-sharded implementation:
#
# def training_step(model, inputs):
#     # The model's layers, parameters, gradients, and optimizer state are split
#     # across GPUs. Backward automatically accumulates each gradient beside its
#     # parameter, so these two terms need no explicit gradient dictionaries.
#     compute_ul_loss(model, inputs).backward()
#     (alpha * compute_up_loss(model, inputs)).backward()
#
#     # autograd.grad leaves parameter.grad unchanged. g_hh, g_bb, and the
#     # finite-difference result remain sharded like the model parameters. Only
#     # one scalar partial sum per GPU moves to compute global norms/cosines.
#     g_sd = compute_sd_grad(model, inputs)
#
#     with torch.no_grad():
#         for name, parameter in model.named_parameters():
#             if name in g_sd:
#                 # parameter, parameter.grad, and g_sd[name] share a device.
#                 parameter.grad.add_(beta * g_sd[name])
#
#     # All three gradient terms are already installed. Return a detached value
#     # for logging; the outer training loop must not backward through it again.
#     return reported_loss.detach()
#
# def compute_sd_grad(model, inputs):
#     parameters = trainable_named_parameters(model)
#     g_hh = named_grad(compute_harmful_loss(model, inputs), parameters)
#     g_bb = named_grad(compute_benign_loss(model, inputs), parameters)
#
#     # Sum squared norms and dot products locally, then reduce just one scalar
#     # per GPU. Broadcast the resulting norms/cosine back to each local shard.
#     norm_hh = global_norm(g_hh)
#     norm_bb = global_norm(g_bb)
#     cosine = global_dot(g_hh, g_bb) / (norm_hh * norm_bb + 1e-8)
#
#     hh_shift, bb_shift = seam_perturbations(
#         g_hh, g_bb, norm_hh, norm_bb, cosine, epsilon
#     )
#     shifted_hh = named_grad_at_shift(
#         compute_harmful_loss, model, inputs, hh_shift
#     )
#     shifted_bb = named_grad_at_shift(
#         compute_benign_loss, model, inputs, bb_shift
#     )
#     return seam_finite_difference(
#         g_hh, g_bb, shifted_hh, shifted_bb, norm_hh, norm_bb, epsilon
#     )

from __future__ import annotations

import contextlib
import gc
import math
import os
import time

import torch
from accelerate import dispatch_model
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import TrainerCallback

from axolotl.core.trainers.base import AxolotlTrainer

from trainers.ul_gd import GradDiffTrainer
from trainers.ul_me import MaxEntTrainer

HARMFUL = 0
REFUSAL = 1
BENIGN = 2


class SEAMRoleSampler(torch.utils.data.Sampler[int]):
    """Yield source-ordered batches matching the authors' three data loaders."""

    def __init__(self, dataset, batch_size):
        if batch_size < 3 or batch_size % 3:
            raise ValueError("SEAM sampling requires a batch size divisible by three")
        roles = dataset["seam_role"]
        sources = dataset["source_index"]
        paired = {HARMFUL: {}, REFUSAL: {}}
        benign = []
        for index, (role, source) in enumerate(zip(roles, sources, strict=True)):
            if role in paired:
                if source in paired[role]:
                    raise ValueError(f"duplicate SEAM source index: {source}")
                paired[role][source] = index
            elif role == BENIGN:
                benign.append(index)
            else:
                raise ValueError(f"unknown SEAM role: {role}")
        if paired[HARMFUL].keys() != paired[REFUSAL].keys():
            raise ValueError("SEAM harmful and refusal rows must have matching sources")
        if not paired[HARMFUL] or not benign:
            raise ValueError("SEAM sampling requires all three data roles")
        self.harmful = paired[HARMFUL]
        self.refusal = paired[REFUSAL]
        self.benign = benign
        self.role_batch_size = batch_size // 3

    @property
    def role_size(self):
        available = min(len(self.harmful), len(self.benign))
        return available // self.role_batch_size * self.role_batch_size

    def __iter__(self):
        sources = list(self.harmful)
        for start in range(0, self.role_size, self.role_batch_size):
            selected = sources[start : start + self.role_batch_size]
            yield from (self.harmful[source] for source in selected)
            yield from (self.refusal[source] for source in selected)
            yield from self.benign[start : start + self.role_batch_size]

    def __len__(self):
        return 3 * self.role_size


def _named_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _gradient_dot(first, second):
    names = first.keys() & second.keys()
    if not names:
        raise RuntimeError("SEAM losses produced no common parameter gradients")
    partials = {}
    for name in names:
        device = first[name].device
        if second[name].device != device:
            raise RuntimeError(f"SEAM gradient device mismatch for {name}")
        value = (first[name].float() * second[name].float()).sum()
        partials[device] = value if device not in partials else partials[device] + value
    reduction_device = next(iter(partials))
    return sum(
        (partial.to(reduction_device) for partial in partials.values()),
        torch.zeros((), device=reduction_device),
    )


def _gradient_norm(gradients):
    return _gradient_dot(gradients, gradients).sqrt()


def _gradient_cosine(first, second, first_norm, second_norm):
    dot = _gradient_dot(first, second)
    denominator = (first_norm * second_norm).to(dot.device) + 1e-8
    return dot / denominator


@contextlib.contextmanager
def _temporary_parameter_update(
    parameters,
    first,
    second,
    first_norm,
    second_norm,
    cosine,
    epsilon,
):
    originals = {
        name: parameters[name].detach().clone() for name in first if name in parameters
    }
    try:
        with torch.no_grad():
            for name in first:
                device = parameters[name].device
                step = epsilon * (
                    first[name] / first_norm.to(first[name].device)
                    - cosine.to(second[name].device)
                    * second[name]
                    / second_norm.to(second[name].device)
                )
                parameters[name].add_(step.to(device))
        yield
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                parameters[name].copy_(original.to(parameters[name].device))


def _layer_device_map(model, devices):
    """Assign each complete Llama block to one of the requested devices."""
    decoder = getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if decoder is None or layers is None or not layers:
        raise TypeError("SEAM layer sharding requires a Llama-style model.layers")
    if len(layers) < len(devices):
        raise ValueError("SEAM layer sharding requires at least one layer per GPU")

    device_map = {"model.embed_tokens": devices[0]}
    for index in range(len(layers)):
        shard = min(index * len(devices) // len(layers), len(devices) - 1)
        device_map[f"model.layers.{index}"] = devices[shard]
    device_map["model.norm"] = devices[-1]
    device_map["lm_head"] = devices[-1]
    return device_map


class _LayerShardedModel(nn.Module):
    """Keep Trainer from applying DataParallel to an already-dispatched model."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def masked_token_cross_entropy(logits, labels, token_mask):
    """Reproduce the released implementation's masked next-token CE exactly."""
    shifted_labels = labels[..., 1:].to(device=logits.device)
    valid = token_mask[..., 1:].to(logits.device).bool().clone()
    valid &= shifted_labels.ne(0) & shifted_labels.ne(-100)
    live = valid & token_mask[..., :-1].to(logits.device).bool()
    selected_logits = logits[..., :-1, :][live]
    loss_sum = torch.nn.functional.cross_entropy(
        selected_logits,
        shifted_labels[live].long(),
        reduction="sum",
    )
    constant_count = valid.sum() - live.sum()
    loss_sum = loss_sum + constant_count * math.log(logits.shape[-1])
    return loss_sum / valid.sum()


def masked_hidden_cross_entropy(hidden, norm, head, labels, token_mask):
    """Equivalent masked CE without materializing logits for ignored tokens."""
    shifted_labels = labels[..., 1:].to(device=hidden.device)
    valid = token_mask[..., 1:].to(hidden.device).bool().clone()
    valid &= shifted_labels.ne(0) & shifted_labels.ne(-100)
    live = valid & token_mask[..., :-1].to(hidden.device).bool()
    # Normalization is token-wise, so select locally before transferring the
    # live hidden states to the final GPU that owns the norm and vocabulary head.
    selected = hidden[..., :-1, :][live].to(
        device=head.weight.device,
        dtype=norm.weight.dtype,
    )
    logits = head(norm(selected).to(dtype=head.weight.dtype))
    loss_sum = torch.nn.functional.cross_entropy(
        logits,
        shifted_labels[live].long(),
        reduction="sum",
    )
    constant_count = valid.sum() - live.sum()
    loss_sum = loss_sum + constant_count * math.log(head.weight.shape[0])
    return loss_sum / valid.sum()


def token_cross_entropy(logits, labels):
    """Released implementation's unmasked next-token cross entropy."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].to(dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.reshape(-1)
    )


class _SegmentCallback(TrainerCallback):
    """Stop after a fixed number of resumed steps and force a full checkpoint."""

    def __init__(self, steps):
        self.steps = steps
        self.target = None

    def on_train_begin(self, args, state, control, **kwargs):
        del args, control, kwargs
        self.target = min(state.global_step + self.steps, state.max_steps)

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if self.target is not None and state.global_step >= self.target:
            control.should_save = True
            control.should_training_stop = True
        return control


class SEAMAlignmentTrainer(AxolotlTrainer):
    """Full three-term SEAM defense used for the paper's BeaverTails setting."""

    def __init__(self, *args, alpha=1.0, beta=0.01, epsilon=1e-3, **kwargs):
        if alpha < 0 or beta < 0 or epsilon <= 0:
            raise ValueError("SEAM requires alpha >= 0, beta >= 0, and epsilon > 0")
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        segment_steps = int(os.environ.get("SEAM_SEGMENT_STEPS", "0"))
        if segment_steps:
            self.add_callback(_SegmentCallback(segment_steps))

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        return SEAMRoleSampler(dataset, self._train_batch_size)

    def _wrap_model(self, model, training=True, dataloader=None):
        """Place complete transformer blocks across four GPUs."""
        del training, dataloader
        if isinstance(model, _LayerShardedModel):
            return model
        if torch.cuda.device_count() < 4:
            raise RuntimeError(
                "paper-faithful SEAM training requires four visible GPUs"
            )
        devices = tuple(torch.device(f"cuda:{index}") for index in range(4))
        dispatch_model(
            model,
            device_map=_layer_device_map(model, devices),
            main_device=devices[0],
        )
        # Accelerate's root hook normally returns every output to the input GPU.
        # Hidden states must remain on their owning layer devices instead.
        if hasattr(model, "_hf_hook"):
            model._hf_hook.io_same_device = False
        model.model_parallel = True
        model.is_parallelizable = True
        return _LayerShardedModel(model)

    @staticmethod
    def split_roles(inputs):
        role = inputs["seam_role"]
        fields = ("input_ids", "attention_mask", "labels", "source_index")

        def select(value):
            mask = role == value
            if not mask.any():
                raise ValueError("SEAM requires all three roles in every batch")
            return {field: inputs[field][mask] for field in fields}

        harmful, refusal, benign = select(HARMFUL), select(REFUSAL), select(BENIGN)
        harmful_order = harmful["source_index"].argsort()
        refusal_order = refusal["source_index"].argsort()
        harmful = {key: value[harmful_order] for key, value in harmful.items()}
        refusal = {key: value[refusal_order] for key, value in refusal.items()}
        if not torch.equal(harmful["source_index"], refusal["source_index"]):
            raise ValueError("SEAM harmful/refusal rows are not paired")
        keep = ("input_ids", "attention_mask", "labels")
        return tuple(
            {key: batch[key] for key in keep} for batch in (harmful, refusal, benign)
        )

    @staticmethod
    def _output_head(model):
        unwrapped = model.module if isinstance(model, _LayerShardedModel) else model
        decoder = getattr(unwrapped, "model", getattr(unwrapped, "base_model", None))
        norm = getattr(decoder, "norm", None)
        head = unwrapped.get_output_embeddings()
        if norm is None or head is None:
            raise TypeError(
                "SEAM layer-wise loss requires a Llama-style norm and LM head"
            )
        return norm, head

    def adversarial_unlearning_loss(self, model, harmful, difference_mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=harmful["input_ids"],
                attention_mask=harmful["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            losses = [
                masked_token_cross_entropy(
                    outputs.logits, harmful["labels"], difference_mask
                )
            ]
            norm, head = self._output_head(model)
            for hidden in outputs.hidden_states:
                losses.append(
                    checkpoint(
                        lambda state: masked_hidden_cross_entropy(
                            state,
                            norm,
                            head,
                            harmful["labels"],
                            difference_mask,
                        ),
                        hidden,
                        use_reentrant=True,
                        preserve_rng_state=False,
                    )
                )
        # Preserve the released implementation, including its `+ 1` placement.
        return -torch.log(torch.stack(losses).sum() / len(outputs.hidden_states) + 1)

    @staticmethod
    def refusal_loss(model, refusal, difference_mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=refusal["input_ids"],
                attention_mask=refusal["attention_mask"],
                use_cache=False,
            )
        return masked_token_cross_entropy(
            outputs.logits, refusal["labels"], difference_mask
        )

    @staticmethod
    def attack_loss(model, harmful, difference_mask):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=harmful["input_ids"],
                attention_mask=harmful["attention_mask"],
                use_cache=False,
            )
        return masked_token_cross_entropy(
            outputs.logits, harmful["labels"], difference_mask
        )

    @staticmethod
    def benign_loss(model, benign):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=benign["input_ids"],
                attention_mask=benign["attention_mask"],
                use_cache=False,
            )
        return token_cross_entropy(outputs.logits, benign["labels"])

    def _named_gradients(self, loss, parameters):
        """Differentiate without changing the UL/UP gradients in parameter.grad."""
        values = tuple(parameters.values())
        gradients = torch.autograd.grad(
            loss / self.current_gradient_accumulation_steps,
            values,
            allow_unused=True,
        )
        return {
            name: (
                gradient.detach()
                if gradient is not None
                else torch.zeros_like(parameter)
            )
            for (name, parameter), gradient in zip(
                parameters.items(), gradients, strict=True
            )
        }

    @staticmethod
    def _accumulate_gradients(parameters, gradients, scale):
        """Add a named, layer-local gradient collection into parameter.grad."""
        with torch.no_grad():
            for name, gradient in gradients.items():
                parameter = parameters[name]
                contribution = gradient.mul(scale)
                if parameter.grad is None:
                    parameter.grad = contribution
                else:
                    parameter.grad.add_(contribution)

    def training_step(self, model, inputs, num_items_in_batch=None):
        del num_items_in_batch
        with self._layer_offload_ctx, self.activation_offload_context:
            return self._seam_step(model, self._prepare_inputs(inputs))

    def _seam_step(self, model, inputs):
        profile = os.environ.get("SEAM_PROFILE") == "1"
        profile_times = {}

        if profile:
            before_sync = time.perf_counter()
            torch.cuda.synchronize()
            profile_times["between_steps"] = time.perf_counter() - before_sync

        def mark(name):
            if profile:
                torch.cuda.synchronize()
                now = time.perf_counter()
                profile_times[name] = now - mark.previous
                mark.previous = now

        mark.previous = time.perf_counter()
        model.train()
        harmful, refusal, benign = self.split_roles(inputs)
        difference_mask = harmful["input_ids"] != refusal["input_ids"]
        if not difference_mask.any():
            raise ValueError("SEAM harmful/refusal pairs have no differing tokens")
        parameters = _named_parameters(model)
        with self.compute_loss_context_manager():
            loss_ul = self.adversarial_unlearning_loss(model, harmful, difference_mask)
        mark("unlearning_forward")
        reported_loss_ul = loss_ul.detach()
        self.accelerator.backward(loss_ul / self.current_gradient_accumulation_steps)
        del loss_ul
        mark("unlearning_backward")

        with self.compute_loss_context_manager():
            loss_up = self.refusal_loss(model, refusal, difference_mask)
        reported_loss_up = loss_up.detach()
        self.accelerator.backward(
            self.alpha * loss_up / self.current_gradient_accumulation_steps
        )
        del loss_up
        mark("refusal")

        with self.compute_loss_context_manager():
            loss_attack = self.attack_loss(model, harmful, difference_mask)
        grad_attack = self._named_gradients(loss_attack, parameters)
        del loss_attack
        mark("attack")

        with self.compute_loss_context_manager():
            loss_benign = self.benign_loss(model, benign)
        grad_benign = self._named_gradients(loss_benign, parameters)
        del loss_benign
        mark("benign")

        attack_norm = _gradient_norm(grad_attack)
        benign_norm = _gradient_norm(grad_benign)
        if not all(
            math.isfinite(value.item()) and value.item() > 0
            for value in (attack_norm, benign_norm)
        ):
            raise RuntimeError(
                "SEAM attack and benign gradients must have finite norms"
            )
        cosine = _gradient_cosine(grad_attack, grad_benign, attack_norm, benign_norm)
        mark("geometry")

        scale = self.beta / self.epsilon
        with _temporary_parameter_update(
            parameters,
            grad_benign,
            grad_attack,
            benign_norm,
            attack_norm,
            cosine,
            self.epsilon,
        ):
            with self.compute_loss_context_manager():
                shifted_attack_loss = self.attack_loss(model, harmful, difference_mask)
            shifted_attack = self._named_gradients(shifted_attack_loss, parameters)
            del shifted_attack_loss
        attack_difference = {
            name: (shifted_attack[name] - grad_attack[name])
            / attack_norm.to(grad_attack[name].device)
            for name in parameters
        }
        self._accumulate_gradients(parameters, attack_difference, scale)
        del shifted_attack, attack_difference
        mark("shifted_attack")

        with _temporary_parameter_update(
            parameters,
            grad_attack,
            grad_benign,
            attack_norm,
            benign_norm,
            cosine,
            self.epsilon,
        ):
            with self.compute_loss_context_manager():
                shifted_benign_loss = self.benign_loss(model, benign)
            shifted_benign = self._named_gradients(shifted_benign_loss, parameters)
            del shifted_benign_loss
        benign_difference = {
            name: (shifted_benign[name] - grad_benign[name])
            / benign_norm.to(grad_benign[name].device)
            for name in parameters
        }
        self._accumulate_gradients(parameters, benign_difference, scale)
        del shifted_benign, benign_difference, grad_attack, grad_benign
        mark("install")
        if profile:
            print(
                "SEAM_PROFILE "
                + " ".join(
                    f"{name}={duration:.3f}" for name, duration in profile_times.items()
                ),
                flush=True,
            )
        reported_loss = (
            (reported_loss_ul + self.alpha * reported_loss_up)
            / self.current_gradient_accumulation_steps
        ).detach()
        # Drop the loss and gradient collections before the next DataParallel
        # replication. Tensor reference counting reclaims these allocations.
        del (
            reported_loss_ul,
            reported_loss_up,
        )
        # The next layer-wise forward peaks near 77 GiB. Release inactive blocks
        # after every completed step so Torch 2.9's allocator can satisfy its
        # 250 MiB vocabulary-logit allocations without fragmentation.
        gc.collect()
        torch.cuda.empty_cache()
        if profile:
            print(
                "SEAM_MEMORY "
                + " ".join(
                    f"cuda:{device}={torch.cuda.memory_allocated(device) / 2**30:.3f}"
                    for device in range(4)
                ),
                flush=True,
            )
        return reported_loss


def grad_old(compute_loss, model, inputs, shift=None):
    """Gradient of ``compute_loss`` by parameter name, at the weights plus ``shift``."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    theta = {n: p.data for n, p in params.items()}
    for n, s in (shift or {}).items():
        params[n].data = params[n].data + s
    g = torch.autograd.grad(compute_loss(model, inputs), list(params.values()))
    for n, p in params.items():
        p.data = theta[n]
    return dict(zip(params, g))


def grad(compute_loss, model, inputs, shift=None):
    """Gradient of ``compute_loss`` by parameter name, at the weights plus ``shift``."""
    theta = {name: param.data for name, param in model.named_parameters()}
    # theta <- theta + delta
    for name, param in model.named_parameters():
        if shift is not None and name in shift:
            param.data = param.data + shift[name]

    loss = compute_loss(model, inputs)
    loss.backward()
    g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    for name, param in model.named_parameters():
        param.data = theta[name]
    return g


def dot(a, b):
    """Dot product of two name-to-tensor dicts that may span several devices."""
    device = next(iter(a.values())).device
    return sum(torch.sum(a[n].float() * b[n].float()).to(device) for n in a)


class SEAMBaseTrainer:
    """Add a self-destructive gradient-coupling term to an unlearning trainer.

    Mix in before an unlearning trainer whose ``compute_forget_loss`` and
    ``compute_retain_loss`` accept a model-like callable and return a loss. The
    coupling term is the cosine similarity between the forget and retain
    gradients; minimizing it makes relearning the forget set degrade utility.
    """

    def __init__(self, *args, sd_coefficient=1.0, sd_epsilon=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.sd_coefficient = sd_coefficient
        self.sd_epsilon = sd_epsilon

    def compute_attack_loss(self, model, inputs):
        """Cross-entropy on the forget rows, as minimized by a relearning attack."""
        forget_mask = inputs["is_forget"].bool()
        fields = ("input_ids", "attention_mask", "labels")
        forget_inputs = {field: inputs[field][forget_mask] for field in fields}
        return self.compute_loss_term(model, forget_inputs, 1.0)

    def compute_sd_loss(self, model, inputs, epsilon=1e-3):
        """Cosine similarity between forget and retain gradients.

        TODO: this differentiates through both gradients (double backward). SEAM
        uses a Hessian-free finite-difference estimate instead; swap it in here.
        """
        theta = {n: p for n, p in model.named_parameters() if p.requires_grad}

        # Gradients at current weights
        g_f = grad(self.compute_attack_loss, model, inputs)  # attacker minimizes -l_f
        g_r = grad(self.compute_retain_loss, model, inputs)

        # Normalize
        norm_f, norm_r = dot(g_f, g_f).sqrt().item(), dot(g_r, g_r).sqrt().item()
        c = dot(g_f, g_r) / (norm_f * norm_r)
        tilde_g_f = {n: g / norm_f for n, g in g_f.items()}
        tilde_g_r = {n: g / norm_r for n, g in g_r.items()}

        # Perturbation directions
        d_f = {n: epsilon * (tilde_g_r[n] - c.item() * tilde_g_f[n]) for n in g_f}
        d_r = {n: epsilon * (tilde_g_f[n] - c.item() * tilde_g_r[n]) for n in g_f}

        # Gradients at temporary perturbed weights; same batches
        g_f_shifted = grad(self.compute_attack_loss, model, inputs, d_f)
        g_r_shifted = grad(self.compute_retain_loss, model, inputs, d_r)

        # Approximate gradient of cos(g_f, g_r)
        g_sd = {
            n: (g_f_shifted[n] - g_f[n]) / (epsilon * norm_f)
            + (g_r_shifted[n] - g_r[n]) / (epsilon * norm_r)
            for n in g_f
        }

        # 2. Inject that estimate through a surrogate loss
        # its value is c, its gradient is g_sd
        loss_sd_surrogate = dot(theta, g_sd)
        loss_sd_surrogate = loss_sd_surrogate - loss_sd_surrogate.detach() + c

        return loss_sd_surrogate

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss_forget = self.compute_forget_loss(model, inputs)
        loss_retain = self.compute_retain_loss(model, inputs)
        loss_sd = self.compute_sd_loss(model, inputs, self.sd_epsilon)
        loss_sd = loss_sd.to(loss_forget.device)
        loss = loss_forget + loss_retain + self.sd_coefficient * loss_sd

        if model.training and torch.is_grad_enabled():
            self.log(
                {
                    "loss_forget": loss_forget.detach().item(),
                    "loss_retain": loss_retain.detach().item(),
                    "loss_coupling": loss_sd.detach().item(),
                }
            )
        return (loss, None) if return_outputs else loss


class SEAMGDTrainer(SEAMBaseTrainer, GradDiffTrainer):
    """SEAM with gradient-difference losses."""


class SEAMMETrainer(SEAMBaseTrainer, MaxEntTrainer):
    """SEAM with MaxEnt losses."""
