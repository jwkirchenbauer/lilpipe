"""Observe baseline TAR training without changing its optimization steps."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from axolotl.core.trainers.base import AxolotlTrainer
from transformers import TrainerCallback

from trainers.tr_tar import ROLE_FIELDS, TARMETrainer, adamw_attack


def load_attack_trace(path, expected_steps=16):
    """Load and validate the first ordered single-row minibatches of an attack trace."""
    batches, learning_rates, hashes = [], [], []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            step = record.get("step")
            if step != len(batches) + 1:
                raise ValueError("Attack trace steps must start at 1 and be contiguous")
            raw = record.get("batch")
            if not isinstance(raw, dict) or set(ROLE_FIELDS) - set(raw):
                raise ValueError(
                    f"Attack trace step {step} lacks required model fields"
                )
            batch = {field: torch.as_tensor(raw[field]) for field in ROLE_FIELDS}
            if any(value.ndim != 2 or value.shape[0] != 1 for value in batch.values()):
                raise ValueError("Replay trace must contain single-row minibatches")
            serial = {field: batch[field].tolist() for field in ROLE_FIELDS}
            digest = hashlib.sha256(
                json.dumps(serial, sort_keys=True).encode()
            ).hexdigest()
            archived_digest = record.get("batch_sha256")
            if archived_digest is not None and archived_digest != digest:
                raise ValueError(f"Attack trace batch hash mismatch at step {step}")
            rate = record.get("learning_rate_used")
            if not isinstance(rate, (float, int)) or rate < 0:
                raise ValueError(f"Invalid archived learning rate at step {step}")
            batches.append(batch)
            learning_rates.append(float(rate))
            hashes.append(digest)
            if len(batches) == expected_steps:
                break
    if len(batches) != expected_steps:
        raise ValueError(
            f"Attack trace has {len(batches)} usable rows; expected {expected_steps}"
        )
    return batches, learning_rates, hashes


def order_attack_trace(trace, order="forward"):
    """Order the same frozen minibatch pool without changing its LR schedule."""
    if order not in {"forward", "reverse"}:
        raise ValueError("order must be 'forward' or 'reverse'")
    batches, learning_rates, hashes = trace
    indices = list(range(len(batches)))
    if order == "reverse":
        indices.reverse()
    return (
        [batches[index] for index in indices],
        list(learning_rates),
        [hashes[index] for index in indices],
    )


def load_attack_schedule(path, expected_steps=1000):
    """Load the full archived LR schedule without retaining training tensors."""
    rates = []
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("step") != len(rates) + 1:
                raise ValueError(
                    "Attack schedule steps must start at 1 and be contiguous"
                )
            rate = record.get("learning_rate_used")
            if not isinstance(rate, (float, int)) or rate < 0:
                raise ValueError(
                    f"Invalid archived learning rate at step {len(rates) + 1}"
                )
            rates.append(float(rate))
    if len(rates) != expected_steps:
        raise ValueError(
            f"Attack schedule has {len(rates)} rows; expected {expected_steps}"
        )
    return rates


def select_attack_dataset(dataset, manifest_path, bootstrap_seed=None):
    """Trace prepared-row identities and optionally bootstrap without changing caches."""
    import numpy as np
    from collections import Counter

    population = len(dataset)
    if not population:
        raise ValueError("Cannot bootstrap an empty prepared dataset")
    fingerprint = getattr(dataset, "_fingerprint", None)
    indices = (
        np.random.default_rng(bootstrap_seed)
        .integers(0, population, size=population)
        .tolist()
        if bootstrap_seed is not None
        else list(range(population))
    )
    selected = dataset.select(indices) if bootstrap_seed is not None else dataset
    lookup = {}
    for source, row in zip(indices, selected):
        batch = {
            key: [torch.as_tensor(row[key]).tolist()]
            for key in ("input_ids", "attention_mask", "labels")
        }
        signature = hashlib.sha256(
            json.dumps(batch, sort_keys=True).encode()
        ).hexdigest()
        lookup.setdefault(signature, set()).add(source)
    lookup = {key: sorted(value) for key, value in lookup.items()}
    manifest = {
        "identity": "row position in the common prepared dataset before bootstrap",
        "prepared_dataset_fingerprint": fingerprint,
        "population_size": population,
        "draw_size": len(indices),
        "bootstrap_seed": bootstrap_seed,
        "unique_source_rows": len(set(indices)),
        "drawn_source_rows": indices,
        "source_multiplicities": dict(Counter(indices)),
        "batch_sources": lookup,
    }
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    temporary.replace(target)
    return selected, lookup


class DynamicsCallback(TrainerCallback):
    def __init__(self, trainer, save_steps):
        self.trainer = trainer
        self.save_steps = set(save_steps)
        state_steps = os.environ.get("DYNAMICS_STATE_STEPS")
        self.state_steps = (
            {int(value) for value in state_steps.split(",") if value}
            if state_steps is not None
            else None
        )
        if self.state_steps is not None and not self.state_steps <= self.save_steps:
            raise ValueError("Optimizer-state steps must also be model-save steps")

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        record = trainer._dynamics_record
        assert record["step"] == state.global_step
        record["elapsed_seconds"] = time.perf_counter() - trainer._dynamics_started
        if trainer._measure_after:
            model = trainer.model
            training = model.training
            batch = trainer._prepare_inputs(trainer._dynamics_batch)
            try:
                model.eval()
                with torch.no_grad(), trainer.accelerator.autocast():
                    record["loss_after_update"] = float(model(**batch).loss)
            finally:
                model.train(training)
            assert math.isfinite(record["loss_after_update"])
        path = Path(args.output_dir) / "observed-batches.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        trainer._dynamics_record = None
        trainer._dynamics_batch = None
        if state.global_step in self.save_steps:
            if self.state_steps is not None:
                args.save_only_model = state.global_step not in self.state_steps
            control.should_save = True
        return control


class RecordingMixin:
    _measure_after = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.gradient_accumulation_steps != 1:
            raise ValueError("This observation run requires accumulation=1")
        self._dynamics_record = None
        self._dynamics_batch = None
        self._dynamics_started = time.perf_counter()
        self._source_lookup = None
        manifest = os.environ.get("DYNAMICS_SOURCE_MANIFEST")
        if manifest:
            seed = os.environ.get("DYNAMICS_BOOTSTRAP_SEED")
            self.train_dataset, self._source_lookup = select_attack_dataset(
                self.train_dataset, manifest, None if seed is None else int(seed)
            )
        steps = [int(value) for value in os.environ["DYNAMICS_SAVE_STEPS"].split(",")]
        self.add_callback(DynamicsCallback(self, steps))

    def training_step(self, model, inputs, *args, **kwargs):
        started = time.perf_counter()
        batch = {
            key: value.detach().cpu().clone()
            for key, value in inputs.items()
            if key in ("input_ids", "attention_mask", "labels", "metadata_ids")
        }
        serial = {key: value.tolist() for key, value in batch.items()}
        record = {
            "step": self.state.global_step + 1,
            "learning_rate_used": self.optimizer.param_groups[0]["lr"],
            "batch": serial,
            "batch_sha256": hashlib.sha256(
                json.dumps(serial, sort_keys=True).encode()
            ).hexdigest(),
        }
        loss = super().training_step(model, inputs, *args, **kwargs)
        record["loss_before_update"] = float(loss.detach())
        record["forward_backward_seconds"] = time.perf_counter() - started
        if self._source_lookup is not None:
            source = self._source_lookup.get(record["batch_sha256"])
            if source is None:
                raise ValueError("Observed batch does not match its source manifest")
            record["source_rows"] = source
        assert math.isfinite(record["loss_before_update"])
        self._dynamics_record = record
        self._dynamics_batch = batch
        return loss


class RecordedTARMETrainer(RecordingMixin, TARMETrainer):
    """Keep the selected TAR objective and record its actual mixed batches."""


class RecordedReplayTARMETrainer(RecordedTARMETrainer):
    """Run bounded TAR inner attacks with optional frozen attack-batch replay."""

    def __init__(
        self,
        *args,
        attack_trace_path=None,
        attack_schedule_trace_path=None,
        attack_schedule_steps=1000,
        attack_trace_order="forward",
        attack_optimizer_mode="legacy",
        attack_max_steps=16,
        **kwargs,
    ):
        if attack_trace_order not in {"forward", "reverse"}:
            raise ValueError("attack_trace_order must be 'forward' or 'reverse'")
        if attack_optimizer_mode not in {"legacy", "matched"}:
            raise ValueError("attack_optimizer_mode must be 'legacy' or 'matched'")
        schedule_path = attack_schedule_trace_path
        if attack_optimizer_mode == "matched" and schedule_path is None:
            schedule_path = attack_trace_path
        if attack_optimizer_mode == "matched" and schedule_path is None:
            raise ValueError("matched optimizer mode requires an archived LR trace")
        if attack_max_steps < 1:
            raise ValueError("attack_max_steps must be positive")
        self.attack_trace_path = attack_trace_path
        self.attack_schedule_trace_path = schedule_path
        self.attack_schedule_steps = attack_schedule_steps
        self.attack_trace_order = attack_trace_order
        self.attack_optimizer_mode = attack_optimizer_mode
        self.attack_max_steps = attack_max_steps
        self._attack_trace_batches = None
        self._attack_trace_rates = None
        self._attack_trace_hashes = None
        self._attack_trace_sha256 = None
        self._attack_schedule_sha256 = None
        self._attack_schedule_rates = None
        if attack_trace_path is not None:
            self._attack_trace_sha256 = hashlib.sha256(
                Path(attack_trace_path).read_bytes()
            ).hexdigest()
            (
                self._attack_trace_batches,
                self._attack_trace_rates,
                self._attack_trace_hashes,
            ) = load_attack_trace(attack_trace_path)
        if schedule_path is not None:
            self._attack_schedule_sha256 = hashlib.sha256(
                Path(schedule_path).read_bytes()
            ).hexdigest()
            self._attack_schedule_rates = load_attack_schedule(
                schedule_path, expected_steps=attack_schedule_steps
            )
        super().__init__(*args, attack_max_steps=attack_max_steps, **kwargs)
        self.sample_attack_fn = self._run_replay_attack

    def _optimizer_parameter_groups(self, model):
        """Mirror Axolotl/Transformers decay and no-decay parameter groups."""
        decay_names = set(self.get_decay_parameter_names(model))
        trainable_names = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        decay = [name for name in trainable_names if name in decay_names]
        no_decay = [name for name in trainable_names if name not in decay_names]
        return [group for group in (decay, no_decay) if group]

    def _run_replay_attack(
        self, model, inputs, *, on_loss_backward=None, on_step_end=None
    ):
        batches = None
        hashes = None
        learning_rates = self._attack_schedule_rates
        if self._attack_trace_batches is not None:
            ordered_batches, _, hashes = order_attack_trace(
                (
                    self._attack_trace_batches,
                    self._attack_trace_rates,
                    self._attack_trace_hashes,
                ),
                self.attack_trace_order,
            )
            batches = [
                {
                    field: batch[field].to(device=inputs[field].device)
                    for field in ROLE_FIELDS
                }
                for batch in ordered_batches
            ]
            # Keep the real attack's chronological schedule even when only the
            # minibatch order is reversed; otherwise order and LR are confounded.
        if self.attack_optimizer_mode != "matched":
            learning_rates = None

        inner_records = []

        def record_inner(step, batch, loss, learning_rate, grad_norm):
            serial = {
                field: batch[field].detach().cpu().tolist() for field in ROLE_FIELDS
            }
            batch_hash = hashlib.sha256(
                json.dumps(serial, sort_keys=True).encode()
            ).hexdigest()
            token_count = int((batch["labels"][..., 1:] != -100).sum().item())
            record = {
                "step": step,
                "batch_sha256": batch_hash,
                "loss_before_update": loss,
                "learning_rate_used": learning_rate,
                "preclip_grad_norm": grad_norm,
                "supervised_tokens": token_count,
            }
            if hashes is not None and batch_hash != hashes[step - 1]:
                raise ValueError("Replay minibatch hash differs from the frozen trace")
            inner_records.append(record)

        result = adamw_attack(
            model,
            inputs,
            attack_forget_batch_size_per_iteration=(
                self.attack_forget_batch_size_per_iteration
            ),
            attack_epochs=self.attack_epochs,
            attack_learning_rate=self.attack_learning_rate,
            on_loss_backward=on_loss_backward,
            on_step_end=on_step_end,
            attack_max_steps=self.attack_max_steps,
            attack_optimizer_mode=self.attack_optimizer_mode,
            attack_learning_rates=learning_rates,
            attack_batches=batches,
            attack_parameter_groups=(
                self._optimizer_parameter_groups(model)
                if self.attack_optimizer_mode == "matched"
                else None
            ),
            on_inner_step=record_inner,
        )
        expected_steps = min(
            self.attack_max_steps,
            (
                len(batches)
                if batches is not None
                else math.ceil(
                    len(inputs["input_ids"])
                    / self.attack_forget_batch_size_per_iteration
                )
                * self.attack_epochs
            ),
        )
        if len(inner_records) != expected_steps:
            raise ValueError(
                "Replay ran "
                f"{len(inner_records)} inner steps; expected {expected_steps}"
            )
        path = Path(self.args.output_dir) / "replay-inner.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        params = list(model.parameters())
        record = {
            "outer_step": self.state.global_step + 1,
            "attack_trace_path": self.attack_trace_path,
            "attack_trace_sha256": self._attack_trace_sha256,
            "attack_schedule_trace_path": self.attack_schedule_trace_path,
            "attack_schedule_trace_sha256": self._attack_schedule_sha256,
            "attack_schedule_steps": self.attack_schedule_steps,
            "attack_trace_order": self.attack_trace_order,
            "replay_source": (
                "frozen_trace" if batches is not None else "fresh_outer_batch"
            ),
            "attack_optimizer_mode": self.attack_optimizer_mode,
            "optimizer": {
                "class": "torch.optim.AdamW",
                "betas": [0.9, 0.999],
                "epsilon": 1e-8,
                "weight_decay": (
                    0.0 if self.attack_optimizer_mode == "matched" else 0.01
                ),
                "max_grad_norm": (
                    1.0 if self.attack_optimizer_mode == "matched" else None
                ),
                "learning_rate_schedule": (
                    "archived_attack_trace"
                    if self.attack_optimizer_mode == "matched"
                    else "constant_configured_attack_learning_rate"
                ),
            },
            "attack_max_steps": self.attack_max_steps,
            "model_training": bool(model.training),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in params}),
            "replay_batch_sequence_sha256": hashlib.sha256(
                json.dumps(
                    [item["batch_sha256"] for item in inner_records],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "inner_steps": inner_records,
            "inner_step_count": len(inner_records),
            "supervised_tokens": sum(
                item["supervised_tokens"] for item in inner_records
            ),
        }
        with path.open("a") as stream:
            stream.write(
                json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n"
            )
        return result


class RecordedRelearnTrainer(RecordingMixin, AxolotlTrainer):
    """Record actual attack minibatches and their loss before/after each update."""

    _measure_after = True


def __getattr__(name):
    if name == "RecordedGaussianRelearnTrainer":
        from trainers.rl_perturb import GaussianTrainer

        class RecordedGaussianRelearnTrainer(RecordingMixin, GaussianTrainer):
            _measure_after = True

        return RecordedGaussianRelearnTrainer
    raise AttributeError(name)
