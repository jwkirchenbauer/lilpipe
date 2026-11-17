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

from trainers.tr_tar import TARMETrainer


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
