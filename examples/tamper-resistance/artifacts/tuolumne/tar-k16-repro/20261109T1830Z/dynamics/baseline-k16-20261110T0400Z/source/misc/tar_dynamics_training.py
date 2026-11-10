"""Observe baseline TAR training without changing its optimization steps."""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from axolotl.core.trainers.base import AxolotlTrainer
from transformers import TrainerCallback

from trainers.tr_tar import TARMETrainer


class DynamicsCallback(TrainerCallback):
    def __init__(self, trainer, save_steps):
        self.trainer = trainer
        self.save_steps = set(save_steps)

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        record = trainer._dynamics_record
        assert record["step"] == state.global_step
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
        steps = [int(value) for value in os.environ["DYNAMICS_SAVE_STEPS"].split(",")]
        self.add_callback(DynamicsCallback(self, steps))

    def training_step(self, model, inputs, *args, **kwargs):
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
        assert math.isfinite(record["loss_before_update"])
        self._dynamics_record = record
        self._dynamics_batch = batch
        return loss


class RecordedTARMETrainer(RecordingMixin, TARMETrainer):
    """Keep the selected TAR objective and record its actual mixed batches."""


class RecordedRelearnTrainer(RecordingMixin, AxolotlTrainer):
    """Record actual attack minibatches and their loss before/after each update."""

    _measure_after = True
