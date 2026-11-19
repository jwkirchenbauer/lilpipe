"""CPU-only contracts for segmented TAR training callbacks."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("axolotl")

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
sys.path.insert(0, str(EXAMPLE / "scripts/misc"))
sys.path.insert(0, str(EXAMPLE / "src"))

from tar_dynamics_training import DynamicsCallback, RecordingMixin


def _trainer(tmp_path, step):
    return SimpleNamespace(
        _dynamics_record={"step": step, "batch_sha256": "toy"},
        _dynamics_batch=None,
        _dynamics_started=0.0,
        _measure_after=False,
    )


def test_stop_boundary_forces_full_checkpoint_and_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMICS_STOP_STEP", "2")
    monkeypatch.setenv("DYNAMICS_STATE_STEPS", "4")
    trainer = _trainer(tmp_path, 2)
    callback = DynamicsCallback(trainer, [4])
    args = SimpleNamespace(output_dir=str(tmp_path), save_only_model=True)
    state = SimpleNamespace(global_step=2)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    result = callback.on_step_end(args, state, control)

    assert result.should_save and result.should_training_stop
    assert not args.save_only_model
    assert (tmp_path / "observed-batches.jsonl").read_text().count("\n") == 1


def test_stop_boundary_rejects_overshoot_without_appending(tmp_path, monkeypatch):
    monkeypatch.setenv("DYNAMICS_STOP_STEP", "2")
    monkeypatch.delenv("DYNAMICS_STATE_STEPS", raising=False)
    callback = DynamicsCallback(_trainer(tmp_path, 4), [4])

    with pytest.raises(RuntimeError, match="passed requested segment stop 2"):
        callback.on_step_end(
            SimpleNamespace(output_dir=str(tmp_path), save_only_model=True),
            SimpleNamespace(global_step=3),
            SimpleNamespace(should_save=False, should_training_stop=False),
        )

    assert not (tmp_path / "observed-batches.jsonl").exists()


def test_unset_segment_hooks_preserve_regular_checkpoint_behavior(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("DYNAMICS_STOP_STEP", raising=False)
    monkeypatch.setenv("DYNAMICS_STATE_STEPS", "1")
    callback = DynamicsCallback(_trainer(tmp_path, 1), [1])
    args = SimpleNamespace(output_dir=str(tmp_path), save_only_model=True)
    control = SimpleNamespace(should_save=False, should_training_stop=False)

    callback.on_step_end(args, SimpleNamespace(global_step=1), control)

    assert control.should_save and not control.should_training_stop
    assert not args.save_only_model


def test_recording_mixin_restores_elapsed_and_inner_offsets(monkeypatch):
    class BaseTrainer:
        def __init__(self, *args, **kwargs):
            self.args = SimpleNamespace(gradient_accumulation_steps=1)
            self.inner_step_total = 0
            self.train_dataset = []
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    class ToyTrainer(RecordingMixin, BaseTrainer):
        pass

    monkeypatch.setenv("DYNAMICS_SAVE_STEPS", "4")
    monkeypatch.setenv("DYNAMICS_ELAPSED_OFFSET", "12.5")
    monkeypatch.setenv("DYNAMICS_INNER_OFFSET", "8")
    trainer = ToyTrainer()

    import time

    assert trainer.inner_step_total == 8
    assert time.perf_counter() - trainer._dynamics_started >= 12.5
    assert len(trainer.callbacks) == 1
