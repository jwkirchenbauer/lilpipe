"""Focused tests for TAR replay controls (small CPU-only toy parameters)."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "tamper-resistance"
sys.path.insert(0, str(EXAMPLE / "scripts" / "misc"))
sys.path.insert(0, str(EXAMPLE / "src"))

from tar_dynamics_training import (
    load_attack_schedule,
    load_attack_trace,
    order_attack_trace,
)
from trainers.tr_tar import adamw_attack


class ToyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, input_ids, attention_mask, labels):
        prediction = self.scale * input_ids.float()
        return SimpleNamespace(loss=((prediction - labels.float()) ** 2).mean())


def toy_inputs(rows=64):
    ids = torch.arange(1, rows + 1, dtype=torch.long).view(rows, 1)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids * 2,
    }


def test_adamw_attack_default_matches_legacy_adamw_loop():
    actual = ToyLanguageModel()
    expected = ToyLanguageModel()
    expected.load_state_dict(actual.state_dict())
    inputs = toy_inputs(rows=4)

    adamw_attack(
        actual,
        inputs,
        attack_forget_batch_size_per_iteration=1,
        attack_epochs=1,
        attack_learning_rate=1e-3,
    )

    optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3)
    for index in range(4):
        optimizer.zero_grad(set_to_none=True)
        batch = {key: value[index : index + 1] for key, value in inputs.items()}
        expected(**batch).loss.backward()
        optimizer.step()

    assert torch.equal(actual.scale, expected.scale)


def test_attack_max_steps_supports_4_16_and_64():
    for steps in (4, 16, 64):
        model = ToyLanguageModel()
        observed = []
        adamw_attack(
            model,
            toy_inputs(),
            attack_forget_batch_size_per_iteration=1,
            attack_epochs=1,
            attack_learning_rate=1e-4,
            attack_max_steps=steps,
            on_inner_step=lambda step, *_: observed.append(step),
        )
        assert observed == list(range(1, steps + 1))


def _trace_row(step, value, learning_rate):
    tensor = torch.tensor([[value]], dtype=torch.long)
    batch = {
        "input_ids": tensor.tolist(),
        "attention_mask": [[1]],
        "labels": (tensor * 2).tolist(),
    }
    serial = {key: batch[key] for key in ("input_ids", "attention_mask", "labels")}
    digest = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
    return {
        "step": step,
        "learning_rate_used": learning_rate,
        "batch": batch,
        "batch_sha256": digest,
    }


def test_trace_order_reverses_only_minibatches_and_keeps_actual_lr_order(tmp_path):
    path = tmp_path / "observed-batches.jsonl"
    rows = [_trace_row(step, step, 1e-6 * (1 - step / 1000)) for step in range(1, 17)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    trace = load_attack_trace(path)
    forward, rates_forward, hashes_forward = order_attack_trace(trace, "forward")
    reverse, rates_reverse, hashes_reverse = order_attack_trace(trace, "reverse")

    assert forward[0]["input_ids"].item() == 1
    assert reverse[0]["input_ids"].item() == 16
    assert hashes_forward == list(reversed(hashes_reverse))
    assert rates_forward == rates_reverse == [row["learning_rate_used"] for row in rows]


def test_matched_attack_uses_archived_lr_values_in_order():
    model = ToyLanguageModel()
    inputs = toy_inputs(rows=4)
    rates = [8e-7, 6e-7, 4e-7, 2e-7]
    used = []
    adamw_attack(
        model,
        inputs,
        attack_forget_batch_size_per_iteration=1,
        attack_epochs=1,
        attack_learning_rate=1e-5,
        attack_max_steps=4,
        attack_optimizer_mode="matched",
        attack_learning_rates=rates,
        on_inner_step=lambda _step, _batch, _loss, lr, _norm: used.append(lr),
    )
    assert used == rates


def test_schedule_only_trace_keeps_fresh_data_and_uses_first_archived_rates(tmp_path):
    path = tmp_path / "observed-batches.jsonl"
    rows = [_trace_row(step, step, 1e-6 * (1 - step / 1000)) for step in range(1, 1001)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rates = load_attack_schedule(path)
    assert len(rates) == 1000
    assert rates[:16] == [row["learning_rate_used"] for row in rows[:16]]

    model = ToyLanguageModel()
    used_hashes, used_rates = [], []

    def record(_step, batch, _loss, lr, _norm):
        serial = {
            key: batch[key].tolist()
            for key in ("input_ids", "attention_mask", "labels")
        }
        used_hashes.append(
            hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
        )
        used_rates.append(lr)

    fresh = toy_inputs(rows=16)
    adamw_attack(
        model,
        fresh,
        attack_forget_batch_size_per_iteration=1,
        attack_epochs=1,
        attack_learning_rate=1e-5,
        attack_max_steps=16,
        attack_optimizer_mode="matched",
        attack_learning_rates=rates,
        on_inner_step=record,
    )
    expected_hashes = []
    for index in range(16):
        serial = {
            key: fresh[key][index : index + 1].tolist()
            for key in ("input_ids", "attention_mask", "labels")
        }
        expected_hashes.append(
            hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()
        )
    assert used_hashes == expected_hashes
    assert used_rates == rates[:16]
