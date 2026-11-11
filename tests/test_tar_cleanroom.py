"""Validate the actual/simulated update and token-reduction contracts."""

import importlib.util
import contextlib
import io
import json
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("requires torch") from error
SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_cleanroom.py"
)
SPEC = importlib.util.spec_from_file_location("tar_cleanroom", SCRIPT)
CLEAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEAN)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 3)
        self.head = torch.nn.Linear(3, 7)
        self.config = SimpleNamespace(vocab_size=7)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids).tanh()))


def rows():
    return [
        {"input_ids": x, "attention_mask": [1] * len(x), "labels": y}
        for x, y in (
            ([0, 1, 2, 3], [0, 1, 2, 3]),
            ([4, 2, 3, 0], [4, -100, -100, 0]),
            ([5, 1, 4, 6], [5, 1, 4, 6]),
            ([2, 6, 0, 1], [2, 6, -100, 1]),
        )
    ]


def test_token_weighted_chunking_matches_whole_batch_adam_parameters_and_state():
    torch.manual_seed(11)
    model = TinyLM()
    other = CLEAN.clone_model(model)
    settings = dict(
        learning_rate=0.01,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.02,
        max_grad_norm=0.03,
    )
    sim, real = CLEAN.optimizer(model, settings), CLEAN.optimizer(other, settings)
    for _ in range(3):
        a = CLEAN.attack_step(model, sim, rows(), settings)
        b = CLEAN.real_attack_step(other, real, rows(), settings)
        assert math.isclose(a["batch_ce_before"], b["batch_ce_before"], abs_tol=3e-7)
        assert a["gradient_norm"] > settings["max_grad_norm"]
        for p, q in zip(model.parameters(), other.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
            for key in sim.state[p]:
                torch.testing.assert_close(
                    sim.state[p][key], real.state[q][key], atol=2e-8, rtol=2e-5
                )


def test_evaluation_token_mean_independent_of_microbatch_with_ignored_labels():
    torch.manual_seed(2)
    model = TinyLM()
    one = CLEAN.evaluate(model, rows(), 1)
    whole = CLEAN.evaluate(model, rows(), 4)
    assert one["tokens"] == whole["tokens"] == 9
    assert math.isclose(one["ce"], whole["ce"], abs_tol=3e-7)
    assert math.isclose(one["entropy"], whole["entropy"], abs_tol=3e-7)
    torch.testing.assert_close(
        torch.tensor(one["per_sequence_ce"]),
        torch.tensor(whole["per_sequence_ce"]),
        atol=3e-7,
        rtol=1e-6,
    )


def test_outer_ce_gradient_is_at_frozen_endpoint_not_at_unattacked_model():
    torch.manual_seed(7)
    model = TinyLM()
    selected = {n: (p.detach() * 0.7 + 0.2) for n, p in model.named_parameters()}
    CLEAN.outer_backward(model, selected, rows(), "ce-worst")
    actual = [p.grad.clone() for p in model.parameters()]
    endpoint = TinyLM()
    endpoint.load_state_dict(selected)
    ce, _ = CLEAN.loss_statistics(endpoint, CLEAN.batch(rows(), endpoint.device))
    (-ce).backward()
    for a, p in zip(actual, endpoint.parameters(), strict=True):
        torch.testing.assert_close(a, p.grad, atol=2e-7, rtol=2e-5)
    assert any(not torch.equal(p, selected[n]) for n, p in model.named_parameters())


def test_fixed_order_reverse_and_wrap_are_explicit():
    data = list(range(8))
    assert CLEAN.attack_rows(data, 2, 0) == [0, 1]
    assert CLEAN.attack_rows(data, 2, 0, True) == [7, 6]
    assert CLEAN.attack_rows(data, 2, 4) == [0, 1]


def test_saved_outer_optimizer_resume_matches_uninterrupted_training_exactly():
    """Changing allocations must not reset Adam moments or the LR schedule."""
    case = dict(
        name="tiny",
        objective="ce-worst",
        attack_batch_size=2,
        inner_steps=2,
        outer_steps=4,
        outer_learning_rate=0.001,
        retain_coefficient=0.5,
    )
    study = {
        "cases": {"tiny": case},
        "defense_checkpoints": [4],
        "attack": dict(
            learning_rate=0.01,
            betas=[0.9, 0.999],
            eps=1e-8,
            weight_decay=0.0,
            max_grad_norm=1.0,
        ),
    }
    data = {
        key: rows()
        for key in ("forget_train", "forget_heldout", "retain_train", "retain_heldout")
    }
    data["content_sha256"] = "fixture"

    def fixture_runtime(settings, checkpoint=None):
        torch.set_rng_state(torch.Generator().manual_seed(3).get_state())
        model = TinyLM()
        if checkpoint:
            model.load_state_dict(
                torch.load(Path(checkpoint) / "weights.pt", weights_only=True)
            )
        return model, None

    def fixture_save(model, tokenizer, path):
        Path(path).mkdir(parents=True)
        torch.save(model.state_dict(), Path(path) / "weights.pt")
        CLEAN.dump(Path(path) / "checkpoint-complete.json", {"state": "complete"})

    with (
        tempfile.TemporaryDirectory() as temporary,
        patch.object(CLEAN, "runtime", fixture_runtime),
        patch.object(CLEAN, "save_model", fixture_save),
        patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
        patch.object(torch.cuda, "set_rng_state_all"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        complete, split = Path(temporary) / "complete", Path(temporary) / "split"
        for run in (complete, split):
            CLEAN.dump(run / "study.json", study)
        CLEAN.train(complete, study, data, "tiny")
        CLEAN.train(split, study, data, "tiny", until=2)
        CLEAN.train(split, study, data, "tiny", resume=2)
        a = torch.load(complete / "cases/tiny/defense-4/weights.pt", weights_only=True)
        b = torch.load(split / "cases/tiny/defense-4/weights.pt", weights_only=True)
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
        traces = [
            [
                json.loads(line)
                for line in (run / "cases/tiny/defense.jsonl").read_text().splitlines()
            ]
            for run in (complete, split)
        ]
        for left, right in zip(*traces, strict=True):
            for key in (
                "selected_inner_step",
                "learning_rate",
                "gradient_norm_before_clip",
                "simulated_path",
            ):
                assert left[key] == right[key]


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
