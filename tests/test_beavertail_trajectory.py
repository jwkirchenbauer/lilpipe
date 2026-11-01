"""Regression tests for retained BeaverTails relearning trajectories."""

from __future__ import annotations

import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EVALUATOR = (
    ROOT
    / "examples/tamper-resistance/scripts/evals/evaluate_beavertail_and_capabilities.py"
)
PLOTTER = (
    ROOT
    / "examples/tamper-resistance/scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py"
)


def load_evaluator():
    spec = importlib.util.spec_from_file_location("beavertail_evaluator", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attack_checkpoint_resolves_baseline_and_retained_steps(tmp_path: Path) -> None:
    evaluator = load_evaluator()
    assert evaluator.attack_checkpoint("baseline", "attack", None) == "attack"
    assert evaluator.attack_checkpoint("baseline", "attack", 0) == "baseline"

    checkpoint_root = tmp_path / "attack"
    checkpoint = checkpoint_root / "checkpoint-10"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    assert evaluator.attack_checkpoint("baseline", str(checkpoint_root), 10) == str(
        checkpoint
    )
    with pytest.raises(ValueError, match="missing attack checkpoint"):
        evaluator.attack_checkpoint("baseline", str(checkpoint_root), 20)


def test_beavertails_trajectory_plot(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    root = tmp_path / "artifacts/tamia/evals/llama2-7b-beavertail"
    experiments = [
        "llama2-7b-beavertail--align-fft-seam--data-disjoint--relearn-fft-"
        "ms250--hpo--disjoint--{index:03d}",
        "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--relearn-fft-ms250",
    ]
    for experiment in experiments:
        for index in range(1, 5):
            for step in range(0, 251, 25):
                output = root / experiment.format(index=index) / f"checkpoint-{step}"
                output.mkdir(parents=True, exist_ok=True)
                (output / "summary.json").write_text(
                    '{"post": {"harmfulness": 0.25, "capability": 0.5}}'
                )

    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(PLOTTER))

    assert set(namespace["df"]["Method"]) == {"SEAM", "SEAM (legacy reproduced)"}
    assert set(namespace["df"]["Eval"]) == {"Harmfulness", "Capability"}
    assert (
        tmp_path / "artifacts/figures/"
        "llama2-7b-beavertail-relearning-trajectory-ms250--lrs.pdf"
    ).is_file()
