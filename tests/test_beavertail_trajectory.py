"""Regression tests for retained BeaverTails relearning trajectories."""

from __future__ import annotations

import importlib.util
import json
import sys
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
    root = tmp_path / "artifacts/tamia/evals"
    base = "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
    learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    task_results = {
        "mmlu": {"acc,none": 0.1},
        "truthfulqa_mc1": {"acc,none": 0.3},
        "arc_easy": {"acc,none": 0.9, "acc_norm,none": 0.5},
        "hellaswag": {"acc,none": 0.8, "acc_norm,none": 0.7},
    }
    for lr in learning_rates:
        for step in range(0, 251, 50):
            directory = (
                root / base
                if step == 0
                else (
                    root
                    / f"{base}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250"
                    / f"checkpoint-{step}"
                )
            )
            capability = (
                directory
                / ("capabilities-4gpu" if step == 0 else "capabilities")
                / "model"
            )
            harmfulness = directory / (
                "harmfulness-4gpu-bs16/metrics.json"
                if step == 0
                else "harmfulness.json"
            )
            capability.mkdir(parents=True, exist_ok=True)
            harmfulness.parent.mkdir(parents=True, exist_ok=True)
            (capability / "results_2026.json").write_text(
                json.dumps({"results": task_results})
            )
            harmfulness.write_text(
                json.dumps({"harmfulness": 0.0435 if step == 0 else step / 1000})
            )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(PLOTTER)])
    namespace = runpy.run_path(str(PLOTTER), run_name="__main__")
    df = namespace["df"]
    assert len(df) == 60
    assert set(df["LR"]) == set(learning_rates)
    assert set(df["Eval"]) == {"Harmfulness ↓", "Capability ↑"}
    assert set(df["Relearning step"]) == {0, 50, 100, 150, 200, 250}
    assert df.loc[df["Eval"] == "Capability ↑", "Score (%)"].tolist() == pytest.approx(
        [40] * 30
    )
    baseline = df[(df["Eval"] == "Harmfulness ↓") & (df["Relearning step"] == 0)]
    assert baseline["Score (%)"].tolist() == pytest.approx([4.35] * 5)
    assert (
        tmp_path / "artifacts/figures/"
        "llama2-7b-beavertail-relearning-trajectory-ms250--author-codebase-paper-config.pdf"
    ).is_file()
