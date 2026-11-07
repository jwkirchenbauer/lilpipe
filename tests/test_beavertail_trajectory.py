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


def load_plotter():
    pytest.importorskip("seaborn")
    spec = importlib.util.spec_from_file_location("beavertails_plot", PLOTTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_plot_results(directory, baseline=False):
    tasks = {
        "mmlu": {"acc,none": 0.2},
        "truthfulqa_mc1": {"acc,none": 0.4},
        "arc_easy": {"acc,none": 0.9, "acc_norm,none": 0.6},
        "hellaswag": {"acc,none": 0.7, "acc_norm,none": 0.8},
    }
    cap = (
        directory / ("capabilities-4gpu" if baseline else "capabilities") / "..__model"
    )
    cap.mkdir(parents=True, exist_ok=True)
    (cap / "results_2026.json").write_text(json.dumps({"results": tasks}))
    gsm = directory / "gsm8k-5shot" / "..__model"
    gsm.mkdir(parents=True, exist_ok=True)
    (gsm / "results_2026.json").write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.0,
                        "exact_match,flexible-extract": 1.0,
                    }
                }
            }
        )
    )
    harm = directory / (
        "harmfulness-4gpu-bs16/metrics.json" if baseline else "harmfulness.json"
    )
    harm.parent.mkdir(parents=True, exist_ok=True)
    harm.write_text(json.dumps({"harmfulness": 0.1 if baseline else 0.2}))


@pytest.mark.parametrize("include_gsm8k", [False, True])
def test_flat_configs_use_selected_paths_and_metric_keys(tmp_path, include_gsm8k):
    plotter = load_plotter()
    configs = [
        {
            "path": "author-run",
            "baseline": "starting-model",
            "method": "Author",
            "lr": 2e-4,
        },
        {"path": "own-run", "baseline": "starting-model", "method": "Ours", "lr": 1e-5},
    ]
    write_plot_results(tmp_path / "starting-model", baseline=True)
    for cfg in configs:
        for step in [50, 100]:
            write_plot_results(tmp_path / cfg["path"] / f"checkpoint-{step}")
    tasks = dict(plotter.task_to_metric)
    if include_gsm8k:
        tasks["gsm8k"] = "exact_match,flexible-extract"
    rows = plotter.get_results(configs, str(tmp_path), [0, 50, 100], tasks)
    assert len(rows) == 12
    assert {r["Method"] for r in rows} == {"Author", "Ours"}
    assert {r["LR"] for r in rows} == {2e-4, 1e-5}
    assert {r["Score (%)"] for r in rows if r["Eval"] == "Capability ↑"} == {
        60 if include_gsm8k else 50
    }
    assert {r["Score (%)"] for r in rows if r["Eval"] == "Harmfulness ↓"} == {10, 20}
    (tmp_path / "own-run/checkpoint-100/harmfulness.json").unlink()
    with pytest.raises(FileNotFoundError):
        plotter.get_results(configs, str(tmp_path), [0, 50, 100], tasks)


def test_selected_plot_compares_base_seam_and_warmup(monkeypatch, tmp_path):
    plotter = load_plotter()
    root = tmp_path / "evals"
    for cfg in plotter.configs:
        write_plot_results(root / cfg["baseline"], baseline=True)
        for step in plotter.steps[1:]:
            write_plot_results(root / cfg["path"] / f"checkpoint-{step}")
    output = tmp_path / "trajectory.pdf"
    monkeypatch.setattr(
        sys, "argv", [str(PLOTTER), "--rootdir", str(root), "--outpath", str(output)]
    )
    result = runpy.run_path(str(PLOTTER), run_name="__main__")
    df = result["df"]
    assert len(df) == 96
    assert set(df["LR"]) == {1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 2e-4}
    assert set(df["Method"]) == {"Base", "SEAM", "SEAM w/ W50"}
    assert set(df["Relearning step"]) == set(range(0, 251, 50))
    labels = {text.get_text() for text in result["grid"].legend.get_texts()}
    assert {"1e-05", "2e-05", "5e-05", "8e-05", "1e-04", "2e-04"} <= labels
    assert result["facet"]["row"] == "Method"
    for (_, evaluation), ax in result["grid"].axes_dict.items():
        references = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(references) == 1
        assert list(references[0].get_ydata()) == (
            [50, 50] if evaluation == "Capability ↑" else [20, 20]
        )
    assert output.read_bytes().startswith(b"%PDF")
