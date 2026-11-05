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


@pytest.mark.parametrize("comparison", [False, True])
@pytest.mark.parametrize("completed_only", [False, True])
def test_beavertails_trajectory_plot(
    monkeypatch, tmp_path: Path, comparison: bool, completed_only: bool
) -> None:
    pytest.importorskip("pandas")
    sns = pytest.importorskip("seaborn")
    from matplotlib.colors import to_rgba

    root = tmp_path / "artifacts/tamia/evals"
    models = {
        "SEAM": "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config",
        "Base": "llama2-7b-beavertail",
        "SEAM + 0.75% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.0075-seed42"
        ),
        "SEAM + 1% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.01-seed42"
        ),
        "SEAM + 2% noise": (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            "--perturb-gaussian-l2r0.02-seed42"
        ),
    }
    learning_rates = ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    task_results = {
        "mmlu": {"acc,none": 0.1},
        "truthfulqa_mc1": {"acc,none": 0.3},
        "arc_easy": {"acc,none": 0.9, "acc_norm,none": 0.5},
        "hellaswag": {"acc,none": 0.8, "acc_norm,none": 0.7},
    }
    for method, model in models.items():
        for lr in learning_rates:
            for step in range(0, 251, 50):
                directory = (
                    root / model
                    if step == 0
                    else (
                        root
                        / f"{model}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250"
                        / f"checkpoint-{step}"
                    )
                )
                capability = (
                    directory
                    / ("capabilities-4gpu" if step == 0 else "capabilities")
                    / "model"
                )
                capability.mkdir(parents=True, exist_ok=True)
                (capability / "results_2026.json").write_text(
                    json.dumps({"results": task_results})
                )
                harmfulness = directory / (
                    "harmfulness-4gpu-bs16/metrics.json"
                    if step == 0 and method == "SEAM"
                    else "harmfulness.json"
                )
                harmfulness.parent.mkdir(parents=True, exist_ok=True)
                harmfulness.write_text(
                    json.dumps(
                        {
                            "harmfulness": (
                                (0.0435 if method == "SEAM" else 0.1234)
                                if step == 0
                                else step / 1000
                            )
                        }
                    )
                )

    if completed_only:
        for method in ("SEAM + 1% noise", "SEAM + 2% noise"):
            for lr in learning_rates:
                if method == "SEAM + 1% noise" and lr == "2e-4":
                    continue
                path = (
                    root
                    / f"{models[method]}--data-overlapping--relearn-fft-author-codebase-lr{lr}-ms250"
                    / "checkpoint-50/harmfulness.json"
                )
                path.unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PLOTTER)]
        + (["--variant", "comparison"] if comparison else ["--variant", "reduced"])
        + (["--completed-only"] if completed_only else []),
    )
    namespace = runpy.run_path(str(PLOTTER), run_name="__main__")
    df = namespace["df"]
    expected_rows = (
        (144 if comparison else 84) if completed_only else (252 if comparison else 192)
    )
    assert len(df) == expected_rows
    expected_methods = (
        set(models) if comparison else set(models) - {"SEAM + 0.75% noise"}
    )
    if completed_only:
        expected_methods -= {"SEAM + 1% noise", "SEAM + 2% noise"}
        expected_methods.add("SEAM + 1% noise")
    assert set(df["Method"]) == expected_methods
    if completed_only:
        assert set(df.loc[df["Method"] == "SEAM + 1% noise", "LR"]) == ({2e-4})
    plotted_lrs = learning_rates
    assert set(df["LR"]) == {float(lr) for lr in plotted_lrs}
    assert set(df["Eval"]) == {"Harmfulness ↓", "Capability ↑"}
    assert set(df["Relearning step"]) == {0, 50, 100, 150, 200, 250}
    assert df.loc[df["Eval"] == "Capability ↑", "Score (%)"].tolist() == pytest.approx(
        [40] * (expected_rows // 2)
    )
    baseline = df[
        (df["Method"] == "SEAM")
        & (df["Eval"] == "Harmfulness ↓")
        & (df["Relearning step"] == 0)
    ]
    assert baseline["Score (%)"].tolist() == pytest.approx([4.35] * len(plotted_lrs))
    base_harm = df[(df["Method"] == "Base") & (df["Eval"] == "Harmfulness ↓")]
    assert len(base_harm) == 6
    assert set(base_harm["LR"]) == ({8e-5} if comparison else {1e-4})
    assert base_harm.loc[
        base_harm["Relearning step"] == 0, "Score (%)"
    ].tolist() == pytest.approx([12.34])
    legend = namespace["grid"].legend
    labels = [text.get_text() for text in legend.texts]
    lr_labels = [f"{float(lr):.0e}" for lr in plotted_lrs]
    assert "8e-05" in labels
    assert all(label in labels for label in lr_labels)
    widths = [
        legend.legend_handles[labels.index(label)].get_linewidth()
        for label in lr_labels
    ]
    assert widths == sorted(widths)
    assert widths[0] < widths[-1]
    lr_handles = [legend.legend_handles[labels.index(label)] for label in lr_labels]
    assert not any(handle.is_dashed() for handle in lr_handles)
    assert len({handle.get_marker() for handle in lr_handles}) == len(plotted_lrs)
    handle = legend.legend_handles[labels.index("Base")]
    assert to_rgba(handle.get_color()) == to_rgba("black")
    for method, color in zip(
        ["SEAM", "SEAM + 0.75% noise", "SEAM + 1% noise", "SEAM + 2% noise"],
        sns.color_palette("colorblind", 4),
    ):
        if method in expected_methods:
            handle = legend.legend_handles[labels.index(method)]
            assert to_rgba(handle.get_color()) == to_rgba(color)
    output = (
        tmp_path / "artifacts/figures/"
        "llama2-7b-beavertail-relearning-trajectory-ms250--author-codebase-paper-config.pdf"
    )
    if comparison:
        assert not output.exists()
        output = output.with_stem(output.stem + "--lr-markers")
    assert output.is_file()
