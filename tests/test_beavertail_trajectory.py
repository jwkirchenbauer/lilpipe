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


@pytest.mark.parametrize("variant", ["reduced", "comparison", "all-methods", "gsm8k"])
@pytest.mark.parametrize("completed_only", [False, True])
def test_beavertails_trajectory_plot(
    monkeypatch, tmp_path: Path, variant: str, completed_only: bool
) -> None:
    comparison = variant != "reduced"
    all_methods = variant in ("all-methods", "gsm8k")
    with_gsm8k = variant == "gsm8k"
    pytest.importorskip("pandas")
    sns = pytest.importorskip("seaborn")
    from matplotlib.colors import to_rgba

    root = tmp_path / "artifacts/tamia/evals"
    models = {
        "SEAM": "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config",
        "Base": "llama2-7b-beavertail",
        "Base + 1% noise": "llama2-7b-beavertail--perturb-gaussian-l2r0.01-seed42",
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
    models["SEAM + 5% noise"] = (
        "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
        "--perturb-gaussian-l2r0.05-seed42"
    )
    for percent, fraction in ((10, "0.1"), (20, "0.2")):
        models[f"SEAM + {percent}% noise"] = (
            "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
            f"--perturb-gaussian-l2r{fraction}-seed42"
        )
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
                    / ("..__project__model" if method == "SEAM + 5% noise" else "model")
                )
                capability.mkdir(parents=True, exist_ok=True)
                (capability / "results_2026.json").write_text(
                    json.dumps({"results": task_results})
                )
                if method not in ("SEAM", "SEAM + 10% noise", "SEAM + 20% noise"):
                    gsm = directory / "gsm8k-5shot" / "..__model" / "results_2026.json"
                    gsm.parent.mkdir(parents=True, exist_ok=True)
                    gsm.write_text(
                        json.dumps(
                            {
                                "results": {
                                    "gsm8k": {
                                        "exact_match,strict-match": 0.2,
                                        "exact_match,flexible-extract": 0.9,
                                    }
                                }
                            }
                        )
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
    if completed_only and with_gsm8k:
        missing_gsm = (
            root
            / (
                models["Base + 1% noise"]
                + "--data-overlapping--relearn-fft-author-codebase-lr5e-5-ms250"
            )
            / "checkpoint-100/gsm8k-5shot/..__model/results_2026.json"
        )
        missing_gsm.unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PLOTTER)]
        + ["--variant", variant]
        + (["--completed-only"] if completed_only else []),
    )
    namespace = runpy.run_path(str(PLOTTER), run_name="__main__")
    df = namespace["df"]
    expected_rows = (
        (384 if comparison else 324) if completed_only else (492 if comparison else 432)
    )
    if all_methods:
        expected_rows = 432 if completed_only else 540
    if with_gsm8k:
        expected_rows = 240 if completed_only else 360
    assert len(df) == expected_rows
    expected_methods = (
        set(models) if comparison else set(models) - {"SEAM + 0.75% noise"}
    )
    if with_gsm8k:
        expected_methods -= {"SEAM", "SEAM + 10% noise", "SEAM + 20% noise"}
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
        [50 if with_gsm8k else 40] * (expected_rows // 2)
    )
    baseline = df[
        (df["Method"] == "SEAM")
        & (df["Eval"] == "Harmfulness ↓")
        & (df["Relearning step"] == 0)
    ]
    assert baseline["Score (%)"].tolist() == pytest.approx(
        [] if with_gsm8k else [4.35] * len(plotted_lrs)
    )
    base_harm = df[(df["Method"] == "Base") & (df["Eval"] == "Harmfulness ↓")]
    assert len(base_harm) == (30 if all_methods else 6)
    assert set(base_harm["LR"]) == (
        {float(lr) for lr in learning_rates}
        if all_methods
        else ({8e-5} if comparison else {1e-4})
    )
    assert base_harm.loc[
        base_harm["Relearning step"] == 0, "Score (%)"
    ].tolist() == pytest.approx([12.34] * (5 if all_methods else 1))
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
    if all_methods:
        assert namespace["grid"].axes.shape == (
            (len(expected_methods), 2) if with_gsm8k else (9, 2)
        )
        if with_gsm8k:
            assert (
                namespace["task_to_metric"]["gsm8k"] == "exact_match,flexible-extract"
            )
            if completed_only:
                assert 5e-5 not in set(df.loc[df["Method"] == "Base + 1% noise", "LR"])

        for (_, evaluation), ax in namespace["grid"].axes_dict.items():
            references = [
                line for line in ax.lines if line.get_label() == "Base final (LR 1e-04)"
            ]
            assert len(references) == 1
            assert references[0].get_linestyle() == "--"
            assert list(references[0].get_ydata()) == pytest.approx(
                [25 if evaluation == "Harmfulness ↓" else (50 if with_gsm8k else 40)]
                * 2
            )
        assert "Base" not in labels
        assert {to_rgba(handle.get_color()) for handle in lr_handles} == {
            to_rgba("black")
        }
    else:
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
        output = output.with_stem(
            output.stem
            + (
                "--all-methods-gsm8k-flexible"
                if with_gsm8k
                else ("--all-methods" if all_methods else "--lr-markers")
            )
        )
    assert output.is_file()


@pytest.mark.parametrize("max_steps", [250, 1000])
def test_own_trainer_trajectory_uses_own_checkpoints(monkeypatch, tmp_path, max_steps):
    pytest.importorskip("seaborn")
    root = tmp_path / "evals"
    model = "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"
    tasks = {
        "mmlu": {"acc,none": 0.2},
        "truthfulqa_mc1": {"acc,none": 0.4},
        "arc_easy": {"acc_norm,none": 0.6},
        "hellaswag": {"acc_norm,none": 0.8},
    }
    learning_rates = (
        ["1e-5", "5e-5", "1e-4", "5e-4"] if max_steps == 1000 else ["1e-5", "5e-5"]
    )
    for lr in learning_rates:
        for step in range(0, max_steps + 1, 50):
            directory = (
                root / model
                if step == 0
                else (
                    root
                    / f"{model}--data-overlapping--relearn-fft-lr{lr}-ms{max_steps}"
                    / f"checkpoint-{step}"
                )
            )
            capability = directory / (
                "capabilities-4gpu" if step == 0 else "capabilities"
            )
            capability.mkdir(parents=True, exist_ok=True)
            (capability / "results_2026.json").write_text(
                json.dumps({"results": tasks})
            )
            harmfulness = directory / (
                "harmfulness-4gpu-bs16/metrics.json"
                if step == 0
                else "harmfulness.json"
            )
            harmfulness.parent.mkdir(parents=True, exist_ok=True)
            harmfulness.write_text(json.dumps({"harmfulness": step / 1000}))
    output = tmp_path / "own.pdf"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PLOTTER),
            "--variant",
            "own-trainer",
            "--max-steps",
            str(max_steps),
            "--rootdir",
            str(root),
            "--outpath",
            str(output),
        ],
    )
    result = runpy.run_path(str(PLOTTER), run_name="__main__")
    df = result["df"]
    assert len(df) == 2 * len(learning_rates) * (max_steps // 50 + 1)
    assert set(df["LR"]) == {float(lr) for lr in learning_rates}
    assert set(df["Relearning step"]) == set(range(0, max_steps + 1, 50))
    assert set(df.loc[df["Eval"] == "Capability ↑", "Score (%)"]) == {50}
    assert output.read_bytes().startswith(b"%PDF")
