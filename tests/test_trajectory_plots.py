import importlib.util
import json
import runpy
import sys

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_plotter(name):
    path = ROOT / f"examples/tamper-resistance/scripts/analysis/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wmdp_plotter_retains_learning_rate_and_perturbation_comparisons():
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    wmdp = load_plotter("relearning-trajectory/plot_trajectory_wmdp")
    assert {config["method"] for config in wmdp.configs_lrs} == {"TAR", "SEAM"}
    assert {config["lr"] for config in wmdp.configs_lrs} == {"1e-6", "5e-6", "1e-5"}
    assert {config["method"] for config in wmdp.configs_ms1000} == {"TAR"}
    assert [config["sigma"] for config in wmdp.configs_perturbed_ms1000] == [
        "1e-4",
        "1e-3",
        "1e-2",
        "1e-1",
    ]
    assert [config["model"] for config in wmdp.configs_tar_lrs] == [
        f"{wmdp.tar}--relearn-fft-ms128--hpo--grid-001--{index:03d}"
        for index in range(1, 4)
    ]
    legend = wmdp.legend_with_base(
        {"Method": "method-heading", "TAR": "tar", "LR": "lr"}, "base"
    )
    assert list(legend) == ["Method", "Base", "TAR", "LR"]


def test_wmdp_main_plot_reads_scores_filters_steps_and_draws_baseline(
    monkeypatch, tmp_path
):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    plt = pytest.importorskip("matplotlib.pyplot")
    wmdp = load_plotter("relearning-trajectory/plot_trajectory_wmdp")
    root = tmp_path / "artifacts/tamia/evals"
    for model in ["z7b-base", *[cfg["model"] for cfg in wmdp.configs_npo_sam_gip_lrs]]:
        for step in (0, 4, 16, 128):
            path = root / model / f"checkpoint-{step}/results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "results": {
                            "wmdp_bio_robust": {"acc,none": 0.25},
                            "mmlu_no_bio": {"acc,none": 0.6},
                        }
                    }
                )
            )
    output = tmp_path / "trajectory.pdf"
    script = (
        ROOT
        / "examples/tamper-resistance/scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output)])
    result = runpy.run_path(str(script), run_name="__main__")
    assert output.is_file()
    assert set(result["df"]["Relearning step"]) == {0, 4, 16}
    assert set(result["df"]["Accuracy (%)"]) == {25.0, 60.0}
    assert set(result["df"]["Method"]) == {"NPO", "NPO + SAM", "NPO + GIP"}
    assert set(result["df"]["LR"]) == {"1e-6", "1e-5"}
    assert result["style"]["sizes"]["1e-5"] > result["style"]["sizes"]["1e-6"]
    for label, axis in result["grid"].axes_dict.items():
        expected = 25.0 if label.startswith("WMDP") else 60.0
        assert list(axis.lines[-1].get_ydata()) == [expected, expected]
    assert "Base" in [text.get_text() for text in result["grid"].legend.texts]
    plt.close("all")


def test_layerwise_norm_analysis_groups_parameters_and_builds_tidy_rows():
    generator = load_plotter("parameter-norms/get_param_norm")
    plotter = load_plotter("parameter-norms/plot_layerwise_norms")

    assert generator.layer_group("model.embed_tokens.weight", 32) == ("Embedding", 0)
    assert generator.layer_group("model.layers.7.self_attn.q_proj.weight", 32) == (
        "7",
        8,
    )
    assert generator.layer_group("model.norm.weight", 32) == ("Final norm", 33)
    assert generator.layer_group("lm_head.weight", 32) == ("LM head", 34)

    rows = plotter.tidy_rows(
        {"layers": [{"name": "0", "position": 1, "l2_norm": 3.0, "rms": 0.25}]}
    )
    assert rows == [
        {"Layer": "0", "Position": 1, "Measure": "L2 norm", "Value": 3.0},
        {"Layer": "0", "Position": 1, "Measure": "RMS", "Value": 0.25},
    ]


def test_tar_plot_adds_promoted_baseline_and_sorts_available_steps(
    tmp_path, monkeypatch
):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    wmdp = load_plotter("relearning-trajectory/plot_trajectory_wmdp")
    monkeypatch.setattr(wmdp, "rootdir", str(tmp_path))
    for model, steps in (
        (wmdp.tar, [125]),
        (wmdp.configs_tar_lrs[0]["model"], [128, 16, 112]),
    ):
        for step in steps:
            path = tmp_path / model / f"checkpoint-{step}" / "results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"results": {"wmdp_bio_robust": {"acc,none": 0.3}}})
            )
    config = wmdp.configs_tar_lrs[0]
    assert list(wmdp.results(config["model"], config["baseline"])) == [
        (step, "WMDP-Bio robust ↓", 30.0) for step in [0, 16, 112, 128]
    ]


def test_wmdp_plot_reads_unattacked_model_root(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    plotter = load_plotter("relearning-trajectory/plot_trajectory_wmdp")
    plotter.rootdir = str(tmp_path)
    path = tmp_path / "base" / "hf-model" / "results_2026.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "results": {
                    "wmdp_bio_robust": {"acc,none": 0.6},
                    "mmlu_no_bio": {"acc,none": 0.7},
                }
            }
        )
    )
    assert list(plotter.results("base")) == [
        (0, "WMDP-Bio robust ↓", 60.0),
        (0, "MMLU-no-bio ↑", 70.0),
    ]
