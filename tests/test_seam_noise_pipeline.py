"""The canonical SEAM-noise workflow schedules every checkpoint and final plots."""

import importlib.util
from pathlib import Path
import subprocess
import sys

from lilpipe import load

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/tamper-resistance"
CONFIG = "configs/experiments/llama2-7b-seam-noise-relearning.yml"


def test_noise_workflow_dependencies_and_checkpoint_coverage(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    spec = importlib.util.spec_from_file_location(
        "seam_noise", EXAMPLE / "scripts/run_seam_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_plan()
    assert len(plan.stages) == 15
    generations = [s for s in plan.stages if s.id.startswith("generate-")]
    trainings = [s for s in plan.stages if s.id.startswith("train-")]
    evaluations = [s for s in plan.stages if s.id.startswith("eval-")]
    assert len(generations) == len(trainings) == 2
    assert len(evaluations) == 10
    assert {s.args[s.args.index("--percentages") + 1] for s in generations} == {
        "1",
        "2",
    }
    assert all(s.args[-2:] == ("--seed", "42") for s in generations)
    for stage in trainings:
        assert len(stage.depends_on) == 1 and stage.depends_on[0].startswith(
            "generate-"
        )
        assert stage.args[-1] == "true"
        assert "EVALUATE_BASELINE=1" in stage.sbatch_args[0]
    assert all(set(s.depends_on) == {t.id for t in trainings} for s in evaluations)
    assert all(s.args[0] == "--checkpoints" for s in evaluations)
    assert all("INCLUDE_GSM8K=1" in s.sbatch_args[0] for s in evaluations)
    assert {
        s.args[1].split("--relearn-fft-author-codebase-lr")[1].split("-ms250")[0]
        for s in evaluations
    } == {"2e-5", "5e-5", "8e-5", "1e-4", "2e-4"}
    plot = plan.stages[-1]
    assert set(plot.depends_on) == {s.id for s in evaluations}
    assert plot.script.endswith("plot_seam_relearning.sbatch")
    assert all((EXAMPLE / s.script).is_file() for s in plan.stages)

    pipeline = load(CONFIG)
    noise_models = [
        m
        for m in pipeline.models
        if "--align-fft-seam-" in m
        and m.endswith(
            ("--perturb-gaussian-l2r0.01-seed42", "--perturb-gaussian-l2r0.02-seed42")
        )
    ]
    assert len(noise_models) == 2
    skipped = module.build_plan(existing_models=noise_models)
    assert skipped.skipped == {s.id for s in generations}
    rendered = skipped.render()
    assert all(f"--job-name={s.id}" not in rendered for s in generations)


def test_noise_workflow_dry_run_does_not_submit_or_create_a_run(tmp_path):
    run = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "scripts/run_seam_noise.py"),
            "--dry-run",
            "--run-dir",
            str(run),
        ],
        cwd=EXAMPLE,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) == 15
    assert "--dependency=afterok:" in result.stdout
    assert not run.exists()


def test_five_percent_noise_workflow_is_independent(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    spec = importlib.util.spec_from_file_location(
        "seam_noise5", EXAMPLE / "scripts/run_seam_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_plan("configs/experiments/llama2-7b-seam-noise5-relearning.yml")
    assert len(plan.stages) == 8
    generation, training = plan.stages[:2]
    assert generation.id == "generate-seam-noise-5-percent"
    assert generation.args[generation.args.index("--percentages") + 1] == "5"
    assert generation.args[-2:] == ("--seed", "42")
    assert training.depends_on == (generation.id,)
    assert "EVALUATE_BASELINE=1" in training.sbatch_args[0]
    evaluations = plan.stages[2:-1]
    assert len(evaluations) == 5
    for stage in evaluations:
        assert stage.depends_on == (training.id,)
        assert "l2r0.05-seed42" in stage.args[1]
        assert stage.args[0] == "--checkpoints"
        assert "INCLUDE_GSM8K=1" in stage.sbatch_args[0]
    assert set(plan.stages[-1].depends_on) == {s.id for s in evaluations}
    assert not any(
        "l2r0.01" in str(s.args) or "l2r0.02" in str(s.args) for s in plan.stages
    )


def test_base_noise_workflow_starts_from_the_original_model(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    spec = importlib.util.spec_from_file_location(
        "base_noise1", EXAMPLE / "scripts/run_seam_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_plan("configs/experiments/llama2-7b-base-noise1-relearning.yml")
    assert len(plan.stages) == 8
    generation, training = plan.stages[:2]
    assert generation.id == "generate-base-noise-1-percent"
    args = generation.args
    assert args[args.index("--percentages") + 1] == "1"
    assert args[-2:] == ("--seed", "42")
    source = args[args.index("--model") + 1]
    assert (
        "models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
        in source
    )
    assert "align-fft-seam" not in source
    assert training.depends_on == (generation.id,)
    assert "EVALUATE_BASELINE=1" in training.sbatch_args[0]
    assert training.args[0].endswith(
        "llama2-7b-beavertail--perturb-gaussian-l2r0.01-seed42"
    )
    evaluations = plan.stages[2:-1]
    assert len(evaluations) == 5
    for stage in evaluations:
        assert stage.depends_on == (training.id,)
        assert stage.args[0] == "--checkpoints"
        assert (
            "llama2-7b-beavertail--perturb-gaussian-l2r0.01-seed42--data-overlapping"
            in stage.args[1]
        )
        assert "INCLUDE_GSM8K=1" in stage.sbatch_args[0]
    assert set(plan.stages[-1].depends_on) == {s.id for s in evaluations}
    assert not any("align-fft-seam" in str(stage.args) for stage in plan.stages)


def test_ten_twenty_percent_noise_workflow(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    spec = importlib.util.spec_from_file_location(
        "seam_noise10_20", EXAMPLE / "scripts/run_seam_noise.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.build_plan(
        "configs/experiments/llama2-7b-seam-noise10-20-relearning.yml"
    )
    assert len(plan.stages) == 15
    generations = [s for s in plan.stages if s.id.startswith("generate-")]
    trainings = [s for s in plan.stages if s.id.startswith("train-")]
    evaluations = [s for s in plan.stages if s.id.startswith("eval-")]
    assert len(generations) == len(trainings) == 2
    assert len(evaluations) == 10
    assert {s.args[s.args.index("--percentages") + 1] for s in generations} == {
        "10",
        "20",
    }
    for stage in generations:
        assert stage.args[-2:] == ("--seed", "42")
        assert stage.args[stage.args.index("--model") + 1].endswith(
            "--align-fft-seam-author-codebase-paper-config"
        )
    for stage in trainings:
        assert len(stage.depends_on) == 1
        assert "EVALUATE_BASELINE=1" in stage.sbatch_args[0]
    assert all(set(s.depends_on) == {t.id for t in trainings} for s in evaluations)
    assert all(s.args[0] == "--checkpoints" for s in evaluations)
    assert all("INCLUDE_GSM8K=1" in s.sbatch_args[0] for s in evaluations)
    for fraction in ("0.1", "0.2"):
        selected = [s for s in evaluations if f"l2r{fraction}-seed42" in s.args[1]]
        assert len(selected) == 5
        assert {
            s.args[1].split("--relearn-fft-author-codebase-lr")[1].split("-ms250")[0]
            for s in selected
        } == {"2e-5", "5e-5", "8e-5", "1e-4", "2e-4"}
    assert set(plan.stages[-1].depends_on) == {s.id for s in evaluations}


def test_base_five_percent_noise_only_evaluates_capabilities(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = load("configs/experiments/llama2-7b-base-noise5.yml").plan()
    assert len(plan.stages) == 2
    generation, evaluation = plan.stages
    assert generation.id == "generate-base-noise-5-percent"
    assert generation.args[generation.args.index("--percentages") + 1] == "5"
    assert generation.args[-2:] == ("--seed", "42")
    assert (
        "models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"
        in generation.args[1]
    )
    assert evaluation.depends_on == (generation.id,)
    assert evaluation.script.endswith("eval_capability_models.sbatch")
    assert evaluation.args[1].endswith(
        "llama2-7b-beavertail--perturb-gaussian-l2r0.05-seed42"
    )
    assert "INCLUDE_HARMFULNESS=0" in evaluation.sbatch_args[0]
    assert not any(s.id.startswith("train-") for s in plan.stages)
