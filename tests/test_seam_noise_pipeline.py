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
        if m.endswith(
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
