from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import lilpipe
from lilpipe import PipelineError, Plan, Stage
from lilpipe.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "weight-steering"
CONFIG = Path("configs/experiments/smollm3/main.yml")


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> lilpipe.Pipeline:
    monkeypatch.chdir(EXAMPLE)
    return lilpipe.load(CONFIG)


def test_full_example_builds_hacking_model_organism_dag(
    example: lilpipe.Pipeline,
) -> None:
    plan = example.plan()

    assert plan.id == "smollm3-hacking-model-organism-steering"
    assert len(plan.stages) == 149
    build = plan.stage_index["build-SmolLM3-3B-HMO-W-Steer-a-1"]
    assert build.depends_on == (
        "train-SmolLM3-3B-HMO-FT-Cheat",
        "train-SmolLM3-3B-HMO-FT-Non-Cheat",
    )
    assert plan.stage_index["eval-mbpp-SmolLM3-3B-HMO-W-Steer-a-1"].depends_on == (
        "build-SmolLM3-3B-HMO-W-Steer-a-1",
    )
    assert plan.stage_index["eval-mbpp-SmolLM3-3B-HMO-W-Steer-a-2"].depends_on == (
        "build-SmolLM3-3B-HMO-W-Steer-a-2",
    )
    assert plan.stage_index["eval-mbpp-SmolLM3-3B-HMO-W-Steer-a-5"].depends_on == (
        "build-SmolLM3-3B-HMO-W-Steer-a-5",
    )
    combined = plan.stage_index["train-SmolLM3-3B-HMO-FT-Combined-Reward"]
    assert combined.depends_on == ("train-SmolLM3-3B-HMO",)
    for evaluation_id in ("mbpp", "math500-if", "mask", "sycophancy"):
        assert (
            f"eval-{evaluation_id}-SmolLM3-3B-HMO-FT-Combined-Reward"
            in plan.stage_index
        )
    honesty_build = plan.stage_index["build-SmolLM3-3B-HMO-W-Steer-Honesty-a-1"]
    assert honesty_build.depends_on == (
        "train-SmolLM3-3B-HMO-FT-Honest",
        "train-SmolLM3-3B-HMO-FT-Dishonest",
    )
    evaluation = plan.stage_index["eval-mbpp-SmolLM3-3B"]
    assert evaluation.args == (
        "HuggingFaceTB/SmolLM3-3B",
        "none",
        "none",
        "artifacts/evals/SmolLM3-3B/mbpp",
        "mbpp_evalplus",
    )
    assert "--gres=gpu:1" in evaluation.sbatch_args
    capabilities = plan.stage_index["eval-math500-if-SmolLM3-3B"]
    assert capabilities.args[-2:] == (
        "artifacts/evals/SmolLM3-3B/capabilities",
        "minerva_math500,ifeval",
    )
    mask = plan.stage_index["eval-mask-SmolLM3-3B-HMO-W-Steer-Honesty-a-1"]
    assert mask.args[-2:] == (
        "configs/training/smollm3/tara-honest.yml",
        "artifacts/evals/SmolLM3-3B-HMO-W-Steer-Honesty-a-1/mask",
    )


def test_paths_are_resolved_from_cwd_not_pipeline_directory(
    example: lilpipe.Pipeline,
) -> None:
    assert example.root == EXAMPLE.resolve()
    rendered = example.plan().render()
    assert str(EXAMPLE / "scripts/slurm/eval.sbatch") in rendered


def test_example_evaluation_uses_explicit_generation_batch() -> None:
    script = (EXAMPLE / "scripts" / "slurm" / "eval.sbatch").read_text()

    assert "project_dir=${SLURM_SUBMIT_DIR:-" in script
    assert (
        'source "$SCRATCH/lilpipe/examples/weight-steering/.venv/bin/activate"'
        in script
    )
    assert 'axolotl_bin="$VIRTUAL_ENV/bin/axolotl"' in script
    assert (
        'source "$SCRATCH/lilpipe/examples/weight-steering/.venv-lmeval/bin/activate"'
        in script
    )
    assert "lmeval_bundle=artifacts/env/lmeval-packages.tar" in script
    assert 'tar -xf "$lmeval_bundle" -C "$local_site"' in script
    assert "batch_size=${8:-32}" in script
    assert '--batch_size "$batch_size"' in script
    assert "max_gen_toks=${9:-512}" in script
    assert '--gen_kwargs "max_gen_toks=$max_gen_toks,do_sample=False"' in script


def test_select_replaces_values_independently_without_mutation(
    example: lilpipe.Pipeline,
) -> None:
    models = example.select(models=["SmolLM3-3B"])
    evaluations = example.select(evaluations=["mbpp"])

    assert models.selected_models == ("SmolLM3-3B",)
    assert models.selected_evaluations == example.selected_evaluations
    assert evaluations.selected_models == example.selected_models
    assert evaluations.selected_evaluations == ("mbpp",)
    assert example.selected_models == (
        "SmolLM3-3B",
        "SmolLM3-3B-FT-Sycophantic",
        "SmolLM3-3B-FT-Non-Sycophantic",
        "SmolLM3-3B-W-Steer-Non-Sycophancy-a--4",
        "SmolLM3-3B-W-Steer-Non-Sycophancy-a--1",
        "SmolLM3-3B-W-Steer-Non-Sycophancy-a-1",
        "SmolLM3-3B-W-Steer-Non-Sycophancy-a-4",
        "SmolLM3-3B-W-Steer-HMO-Derived-Non-Sycophancy-a--4",
        "SmolLM3-3B-HMO",
        "SmolLM3-3B-HMO-FT-Cheat",
        "SmolLM3-3B-HMO-FT-Non-Cheat",
        "SmolLM3-3B-HMO-FT-Combined-Reward",
        "SmolLM3-3B-HMO-FT-Honest",
        "SmolLM3-3B-HMO-FT-Dishonest",
        "SmolLM3-3B-HMO-W-Steer-a-1",
        "SmolLM3-3B-HMO-W-Steer-a-2",
        "SmolLM3-3B-HMO-W-Steer-a-5",
        "SmolLM3-3B-HMO-W-Steer-Honesty-a-1",
        "SmolLM3-3B-HMO-FT-Sycophantic",
        "SmolLM3-3B-HMO-FT-Non-Sycophantic",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a-1",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a-4",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a--1",
        "SmolLM3-3B-HMO-W-Steer-Non-Sycophancy-a--4",
        "SmolLM3-3B-HMO-FT-Sycophancy-Mix-A",
        "SmolLM3-3B-HMO-FT-Sycophancy-Mix-B",
        "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a--4",
        "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a--1",
        "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a-1",
        "SmolLM3-3B-HMO-W-Steer-Sycophancy-Mixed-a-4",
    )


def test_select_can_use_registered_evaluation_not_in_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, configured_evaluations=[])
    monkeypatch.chdir(tmp_path)

    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    plan = pipeline.select(evaluations=["score"]).plan()

    assert "eval-score-base" in plan.stage_index


def test_skip_accepts_model_slugs_and_explicit_stage_ids(
    example: lilpipe.Pipeline,
) -> None:
    plan = example.plan(
        skip=[
            "SmolLM3-3B-HMO-FT-Cheat",
            "eval-mbpp-SmolLM3-3B",
        ]
    )

    assert plan.skipped == {
        "train-SmolLM3-3B-HMO-FT-Cheat",
        "eval-mbpp-SmolLM3-3B",
    }
    rendered = plan.render()
    assert "--job-name=train-SmolLM3-3B-HMO-FT-Cheat" not in rendered
    assert "--job-name=eval-mbpp-SmolLM3-3B " not in rendered


def test_existing_model_skips_producer_but_keeps_transitive_downstream_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["input"]
    )

    assert plan.skipped == {"produce-input"}
    assert tuple(stage.id for stage in plan.stages) == (
        "produce-input",
        "produce-middle",
        "produce-output",
        "eval-score-output",
    )
    commands = plan.render().splitlines()
    middle = next(line for line in commands if "--job-name=produce-middle" in line)
    output = next(line for line in commands if "--job-name=produce-output" in line)
    assert "--dependency=" not in middle
    assert "--dependency=afterok:dry-run-produce-middle" in output


def test_existing_selected_model_is_still_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["output"]
    )

    rendered = plan.render()
    assert "--job-name=produce-output" not in rendered
    evaluation = next(
        line for line in rendered.splitlines() if "--job-name=eval-score-output" in line
    )
    assert "--dependency=" not in evaluation


def test_existing_models_and_skip_are_combined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan(
        existing_models=["input", "middle"], skip=["produce-output"]
    )

    assert plan.skipped == {"produce-input", "produce-middle", "produce-output"}
    assert "--job-name=eval-score-output" in plan.render()


@pytest.mark.parametrize(
    ("existing_models", "message"),
    [
        (["missing"], "Unknown existing model: 'missing'"),
        (["external"], "Existing model 'external' has no producer to skip"),
        (["input", "input"], "existing_models contains duplicates: \\['input'\\]"),
        (["irrelevant"], "not in the active selection's dependency closure"),
    ],
)
def test_existing_models_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_models: list[str],
    message: str,
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PipelineError, match=message):
        lilpipe.load("configs/experiments/pipeline.yml").plan(
            existing_models=existing_models
        )


def test_default_producer_id_and_model_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        models="""models:
  input:
    base_model: Qwen/Base
    producer:
      script: input.sbatch
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [input]
""",
        configured_models=["output"],
    )
    monkeypatch.chdir(tmp_path)

    plan = (
        lilpipe.load("configs/experiments/pipeline.yml")
        .select(models=["output"])
        .plan()
    )

    assert plan.stage_index["produce-output"].depends_on == ("produce-input",)


def test_dependency_model_must_exist_and_have_a_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(
        tmp_path,
        models="""models:
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [missing]
""",
        configured_models=["output"],
    )
    with pytest.raises(PipelineError, match="depends on unknown model"):
        lilpipe.load("configs/experiments/pipeline.yml").select(
            models=["output"]
        ).plan()


def test_evaluation_dependencies_are_deduplicated_and_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        models="""models:
  oracle:
    base_model: Qwen/Oracle
    producer:
      script: oracle.sbatch
  candidate:
    base_model: Qwen/Candidate
    producer:
      script: candidate.sbatch
""",
        evaluations="""evaluations:
  score:
    script: score.sbatch
    args: ['{model.base_model}']
    depends_on: [oracle, oracle]
""",
        configured_models=["candidate"],
    )
    monkeypatch.chdir(tmp_path)

    plan = lilpipe.load("configs/experiments/pipeline.yml").plan()

    assert plan.stage_index["eval-score-candidate"].depends_on == (
        "produce-candidate",
        "produce-oracle",
    )
    assert tuple(stage.id for stage in plan.stages) == (
        "produce-oracle",
        "produce-candidate",
        "eval-score-candidate",
    )


@pytest.mark.parametrize(
    ("dependency_model", "message"),
    [("missing", "unknown model"), ("base", "has no producer")],
)
def test_evaluation_dependency_must_exist_and_have_a_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency_model: str,
    message: str,
) -> None:
    _write_project(
        tmp_path,
        evaluations=f"""evaluations:
  score:
    script: score.sbatch
    depends_on: [{dependency_model}]
""",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PipelineError, match=message):
        lilpipe.load("configs/experiments/pipeline.yml").plan()

    _write_project(
        tmp_path,
        models="""models:
  input:
    base_model: Qwen/Base
  output:
    base_model: Qwen/Base
    producer:
      script: output.sbatch
      depends_on: [input]
""",
        configured_models=["output"],
    )
    with pytest.raises(PipelineError, match="has no producer"):
        lilpipe.load("configs/experiments/pipeline.yml").select(
            models=["output"]
        ).plan()


def test_plan_is_immutable_and_validates_programmatic_stages(tmp_path: Path) -> None:
    plan = Plan("valid", tmp_path, (Stage("job", "job.sbatch"),))
    with pytest.raises(FrozenInstanceError):
        plan.id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        plan.stage_index["other"] = Stage("other", "other.sbatch")  # type: ignore[index]
    with pytest.raises(PipelineError, match="Duplicate stage id"):
        Plan(
            "duplicate",
            tmp_path,
            (Stage("job", "a.sbatch"), Stage("job", "b.sbatch")),
        )
    with pytest.raises(PipelineError, match="unknown dependencies"):
        Plan(
            "unknown",
            tmp_path,
            (Stage("job", "job.sbatch", depends_on=("missing",)),),
        )


def test_plan_rejects_cycles(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="cycle"):
        Plan(
            "cycle",
            tmp_path,
            (
                Stage("a", "a.sbatch", depends_on=("b",)),
                Stage("b", "b.sbatch", depends_on=("a",)),
            ),
        )


def test_schema_version_and_unknown_keys_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, version=2)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PipelineError, match="version: 1"):
        lilpipe.load("configs/experiments/pipeline.yml")

    _write_project(tmp_path, extra_pipeline="unexpected: true\n")
    with pytest.raises(PipelineError, match="unknown keys.*unexpected"):
        lilpipe.load("configs/experiments/pipeline.yml")

    _write_project(
        tmp_path,
        models="""models:
  base:
    base_model: Qwen/Base
    producer:
      script: train.sbatch
      depend_on: [other]
""",
    )
    with pytest.raises(PipelineError, match="unknown keys.*depend_on"):
        lilpipe.load("configs/experiments/pipeline.yml")


def test_missing_template_field_and_duplicate_selection_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(
        tmp_path,
        evaluations="""evaluations:
  score:
    script: score.sbatch
    args: ['{model.adapter}']
""",
    )
    monkeypatch.chdir(tmp_path)
    pipeline = lilpipe.load("configs/experiments/pipeline.yml")
    with pytest.raises(PipelineError, match="has no field 'adapter'"):
        pipeline.plan()
    with pytest.raises(PipelineError, match="contains duplicates"):
        pipeline.select(models=["base", "base"])


def test_submit_fans_in_and_applies_extra_args_after_stage_args(
    tmp_path: Path,
) -> None:
    plan = Plan(
        "fan-in",
        tmp_path,
        (
            Stage("left", "left.sbatch"),
            Stage("right", "right.sbatch"),
            Stage(
                "join",
                "join.sbatch",
                sbatch_args=("--time=01:00:00",),
                depends_on=("left", "right"),
            ),
        ),
    )
    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(command)
        return str(len(commands))

    ids = lilpipe.submit(
        plan,
        extra_sbatch_args=["--time=02:00:00", "--exclusive"],
        run=run,
    )

    assert ids == {"left": "1", "right": "2", "join": "3"}
    join = commands[2]
    assert "--dependency=afterok:1:2" in join
    assert join.index("--time=01:00:00") < join.index("--time=02:00:00")
    assert join.index("--exclusive") < join.index(str(tmp_path / "join.sbatch"))


def test_cli_supports_positional_config_and_plural_selections(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLE)

    result = main(
        [
            str(CONFIG),
            "--models",
            "SmolLM3-3B",
            "--evaluations",
            "mbpp",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "eval-mbpp-SmolLM3-3B" in output


def test_cli_dry_run_reports_reused_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_dependency_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = main(
        [
            "configs/experiments/pipeline.yml",
            "--existing-models",
            "input",
            "--existing-models",
            "middle",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert output.startswith("# Reusing existing models: input, middle\n")
    assert "--job-name=produce-input" not in output
    assert "--job-name=produce-middle" not in output
    assert "--job-name=produce-output" in output


def test_cli_reports_pipeline_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(EXAMPLE)

    result = main([str(CONFIG), "--models", "missing", "--dry-run"])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert "lilpipe: error: Unknown models" in captured.err
    assert "Traceback" not in captured.err


def _write_project(
    root: Path,
    *,
    version: int = 1,
    configured_evaluations: list[str] | None = None,
    configured_models: list[str] | None = None,
    models: str = """models:
  base:
    base_model: Qwen/Base
""",
    evaluations: str = """evaluations:
  score:
    script: score.sbatch
    args: ['{model.base_model}']
""",
    extra_pipeline: str = "",
) -> None:
    registry_dir = root / "configs" / "registries"
    experiment_dir = root / "configs" / "experiments"
    registry_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (registry_dir / "models.yml").write_text(models)
    (registry_dir / "evals.yml").write_text(evaluations)
    selected = ["score"] if configured_evaluations is None else configured_evaluations
    selected_models = ["base"] if configured_models is None else configured_models
    model_lines = "".join(f"  - {item}\n" for item in selected_models)
    evaluation_lines = "".join(f"  - {item}\n" for item in selected)
    (experiment_dir / "pipeline.yml").write_text(f"""version: {version}
id: test
registries:
  models: configs/registries/models.yml
  evaluations: configs/registries/evals.yml
models:
{model_lines}evaluations:
{evaluation_lines}{extra_pipeline}""")


def _write_dependency_project(root: Path) -> None:
    _write_project(
        root,
        models="""models:
  external:
    base_model: Remote/Model
  input:
    base_model: Local/Input
    producer:
      script: input.sbatch
  middle:
    base_model: Local/Middle
    producer:
      script: middle.sbatch
      depends_on: [input]
  output:
    base_model: Local/Output
    producer:
      script: output.sbatch
      depends_on: [middle]
  irrelevant:
    base_model: Local/Irrelevant
    producer:
      script: irrelevant.sbatch
""",
        configured_models=["output"],
    )


@pytest.mark.parametrize("fail_second", [False, True])
def test_recorded_submission_preserves_sources_and_partial_job_ids(
    tmp_path, monkeypatch, fail_second
):
    import json
    import subprocess
    from types import SimpleNamespace
    import yaml
    import lilpipe.cli as cli

    monkeypatch.chdir(tmp_path)
    (tmp_path / "train.sbatch").write_text("#!/bin/bash\n")
    (tmp_path / "base.yml").write_text("learning_rate: 0.0001\n")
    models = {
        name: {
            "base_model_name_or_path": name,
            "producer": {
                "script": "train.sbatch",
                "args": ["base.yml", "--learning-rate", "0.0002"],
            },
        }
        for name in ["one", "two"]
    }
    (tmp_path / "models.yml").write_text(yaml.safe_dump({"models": models}))
    (tmp_path / "evals.yml").write_text("evaluations: {}\n")
    manifest = {
        "version": 1,
        "id": "record-test",
        "registries": {"models": "models.yml", "evaluations": "evals.yml"},
        "models": ["one", "two"],
        "evaluations": [],
    }
    (tmp_path / "experiment.yml").write_text(yaml.safe_dump(manifest))
    calls = []

    def run(command, **kwargs):
        if command[0] == "git":
            return SimpleNamespace(
                stdout="version\n" if command[1] == "rev-parse" else ""
            )
        calls.append(command)
        if fail_second and len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=str(100 + len(calls)) + ";cluster\n")

    monkeypatch.setattr(cli.subprocess, "run", run)
    assert main(["experiment.yml", "--record-dir", "record", "--dry-run"]) == 0
    assert not (tmp_path / "record").exists()
    assert main(["experiment.yml", "--record-dir", "record"]) == (
        2 if fail_second else 0
    )
    record = json.loads((tmp_path / "record/jobs.json").read_text())
    assert record["jobs"][0]["job_id"] == "101"
    assert record["status"] == ("partial submission" if fail_second else "submitted")
    assert len(record["sources"]) == 5
    assert (tmp_path / "record/submitted/base.yml").read_bytes() == (
        tmp_path / "base.yml"
    ).read_bytes()
    assert main(["experiment.yml", "--record-dir", "record"]) == 2
    assert len(calls) == 2
