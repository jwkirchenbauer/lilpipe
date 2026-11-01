import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/tamper-resistance"
SCRIPTS = EXAMPLE / "scripts/slurm/mila"


@pytest.fixture
def job(tmp_path):
    """Run the real shell wrappers with local stand-ins for training/evaluation."""
    local_bin = tmp_path / ".local/bin"
    local_bin.mkdir(parents=True)
    calls = tmp_path / "calls.jsonl"
    record = (
        "import json, os, sys\nfrom pathlib import Path\n"
        "def record(value):\n"
        "    with open(os.environ['CALLS'], 'a') as handle:\n"
        "        handle.write(json.dumps(value) + '\\n')\n"
    )
    programs = {
        "uv": record
        + "record({'command': 'uv', 'args': sys.argv[1:], 'cache_dir': os.environ['UV_CACHE_DIR'], 'offline': os.environ.get('UV_OFFLINE')})\n"
        + "sys.exit(int(os.environ.get('FAIL_SYNC', '0')))\n",
        "axolotl": record
        + "record({'command': 'axolotl', 'args': sys.argv[1:], 'wandb_dir': os.environ['WANDB_DIR']})\n",
    }
    for name, body in programs.items():
        path = local_bin / name
        path.write_text(f"#!{sys.executable}\n" + body)
        path.chmod(0o755)
    (local_bin / "python").symlink_to(sys.executable)
    scripts = tmp_path / "scripts/evals"
    scripts.mkdir(parents=True)
    (scripts / "evaluate_beavertail_and_capabilities.py").write_text(
        record
        + "args = sys.argv[1:]\n"
        + "record({'command': 'evaluator', 'args': args})\n"
        + "step = args[args.index('--checkpoint-step') + 1] if args[0] == 'trajectory' else ''\n"
        + "if step and step == os.environ.get('FAIL_STEP'): sys.exit(7)\n"
        + "output = Path(args[7] if step else args[4]) / args[1]\n"
        + "if step: output /= 'checkpoint-' + step\n"
        + "output.mkdir(parents=True, exist_ok=True)\n"
        + "(output / 'summary.json').write_text(json.dumps({'model_id': args[1], 'checkpoint_step': int(step) if step else None, 'post': {'harmfulness': 0.1, 'capability': 0.6, 'tasks': {'mmlu': 0.5, 'arc_easy': 0.7}}}))\n"
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "SCRATCH": str(tmp_path / "scratch"),
        "SLURM_TMPDIR": str(tmp_path / "job"),
        "SLURM_JOB_ID": "123",
        "CALLS": str(calls),
    }
    env.pop("UV_OFFLINE", None)

    def run(script, arguments, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPTS / script), *arguments],
            cwd=tmp_path,
            env={**env, **overrides},
            capture_output=True,
            text=True,
        )

    def events():
        return [json.loads(line) for line in calls.read_text().splitlines()]

    return SimpleNamespace(
        root=tmp_path,
        run=run,
        events=events,
    )


def trajectory(job):
    path = job.root / "models/candidate"
    for directory in (path, path / "checkpoint-100", path / "checkpoint-200"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}")
    # A matching non-directory must not be evaluated or removed.
    (path / "checkpoint-not-a-directory").write_text("keep")
    arguments = [
        "candidate",
        "baseline",
        str(path),
        "1e-4",
        "disjoint",
        "SEAM legacy",
        str(job.root / "results"),
    ]
    return path, arguments


def test_training_uses_python_launcher_and_job_local_cache(job):
    result = job.run("train.sbatch", ["recipe.yml"])
    assert result.returncode == 0, result.stderr
    install, train = job.events()
    assert install == {
        "command": "uv",
        "args": ["sync", "--frozen", "--group", "train"],
        "cache_dir": str(job.root / "job/.cache/uv-123"),
        "offline": None,
    }
    assert train["args"] == ["train", "recipe.yml", "--launcher", "python"]
    assert train["wandb_dir"] == str(job.root / "artifacts/mila/logs")


def test_failed_install_does_not_train(job):
    result = job.run("train.sbatch", ["recipe.yml"], FAIL_SYNC="6")
    assert result.returncode == 6
    assert [event["command"] for event in job.events()] == ["uv"]


def test_single_evaluation_passes_metadata_and_preserves_summary_path(job):
    output = job.root / "results"
    arguments = ["candidate", "model", "SEAM legacy", str(output)]
    result = job.run("eval_beavertails_capabilities.sbatch", arguments)
    assert result.returncode == 0, result.stderr
    install, evaluation = job.events()
    assert install["args"][-1] == "eval"
    assert install["cache_dir"] == str(job.root / "job/.cache/uv-123")
    assert install["offline"] is None
    assert evaluation["args"] == ["single", *arguments]
    summary = json.loads((output / "candidate/summary.json").read_text())
    assert summary["post"]["tasks"] == {"mmlu": 0.5, "arc_easy": 0.7}


@pytest.mark.parametrize("flags", [[], ["true"], ["false", "true"], ["true", "true"]])
def test_trajectory_discovers_directories_and_removes_only_when_requested(job, flags):
    path, arguments = trajectory(job)
    result = job.run(
        "eval_beavertails_capabilities_trajectory.sbatch", arguments + flags
    )
    assert result.returncode == 0, result.stderr
    evaluations = [
        event["args"] for event in job.events() if event["command"] == "evaluator"
    ]
    assert len(evaluations) == 2
    assert [args[args.index("--checkpoint-step") + 1] for args in evaluations] == [
        "100",
        "200",
    ]
    for args, step in zip(evaluations, (100, 200), strict=True):
        assert args[:8] == ["trajectory", *arguments]
        assert args[args.index("--checkpoint-path") + 1] == str(
            path / f"checkpoint-{step}"
        )
        assert (
            job.root / f"results/candidate/checkpoint-{step}/summary.json"
        ).is_file()
    assert not (job.root / "results/candidate/checkpoint-0").exists()
    assert not (job.root / "results/candidate/checkpoint-250").exists()
    if len(flags) == 2 and flags[1] == "true":
        assert not path.exists()
    else:
        assert path.is_dir()
        assert (path / "checkpoint-not-a-directory").read_text() == "keep"
        assert (path / "checkpoint-100").exists() == (flags != ["true"])
        assert (path / "checkpoint-200").exists() == (flags != ["true"])


@pytest.mark.parametrize("step", ["100", "200"])
def test_failed_checkpoint_stops_before_its_deletion_and_final_removal(job, step):
    path, arguments = trajectory(job)
    result = job.run(
        "eval_beavertails_capabilities_trajectory.sbatch",
        arguments + ["true", "true"],
        FAIL_STEP=step,
    )
    assert result.returncode == 7
    assert path.is_dir()
    assert (path / f"checkpoint-{step}").is_dir()
    if step == "100":
        assert (path / "checkpoint-200").is_dir()
    else:
        assert not (path / "checkpoint-100").exists()


def test_empty_trajectory_does_not_evaluate_the_literal_glob(job):
    path = job.root / "models/candidate"
    path.mkdir(parents=True)
    result = job.run(
        "eval_beavertails_capabilities_trajectory.sbatch",
        [
            "candidate",
            "baseline",
            str(path),
            "1e-4",
            "disjoint",
            "SEAM",
            str(job.root / "results"),
        ],
    )
    assert result.returncode == 0, result.stderr
    assert [event["command"] for event in job.events()] == ["uv"]
    assert path.is_dir()


def test_mila_scripts_and_registry_names_existing_files():
    assert {path.name for path in SCRIPTS.glob("*.sbatch")} == {
        "train.sbatch",
        "eval_beavertails_capabilities.sbatch",
        "eval_beavertails_capabilities_trajectory.sbatch",
        "analysis_get_param_norm.sbatch",
    }
    for path in SCRIPTS.glob("*.sbatch"):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "#SBATCH --gres=gpu:l40s:1" in text
        if path.name == "analysis_get_param_norm.sbatch":
            assert "#SBATCH --cpus-per-task=4" in text
            assert "#SBATCH --mem=64G" in text
        else:
            assert "#SBATCH --cpus-per-task=8" in text
            assert "#SBATCH --mem=32G" in text
        assert "tamia_evaluation.py" not in text
    registry = yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
        "evaluations"
    ]
    for entry in registry.values():
        assert (EXAMPLE / entry["script"]).is_file()
    entry = registry["mila-l40s-q06b--beavertails-and-capabilities-trajectory"]
    assert len(entry["args"]) == 7
    inventory = (EXAMPLE / "docs/mila-refactor.md").read_text()
    assert inventory.count("### `") == 18
