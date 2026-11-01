import importlib.util
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
SCRIPTS = EXAMPLE / "scripts/slurm/tamia"
MODEL_ID = (
    "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--relearn-fft-ms250"
)


@pytest.fixture
def evaluation():
    spec = importlib.util.spec_from_file_location(
        "tamia_evaluation", EXAMPLE / "scripts/evals/tamia_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(path, adapter=False):
    path.mkdir(parents=True, exist_ok=True)
    (path / ("adapter_config.json" if adapter else "config.json")).write_text("{}")
    return str(path)


def trajectory_args(evaluation, mode="beavertails-trajectory", steps="0,100,200,250"):
    trajectory = f"artifacts/tamia/models/{MODEL_ID}"
    arguments = [mode, MODEL_ID, "baseline", trajectory]
    if mode == "beavertails-trajectory":
        arguments += ["1e-4", "disjoint", "SEAM"]
    else:
        arguments += [trajectory, "-"]
    return evaluation.parse_args(
        arguments
        + [
            "artifacts/tamia/evals",
            steps,
            "--final-step",
            "250",
            "--final-model",
            trajectory,
        ]
    )


def beavertails_results(args):
    scores = {
        "harmfulness": 0.1,
        "capability": 0.6,
        "tasks": {"mmlu": 0.5, "arc_easy": 0.7},
    }
    for step in args.steps:
        path = args.result_root / args.model_id / f"checkpoint-{step}/summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_id": args.model_id,
                    "checkpoint_step": step,
                    "attacked_model": args.trajectory,
                    "baseline_model": args.baseline,
                    "pre": scores,
                    "post": scores,
                }
            )
        )


def test_combined_results_keep_both_task_directories(evaluation, tmp_path):
    source = tmp_path / "combined/model/results.json"
    source.parent.mkdir(parents=True)
    data = {"results": {task: {"acc,none": 0.5} for task, _ in evaluation.WMDP_TASKS}}
    source.write_text(json.dumps(data))
    destination = tmp_path / "results"
    evaluation.publish_wmdp(source.parent.parent, destination)
    for _, folder in evaluation.WMDP_TASKS:
        assert (
            json.loads((destination / folder / "model/results.json").read_text())
            == data
        )


@pytest.mark.parametrize("content", ["{", '{"results": {}}'])
def test_incomplete_combined_run_is_not_published(evaluation, tmp_path, content):
    source = tmp_path / "combined/results.json"
    source.parent.mkdir()
    source.write_text(content)
    destination = tmp_path / "results"
    with pytest.raises(ValueError):
        evaluation.publish_wmdp(source.parent, destination)
    assert not destination.exists()


def test_checkpoint_zero_and_strict_terminal_resolution(evaluation, tmp_path):
    baseline = model(tmp_path / "baseline")
    trajectory = tmp_path / "attack"
    assert evaluation.checkpoint_model(baseline, str(trajectory), 0) == baseline
    zero = model(trajectory / "checkpoint-0")
    assert evaluation.checkpoint_model(baseline, str(trajectory), 0) == zero
    final = model(trajectory)
    with pytest.raises(ValueError, match="missing model checkpoint"):
        evaluation.checkpoint_model(baseline, final, 100)
    assert (
        evaluation.checkpoint_model(
            baseline, final, 250, final_step=250, final_model=final
        )
        == final
    )
    with pytest.raises(ValueError, match="missing model checkpoint"):
        evaluation.checkpoint_model(
            baseline, final, 100, final_step=250, final_model=final
        )


def test_cached_hub_baseline(evaluation, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    cache = tmp_path / "hub/models--org--base"
    (cache / "refs").mkdir(parents=True)
    (cache / "refs/main").write_text("pinned")
    snapshot = model(cache / "snapshots/pinned")
    assert evaluation.checkpoint_model("org/base", "absent", 0) == snapshot


def test_wmdp_trajectory_pairs_each_checkpoint_and_keeps_adapter_zero(
    evaluation, tmp_path, monkeypatch
):
    baseline = model(tmp_path / "baseline")
    trajectory = tmp_path / "adapter"
    zero = model(trajectory / "checkpoint-0", adapter=True)
    final = model(trajectory, adapter=True)
    args = evaluation.parse_args(
        [
            "wmdp-trajectory",
            "model",
            baseline,
            final,
            baseline,
            final,
            str(tmp_path / "evals"),
            "0,250",
            "--final-step",
            "250",
            "--final-model",
            final,
        ]
    )
    points = evaluation.wmdp_points(args)
    assert points == [
        (
            evaluation.model_arguments(baseline, zero),
            tmp_path / "evals/model/checkpoint-0",
        ),
        (
            evaluation.model_arguments(baseline, final),
            tmp_path / "evals/model/checkpoint-250",
        ),
    ]


def test_single_and_final_paths_remain_compatible(evaluation, tmp_path):
    final = model(tmp_path / "trained")
    args = ["wmdp-single", "model", final, "-", str(tmp_path / "evals")]
    single = evaluation.wmdp_points(evaluation.parse_args(args))
    terminal = evaluation.wmdp_points(
        evaluation.parse_args(
            args + ["--checkpoint-step", "125", "--final-model", final]
        )
    )
    assert [single[0][1], terminal[0][1]] == [
        tmp_path / "evals/model",
        tmp_path / "evals/model/checkpoint-125",
    ]


def test_beavertails_terminal_root_keeps_summary_contract(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "beavertails_terminal",
        EXAMPLE / "scripts/evals/evaluate_beavertail_and_capabilities.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *args, **kwargs: [
                {"is_safe": False, "prompt": "question"}
            ]
            * 100
        ),
    )
    scores = {
        "harmfulness": 0.1,
        "capability": 0.6,
        "tasks": {"mmlu": 0.5, "arc_easy": 0.7},
    }
    monkeypatch.setattr(module, "cached_evaluation", lambda *args: scores)
    evaluated = []

    def evaluate(path, prompts):
        evaluated.append(path)
        return scores

    monkeypatch.setattr(module, "evaluate_model", evaluate)
    final = model(tmp_path / "final")
    module.main(
        [
            "trajectory",
            "model",
            "baseline",
            final,
            "1e-4",
            "disjoint",
            "SEAM",
            str(tmp_path / "evals"),
            "--checkpoint-step",
            "250",
            "--checkpoint-path",
            final,
        ]
    )
    summary = json.loads(
        (tmp_path / "evals/model/checkpoint-250/summary.json").read_text()
    )
    assert evaluated == [final]
    assert summary["checkpoint_step"] == 250
    assert summary["post"] == scores
    assert summary["attacked_model"] == final


def test_hpo_checkpoint_metadata_comes_from_each_candidate(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "tamia_hpo", EXAMPLE / "scripts/hpo.py"
    )
    hpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hpo)
    (tmp_path / "base.yml").write_text(
        yaml.safe_dump({"max_steps": 125, "save_steps": 100})
    )
    sweep = tmp_path / "sweep.yml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "id": "vary-steps",
                "producer": {"script": "scripts/slurm/tamia/train.sbatch"},
                "sweeps": [{"name": "length", "parameters": {"max_steps": [125, 250]}}],
                "evaluations": ["tamia-h100-z7b--wmdp-and-mmlu-final"],
            }
        )
    )
    monkeypatch.chdir(EXAMPLE)
    manifest = hpo.generate(sweep, tmp_path / "generated")
    models = yaml.safe_load(manifest.with_name("models.yml").read_text())["models"]
    assert [
        (m["evaluation_final_step"], m["evaluation_checkpoint_steps"])
        for m in models.values()
    ] == [("125", "0,100,125"), ("250", "0,100,200,250")]
    from lilpipe import load

    stages = load(manifest).plan().stages
    assert len(stages) == 4
    evaluations = [stage for stage in stages if stage.id.startswith("eval-")]
    assert [stage.args[5] for stage in evaluations] == ["125", "250"]
    assert all(len(stage.depends_on) == 1 for stage in evaluations)


@pytest.mark.parametrize("failed", [False, True])
def test_beavertails_waves_join_workers_and_include_terminal_step(
    evaluation, tmp_path, monkeypatch, failed
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLURM_GPUS_ON_NODE", "2")
    args = trajectory_args(evaluation)
    model(Path("baseline"))
    for step in (100, 200):
        model(Path(args.trajectory) / f"checkpoint-{step}")
    model(Path(args.trajectory))
    calls = []
    waited = []

    class Worker:
        def __init__(self, command, env):
            self.step = command[command.index("--checkpoint-step") + 1]
            calls.append((self.step, env["CUDA_VISIBLE_DEVICES"]))

        def wait(self):
            waited.append(self.step)
            return int(failed and self.step == "0")

        def poll(self):
            return 0

    monkeypatch.setattr(evaluation.subprocess, "Popen", Worker)
    if failed:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluation.evaluate_beavertails(args)
        assert calls == [("0", "0"), ("100", "1")]
    else:
        evaluation.evaluate_beavertails(args)
        assert calls == [("0", "0"), ("100", "1"), ("200", "0"), ("250", "1")]
    assert waited == [step for step, _ in calls]
    assert Path(args.trajectory).exists()


def test_cleanup_requires_opt_in_and_all_summaries(evaluation, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = trajectory_args(evaluation)
    model(Path(args.trajectory))
    evaluation.delete_trajectory(args)
    assert Path(args.trajectory).exists()
    args.delete_trajectory = True
    with pytest.raises(FileNotFoundError):
        evaluation.delete_trajectory(args)
    beavertails_results(args)
    terminal = args.result_root / args.model_id / "checkpoint-250/summary.json"
    terminal.unlink()
    with pytest.raises(FileNotFoundError):
        evaluation.delete_trajectory(args)
    assert Path(args.trajectory).exists()
    beavertails_results(args)
    evaluation.delete_trajectory(args)
    assert not Path(args.trajectory).exists()
    assert terminal.exists()


@pytest.mark.parametrize("invalid", ["malformed", "identity", "metrics", "step"])
def test_cleanup_rejects_invalid_results(evaluation, tmp_path, monkeypatch, invalid):
    monkeypatch.chdir(tmp_path)
    args = trajectory_args(evaluation)
    model(Path(args.trajectory))
    args.delete_trajectory = True
    beavertails_results(args)
    path = args.result_root / args.model_id / "checkpoint-250/summary.json"
    data = json.loads(path.read_text())
    if invalid == "identity":
        data["model_id"] = "another-model"
    elif invalid == "step":
        data["checkpoint_step"] = 200
    else:
        data["post"]["tasks"].pop("arc_easy")
    path.write_text("{" if invalid == "malformed" else json.dumps(data))
    with pytest.raises(ValueError):
        evaluation.delete_trajectory(args)
    assert Path(args.trajectory).exists()


@pytest.mark.parametrize(
    "invalid", ["id", "outside", "symlink", "child-symlink", "baseline", "terminal"]
)
def test_cleanup_rejects_unsafe_paths(evaluation, tmp_path, monkeypatch, invalid):
    monkeypatch.chdir(tmp_path)
    args = trajectory_args(evaluation)
    path = Path(args.trajectory)
    model(path)
    beavertails_results(args)
    args.delete_trajectory = True
    if invalid == "id":
        args.model_id = "other"
    elif invalid == "outside":
        args.trajectory = "somewhere/" + args.model_id
    elif invalid == "baseline":
        args.baseline = args.trajectory
    elif invalid == "terminal":
        args.steps = (0, 100, 200)
    else:
        outside = tmp_path / "outside"
        model(outside)
        if invalid == "symlink":
            (path / "config.json").unlink()
            path.rmdir()
            path.symlink_to(outside, target_is_directory=True)
        else:
            (path / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        evaluation.delete_trajectory(args)
    assert path.exists()


def test_generated_cleanup_checks_candidate_identity(evaluation, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = trajectory_args(evaluation)
    args.model_id = "sweep--relearn-fft--001"
    args.trajectory = f"artifacts/generated/hpo/sweep/models/{args.model_id}"
    model(Path(args.trajectory))
    args.delete_trajectory = True
    beavertails_results(args)
    evaluation.delete_trajectory(args)
    assert not Path(args.trajectory).exists()


def test_wmdp_cleanup_requires_both_tasks_and_matching_model(
    evaluation, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = trajectory_args(evaluation, "wmdp-trajectory", "0,250")
    args.model_id = sorted(evaluation.WMDP_MODELS)[0]
    args.trajectory = f"artifacts/tamia/models/{args.model_id}"
    args.model = args.final_model = args.trajectory
    model(Path(args.trajectory))
    model(Path("baseline"))
    args.delete_trajectory = True
    for step in args.steps:
        for task, folder in evaluation.WMDP_TASKS:
            path = (
                args.result_root
                / args.model_id
                / f"checkpoint-{step}"
                / folder
                / "results.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pretrained = "baseline" if step == 0 else args.trajectory
            path.write_text(
                json.dumps(
                    {
                        "config": {"model_args": {"pretrained": pretrained}},
                        "results": {task: {"acc,none": 0.5}},
                    }
                )
            )
    path.unlink()
    with pytest.raises(ValueError, match="missing mmlu_no_bio"):
        evaluation.delete_trajectory(args)
    path.write_text(
        json.dumps(
            {
                "config": {"model_args": "pretrained=wrong"},
                "results": {task: {"acc,none": 0.5}},
            }
        )
    )
    with pytest.raises(ValueError, match="incomplete mmlu_no_bio"):
        evaluation.delete_trajectory(args)
    path.write_text(
        json.dumps(
            {
                "config": {"model_args": f"pretrained={pretrained}"},
                "results": {task: {"acc,none": 0.5}},
            }
        )
    )
    evaluation.delete_trajectory(args)
    assert not Path(args.trajectory).exists()


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("single_gpu", [False, True])
def test_train_wrapper_forwards_launcher_and_merges_only_after_success(
    tmp_path, fail, single_gpu
):
    local_bin = tmp_path / ".local/bin"
    local_bin.mkdir(parents=True)
    calls = tmp_path / "calls"
    for command, body in {
        "uv": "exit 0\n",
        "axolotl": 'printf "%s|%s\\n" "${CUDA_VISIBLE_DEVICES:-all}" "$*" >> "$CALLS"\nif [[ $1 == train && $FAIL_TRAIN == 1 ]]; then exit 1; fi\n',
    }.items():
        path = local_bin / command
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { :; }\n")
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "SCRATCH": str(tmp_path / "scratch"),
        "SLURM_TMPDIR": str(tmp_path / "job"),
        "SLURM_JOB_ID": "123",
        "CALLS": str(calls),
        "BASH_ENV": str(bash_env),
        "FAIL_TRAIN": str(int(fail)),
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    }
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "train.sbatch"),
            "recipe.yml",
            *(["--single-gpu"] if single_gpu else []),
            "--merge",
            "--launcher",
            "python",
            "--seed",
            "42",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert (completed.returncode != 0) == fail, completed.stderr
    visible = "0" if single_gpu else "0,1,2,3"
    assert calls.read_text().splitlines() == [
        f"{visible}|train recipe.yml --launcher python --seed 42"
    ] + ([] if fail else [f"{visible}|merge-lora recipe.yml"])


@pytest.mark.parametrize(
    "wrapper",
    [
        "eval_wmdp_mmlu_trajectory.sbatch",
        "eval_beavertails_capabilities_trajectory.sbatch",
    ],
)
def test_trajectory_wrapper_never_reaches_cleanup_after_failure(tmp_path, wrapper):
    local_bin = tmp_path / ".local/bin"
    local_bin.mkdir(parents=True)
    for command, body in {
        "uv": "exit 0\n",
        "python": 'printf "%s\\n" "$*" >> "$CALLS"\nexit 7\n',
    }.items():
        path = local_bin / command
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { :; }\n")
    calls = tmp_path / "calls"
    result = subprocess.run(
        ["bash", str(SCRIPTS / wrapper), "model", "--delete-trajectory"],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "SCRATCH": str(tmp_path / "scratch"),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
            "BASH_ENV": str(bash_env),
            "CALLS": str(calls),
        },
    )
    assert result.returncode == 7
    assert len(calls.read_text().splitlines()) == 1
    assert "--cleanup-only" not in calls.read_text()


@pytest.mark.parametrize("trajectory", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_wmdp_wrappers_launch_both_tasks_once_per_checkpoint(
    tmp_path, trajectory, fail
):
    local_bin = tmp_path / ".local/bin"
    local_bin.mkdir(parents=True)
    (local_bin / "uv").write_text("#!/bin/bash\nexit 0\n")
    (local_bin / "python").symlink_to(sys.executable)
    accelerate = local_bin / "accelerate"
    accelerate.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "arguments = sys.argv[1:]\n"
        "with open(os.environ['CALLS'], 'a') as handle: handle.write(json.dumps(arguments) + '\\n')\n"
        "if os.environ['FAIL_EVAL'] == '1': sys.exit(7)\n"
        "output = Path(arguments[arguments.index('--output_path') + 1]) / 'model/results.json'\n"
        "output.parent.mkdir(parents=True, exist_ok=True)\n"
        "model_args = arguments[arguments.index('--model_args') + 1]\n"
        "output.write_text(json.dumps({'config': {'model_args': model_args}, 'results': {task: {'acc,none': 0.5} for task in ('mmlu_no_bio', 'wmdp_bio_robust')}}))\n"
    )
    for command in (local_bin / "uv", accelerate):
        command.chmod(0o755)
    (tmp_path / "scripts/evals").mkdir(parents=True)
    (tmp_path / "scripts/evals/tamia_evaluation.py").symlink_to(
        EXAMPLE / "scripts/evals/tamia_evaluation.py"
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    bash_env = tmp_path / "bash-env"
    bash_env.write_text("module() { :; }\n")
    baseline = model(tmp_path / "baseline")
    attack = model(tmp_path / "attack")
    calls = tmp_path / "calls"
    output = tmp_path / "evals"
    if trajectory:
        arguments = [
            "candidate",
            baseline,
            attack,
            attack,
            "-",
            str(output),
            "0,250",
            "--final-step",
            "250",
            "--final-model",
            attack,
        ]
        wrapper = "eval_wmdp_mmlu_trajectory.sbatch"
    else:
        arguments = ["candidate", baseline, "-", str(output)]
        wrapper = "eval_wmdp_mmlu.sbatch"
    result = subprocess.run(
        ["bash", str(SCRIPTS / wrapper), *arguments],
        cwd=tmp_path,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "SCRATCH": str(tmp_path / "scratch"),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
            "SLURM_GPUS_ON_NODE": "4",
            "BASH_ENV": str(bash_env),
            "CALLS": str(calls),
            "FAIL_EVAL": str(int(fail)),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (7 if fail else 0), result.stderr
    launches = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(launches) == (2 if trajectory and not fail else 1)
    for command in launches:
        assert command[:6] == [
            "launch",
            "--num_processes=4",
            "-m",
            "lm_eval",
            "run",
            "--model",
        ]
        assert command[command.index("--tasks") + 1] == "mmlu_no_bio,wmdp_bio_robust"
        assert command[command.index("--batch_size") + 1] == "32"
    assert Path(attack).is_dir()
    if fail:
        assert not output.exists()
    else:
        destinations = (
            [output / "candidate" / f"checkpoint-{step}" for step in (0, 250)]
            if trajectory
            else [output / "candidate"]
        )
        for destination in destinations:
            assert (destination / "wmdp-bio-robust/model/results.json").is_file()
            assert (destination / "mmlu-no-bio/model/results.json").is_file()


def test_six_scripts_and_seven_descriptive_registry_entries():
    assert {path.name for path in SCRIPTS.glob("*.sbatch")} == {
        "train.sbatch",
        "analysis_gen_ll.sbatch",
        "eval_wmdp_mmlu.sbatch",
        "eval_wmdp_mmlu_trajectory.sbatch",
        "eval_beavertails_capabilities.sbatch",
        "eval_beavertails_capabilities_trajectory.sbatch",
    }
    for path in SCRIPTS.glob("*.sbatch"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    registry = yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
        "evaluations"
    ]
    assert len(registry) == 7
    assert sum("/tamia/" in entry["script"] for entry in registry.values()) == 5
    assert all(
        name.startswith(
            ("mila-l40s-q06b--", "tamia-h100-z7b--", "tamia-h100-llama2-7b--")
        )
        for name in registry
    )
    assert not any(
        "minicpm" in str(path).lower() for path in EXAMPLE.glob("configs/**/*")
    )
    models = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]
    assert not any("minicpm" in name for name in models)
