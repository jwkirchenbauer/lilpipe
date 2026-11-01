import importlib.util
import itertools
import json
from pathlib import Path
import subprocess

import yaml

from lilpipe import load, load_results

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
NAME = "q06b-wmdp-bio--5pp--unlearn-fft-me"
DIRECTORY = EXAMPLE / "configs/stages-wmdp/0_unlearn" / NAME


def test_q06b_me_grid_plans_fifteen_mila_training_evaluation_pairs(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(EXAMPLE)
    spec = importlib.util.spec_from_file_location(
        "q06b_hpo", EXAMPLE / "scripts/hpo.py"
    )
    hpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hpo)
    manifest = hpo.generate(DIRECTORY / "sweep.yml", tmp_path)
    configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "configs").glob("*.yml"))
    ]
    assert {
        (config["learning_rate"], config["custom_trainer_args"]["retain_coefficient"])
        for config in configs
    } == set(itertools.product((1e-6, 5e-6, 1e-5, 2e-5, 1e-4), (0.25, 1, 4)))
    assert len(configs) == 15
    added = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "configs").glob("*--grid-002--*.yml"))
    ]
    assert [
        (config["learning_rate"], config["custom_trainer_args"]["retain_coefficient"])
        for config in added
    ] == [(1e-5, 0.25), (1e-5, 1), (1e-5, 4)]
    for config in configs:
        assert config["base_model"] == "Qwen/Qwen3-0.6B"
        assert config["trainer_cls"] == "trainers.ul_me.MaxEntTrainer"
        assert config["custom_trainer_args"]["forget_coefficient"] == 1
        assert config["custom_trainer_args"]["sampling"] == "optml"
        assert config["max_steps"] == config["save_steps"] == 125
        assert config["micro_batch_size"] == 8
        assert "device_map" not in config
        assert "artifacts/mila/cache/" in config["dataset_prepared_path"]

    pipeline = load(manifest)
    assert len(pipeline.plan().stages) == 30
    for candidate in pipeline.selected_models:
        training, evaluation = pipeline.select(models=[candidate]).plan().stages
        assert training.script == "scripts/slurm/mila/train.sbatch"
        assert evaluation.script == "scripts/slurm/mila/eval_wmdp_mmlu.sbatch"
        assert evaluation.depends_on == (training.id,)
        assert evaluation.args[1] == f"artifacts/mila/evals/{candidate}/checkpoint-125"
        assert "--gres=gpu:l40s:1" in training.sbatch_args
        assert "--gres=gpu:l40s:1" in evaluation.sbatch_args
    assert not (DIRECTORY / "selected.yml").exists()


def test_q06b_baseline_evaluates_without_a_producer(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    (stage,) = load("configs/experiments/q06b-wmdp-bio-base.yml").plan().stages
    assert stage.args == ("Qwen/Qwen3-0.6B", "artifacts/mila/evals/q06b")
    assert not stage.depends_on
    assert stage.script == "scripts/slurm/mila/eval_wmdp_mmlu.sbatch"


def test_q06b_evaluation_script_passes_matching_protocol(tmp_path):
    script = EXAMPLE / "scripts/slurm/mila/eval_wmdp_mmlu.sbatch"
    subprocess.run(["bash", "-n", script], check=True)
    # Execute the application section with a stub environment and evaluator.
    text = script.read_text()
    application = text[text.index("base_model_name_or_path=") :]
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin/activate").touch()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nuv() { :; }\npython() { printf "%s\\n" "$@"; }\n'
            'UV_PROJECT_ENVIRONMENT="$1"\nshift\n' + application,
            "eval",
            str(env),
            "model path",
            "output path",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [
        "-m",
        "lm_eval",
        "run",
        "--model",
        "hf",
        "--model_args",
        "pretrained=model path,dtype=bfloat16,trust_remote_code=True",
        "--include_path",
        "lm_eval_tasks",
        "--tasks",
        "mmlu_no_bio,wmdp_bio_robust",
        "--num_fewshot",
        "0",
        "--batch_size",
        "32",
        "--output_path",
        "output path",
    ]


def test_q06b_results_reads_group_metrics_for_base_and_sweep(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = yaml.safe_load((DIRECTORY / "results.yml").read_text())
    assert len(config["rows"]) == 16
    for row in config["rows"]:
        root = tmp_path / row["root"] / "model"
        root.mkdir(parents=True)
        (root / "results_test.json").write_text(
            json.dumps(
                {
                    "results": {
                        "wmdp_bio_robust": {"acc,none": 0.3},
                        "mmlu_no_bio": {"acc,none": 0.5},
                    }
                }
            )
        )
    table = load_results(DIRECTORY / "results.yml").collect()
    assert len(table.rows) == 16
    assert all(row["wmdp"].value == 0.3 for row in table.rows.values())
    assert all(row["mmlu"].value == 0.5 for row in table.rows.values())
