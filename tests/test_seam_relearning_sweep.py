"""The sweep isolates GPUs, preserves attack settings, and propagates failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/seam_codebase_train_seam_relearning_sweep.sbatch"
)


@pytest.mark.parametrize("learning_rates", [None, ["1e-4", "2e-4"]])
@pytest.mark.parametrize("failure", ["none", "8e-5"])
@pytest.mark.parametrize("model_slug", [None, "llama-base-control"])
@pytest.mark.parametrize("evaluate_baseline", [False, True])
@pytest.mark.parametrize("attack_format", [None, "space"])
def test_sweep_gpu_assignment_checkpoints_and_failures(
    tmp_path, failure, model_slug, evaluate_baseline, learning_rates, attack_format
):
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    (tmp_path / "seam").mkdir()
    baseline_call = tmp_path / "baseline-call"
    baseline = workdir / "scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        '#!/bin/bash\ntest "$INCLUDE_GSM8K" = 1\nprintf "%s\\n" "$*" > "$BASELINE_CALL"\n'
    )
    model = tmp_path / "defended"
    models = tmp_path / "models"
    runs = tmp_path / "runs"
    for directory in (model, models, runs):
        directory.mkdir()
    seam_python = tmp_path / "env/bin/python"
    seam_python.parent.mkdir(parents=True)
    seam_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1] == '-':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "args = sys.argv[2:]\n"
        "if os.environ['EVALUATE_BASELINE'] == '1': assert Path(os.environ['BASELINE_CALL']).exists()\n"
        "def value(flag): return args[args.index(flag) + 1]\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}) + '\\n')\n"
        "if value('--learning_rate') == os.environ['FAILURE']: sys.exit(7)\n"
        "root = Path(value('--output_dir'))\n"
        "steps = [1] if '--max_steps' in args else [50, 100, 150, 200, 250]\n"
        "for step, directory in [(s, root / f'checkpoint-{s}') for s in steps] + [(steps[-1], root)]:\n"
        "    directory.mkdir(parents=True, exist_ok=True)\n"
        "    (directory / 'trainer_state.json').write_text(json.dumps({'global_step': step}))\n"
        "    (directory / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'model.safetensors'}}))\n"
        "    (directory / 'model.safetensors').touch()\n"
        "    (directory / 'tokenizer_config.json').write_text('{}')\n"
    )
    seam_python.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    result = subprocess.run(
        ["bash", str(SCRIPT), str(model), str(models), str(runs), "true"],
        cwd=workdir,
        env={
            **os.environ,
            "SEAM_ENV": str(seam_python.parents[1]),
            "SCRATCH": str(tmp_path),
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",
            "CALLS": str(calls),
            "FAILURE": failure,
            "EVALUATE_BASELINE": "1" if evaluate_baseline else "0",
            "BASELINE_CALL": str(baseline_call),
            **({"MODEL_SLUG": model_slug} if model_slug else {}),
            **({"LEARNING_RATES": " ".join(learning_rates)} if learning_rates else {}),
            "ATTACK_FORMAT": attack_format or "newline",
        },
        capture_output=True,
        text=True,
    )
    rates = learning_rates or ["2e-5", "5e-5", "8e-5", "1e-4", "2e-4"]
    assert result.returncode == (1 if failure in rates else 0), result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == len(rates) + 1
    assert baseline_call.exists() == evaluate_baseline
    if evaluate_baseline:
        assert (
            baseline_call.read_text().strip()
            == f"{model} artifacts/tamia/evals/{model.name}"
        )
    full = [record for record in records if "--max_steps" not in record["args"]]
    expected = dict(zip(rates, ("4", "5", "6", "7", "4")))
    for record in full:
        args = record["args"]
        lr = args[args.index("--learning_rate") + 1]
        assert record["gpu"] == expected[lr]
        slug = model_slug or model.name
        tag = "-answer-space" if attack_format == "space" else ""
        assert Path(args[args.index("--output_dir") + 1]).name == (
            f"{slug}--data-overlapping--relearn-fft-author-codebase{tag}-lr{lr}-ms250"
        )
        assert args[args.index("--attack_response_separator") + 1] == (
            " " if attack_format == "space" else "\n"
        )
        assert args[args.index("--save_strategy") + 1] == "steps"
        assert args[args.index("--save_steps") + 1] == "50"
        assert args[args.index("--save_only_model") + 1] == "True"
        assert args[args.index("--per_device_train_batch_size") + 1] == "4"
        assert args[args.index("--gradient_accumulation_steps") + 1] == "1"
        assert args[args.index("--num_train_epochs") + 1] == "1"
