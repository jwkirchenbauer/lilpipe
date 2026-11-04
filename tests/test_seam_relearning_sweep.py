"""The sweep isolates GPUs, preserves attack settings, and propagates failures."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/train_seam_relearning_sweep.sbatch"
)


@pytest.mark.parametrize("failure", ["none", "8e-5"])
def test_sweep_gpu_assignment_checkpoints_and_failures(tmp_path, failure):
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    (tmp_path / "seam").mkdir()
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
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (0 if failure == "none" else 1), result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 6
    full = [record for record in records if "--max_steps" not in record["args"]]
    expected = dict(
        zip(("2e-5", "5e-5", "8e-5", "1e-4", "2e-4"), ("4", "5", "6", "7", "4"))
    )
    for record in full:
        args = record["args"]
        lr = args[args.index("--learning_rate") + 1]
        assert record["gpu"] == expected[lr]
        assert args[args.index("--save_strategy") + 1] == "steps"
        assert args[args.index("--save_steps") + 1] == "50"
        assert args[args.index("--save_only_model") + 1] == "True"
        assert args[args.index("--per_device_train_batch_size") + 1] == "4"
        assert args[args.index("--gradient_accumulation_steps") + 1] == "1"
        assert args[args.index("--num_train_epochs") + 1] == "1"
