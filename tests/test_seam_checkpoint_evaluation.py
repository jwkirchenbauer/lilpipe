"""Evaluate unique checkpoints and reuse only verified identical final weights."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch"
)


@pytest.mark.parametrize("identical", [True, False])
def test_checkpoint_evaluation_and_final_reuse(tmp_path, identical):
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    wrapper = tmp_path / "seam/scripts/eval_harmfulness.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()
    model = tmp_path / "model"
    for directory in (model, model / "checkpoint-250"):
        directory.mkdir(parents=True, exist_ok=True)
        index = {"weight_map": {"weight": "model.safetensors"}}
        (directory / "model.safetensors.index.json").write_text(json.dumps(index))
        for name in (
            "model.safetensors",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            (directory / name).write_text("identical")
    if not identical:
        (model / "model.safetensors").write_text("different")
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[2:]\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'stage': sys.argv[1], 'args': args, 'cache': os.environ['HF_DATASETS_CACHE']}) + '\\n')\n"
        "if sys.argv[1] == 'capabilities':\n"
        "    output = Path(args[args.index('--output_path') + 1])\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'results.json').write_text('{}')\n"
        "else:\n"
        "    output = Path(args[args.index('--output') + 1])\n"
        "    output.write_text(json.dumps({'model': args[args.index('--batch-size') - 1], 'harmfulness': 0.1}))\n"
    )
    seam_python = tmp_path / "env/bin/python"
    seam_python.parent.mkdir(parents=True)
    seam_python.write_text(
        '#!/bin/bash\nif [[ $1 == - ]]; then exec "$TEST_PYTHON" "$@"; fi\n'
        'exec "$TEST_PYTHON" "$TEST_LAUNCHER" harmfulness "$@"\n'
    )
    seam_python.chmod(0o755)
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        "uv() { :; }\n"
        'accelerate() { "$TEST_PYTHON" "$TEST_LAUNCHER" capabilities "$@"; }\n'
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    calls = tmp_path / "calls.jsonl"
    output = tmp_path / "evals"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--checkpoints", str(model), str(output)],
        cwd=workdir,
        env={
            **os.environ,
            "SCRATCH": str(tmp_path),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
            "SEAM_ENV": str(seam_python.parents[1]),
            "BASH_ENV": str(bash_env),
            "TEST_PYTHON": sys.executable,
            "TEST_LAUNCHER": str(launcher),
            "CALLS": str(calls),
        },
        capture_output=True,
        text=True,
    )
    if not identical:
        assert result.returncode != 0
        assert not calls.exists()
        return
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [record["stage"] for record in records] == [
        "capabilities",
        "harmfulness",
    ] * 5
    for index, step in enumerate(range(50, 251, 50)):
        cap, harm = records[index * 2 : index * 2 + 2]
        assert cap["cache"] == str(tmp_path / ".cache/huggingface/datasets")
        assert harm["cache"] == str(tmp_path / ".cache/huggingface-seam-datasets-3.3.1")
        assert str(model / f"checkpoint-{step}") in harm["args"]
        assert (output / f"checkpoint-{step}/harmfulness.json").is_file()
    assert (output / "capabilities/results.json").is_file()
    report = json.loads((output / "harmfulness.json").read_text())
    assert report["model"] == str(model)
    assert report["evaluated_model"] == str(model / "checkpoint-250")
    assert report["reused_from"] == "checkpoint-250/harmfulness.json"
