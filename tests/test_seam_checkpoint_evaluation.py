"""Evaluate unique checkpoints and reuse only verified identical final weights."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_harmfulness_and_capabilities.sbatch"
)


@pytest.mark.parametrize("identical", [True, False])
@pytest.mark.parametrize("include_gsm8k", [False, True])
@pytest.mark.parametrize("save_results", [False, True])
def test_checkpoint_evaluation_and_final_reuse(
    tmp_path, identical, include_gsm8k, save_results
):
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    shared = workdir / "scripts/slurm/tamia/eval_capabilities.sbatch"
    shared.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT.with_name("eval_capabilities.sbatch"), shared)
    wrapper = tmp_path / "seam/scripts/eval_harmfulness.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()
    model = tmp_path / "model"
    for step in range(50, 251, 50):
        (model / f"checkpoint-{step}").mkdir(parents=True)
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
        "    if os.environ['SAVE_RESULTS'] == '1':\n"
        "        (output / 'results_2026.json').write_text('{}')\n"
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
            "INCLUDE_GSM8K": "1" if include_gsm8k else "0",
            "SAVE_RESULTS": "1" if save_results else "0",
        },
        capture_output=True,
        text=True,
    )
    if not identical:
        assert result.returncode != 0
        assert not calls.exists()
        return
    if not save_results:
        assert result.returncode != 0
        assert "lm-eval did not save results" in result.stderr
        assert not (output / "final-evaluation.json").exists()
        return
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    stages = (
        ["capabilities", "capabilities", "harmfulness"]
        if include_gsm8k
        else ["capabilities", "harmfulness"]
    )
    assert [record["stage"] for record in records] == stages * 5
    for index, step in enumerate(range(50, 251, 50)):
        per_step = records[index * len(stages) : (index + 1) * len(stages)]
        cap, harm = per_step[0], per_step[-1]
        assert cap["cache"] == str(tmp_path / ".cache/huggingface/datasets")
        assert harm["cache"] == str(tmp_path / ".cache/huggingface-seam-datasets-3.3.1")
        assert str(model / f"checkpoint-{step}") in harm["args"]
        model_args = cap["args"][cap["args"].index("--model_args") + 1]
        assert f"pretrained=../model/checkpoint-{step}," in model_args
        assert (output / f"checkpoint-{step}/harmfulness.json").is_file()
        if include_gsm8k:
            gsm = per_step[1]["args"]
            assert gsm[gsm.index("--tasks") + 1] == "gsm8k"
            assert gsm[gsm.index("--num_fewshot") + 1] == "5"
            assert gsm[gsm.index("--max_batch_size") + 1] == "32"
            assert (
                output / f"checkpoint-{step}/gsm8k-5shot/results_2026.json"
            ).is_file()
    if include_gsm8k:
        assert (output / "gsm8k-5shot/results_2026.json").is_file()
    assert (output / "capabilities/results_2026.json").is_file()
    report = json.loads((output / "harmfulness.json").read_text())
    assert report["model"] == str(model)
    assert report["evaluated_model"] == str(model / "checkpoint-250")
    assert report["reused_from"] == "checkpoint-250/harmfulness.json"
