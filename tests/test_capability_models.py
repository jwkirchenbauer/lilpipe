"""Multi-model evaluation reuses the established task protocols and fails promptly."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_capability_models.sbatch"
)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("include_harmfulness", [False, True])
def test_multiple_models_and_task_settings(tmp_path, fail, include_harmfulness):
    stub = tmp_path / "scripts/slurm/tamia/eval_capabilities.sbatch"
    stub.parent.mkdir(parents=True)
    stub.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$TEST_LAUNCHER" "$@"\n')
    stub.with_name("eval_harmfulness_and_capabilities.sbatch").write_text(
        stub.read_text()
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], 'tasks': os.environ.get('CAPABILITY_TASKS'),"
        " 'shots': os.environ.get('NUM_FEWSHOT'), 'max_batch': os.environ.get('MAX_BATCH_SIZE'), 'harmfulness_only': os.environ.get('HARMFULNESS_ONLY')}) + '\\n')\n"
        "sys.exit(7 if os.environ['FAIL'] == 'True' else 0)\n"
    )
    calls = tmp_path / "calls.jsonl"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "evaluation output",
            "models/model one",
            "models/model two",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "TEST_PYTHON": sys.executable,
            "TEST_LAUNCHER": str(launcher),
            "CALLS": str(calls),
            "FAIL": str(fail),
            "INCLUDE_HARMFULNESS": "1" if include_harmfulness else "0",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (7 if fail else 0), result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    if fail:
        assert len(records) == 1
        return
    stages = 3 if include_harmfulness else 2
    assert len(records) == 2 * stages
    for index, name in enumerate(("model one", "model two")):
        per_model = records[index * stages : (index + 1) * stages]
        cap, gsm = per_model[:2]
        if include_harmfulness:
            assert per_model[2]["args"] == [
                f"models/{name}",
                f"evaluation output/{name}",
            ]
            assert per_model[2]["harmfulness_only"] == "1"
        assert cap["args"] == [
            f"models/{name}",
            f"evaluation output/{name}/capabilities-4gpu",
        ]
        assert cap["tasks"] == "mmlu,truthfulqa_mc1,arc_easy,hellaswag"
        assert cap["shots"] == "0" and cap["max_batch"] == "128"
        assert gsm["args"] == [
            f"models/{name}",
            f"evaluation output/{name}/gsm8k-5shot",
        ]
        assert gsm["tasks"] == "gsm8k"
        assert gsm["shots"] == "5" and gsm["max_batch"] == "32"
