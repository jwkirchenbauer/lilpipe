"""Standalone evaluation preserves multi-model protocols and failure propagation."""

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


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("include_harmfulness", [False, True])
def test_multiple_models_and_task_settings(tmp_path, fail, include_harmfulness):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "with open(os.environ['CALLS'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if os.environ['FAIL']=='True': sys.exit(7)\n"
        "if sys.argv[1]=='capabilities':\n"
        " p=Path(sys.argv[sys.argv.index('--output_path')+1]);p.mkdir(parents=True,exist_ok=True)\n"
        " (p/'results_2026.json').write_text('{}')\n"
    )
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        "uv() { :; }\n"
        'accelerate() { "$TEST_PYTHON" "$TEST_LAUNCHER" capabilities "$@"; }\n'
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    seam_python = tmp_path / "env/bin/python"
    seam_python.parent.mkdir(parents=True)
    seam_python.write_text(
        '#!/bin/bash\nexec "$TEST_PYTHON" "$TEST_LAUNCHER" harmfulness "$@"\n'
    )
    seam_python.chmod(0o755)
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    wrapper = tmp_path / "seam/scripts/eval_harmfulness.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.touch()
    calls = tmp_path / "calls.jsonl"
    arguments = [
        argument
        for name in ["model one", "model two"]
        for argument in [f"models/{name}", f"evaluation output/{name}"]
    ]
    result = subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        cwd=workdir,
        env={
            **os.environ,
            "BASH_ENV": str(bash_env),
            "TEST_PYTHON": sys.executable,
            "TEST_LAUNCHER": str(launcher),
            "SEAM_ENV": str(seam_python.parents[1]),
            "SCRATCH": str(tmp_path),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
            "CALLS": str(calls),
            "FAIL": str(fail),
            "INCLUDE_HARMFULNESS": "1" if include_harmfulness else "0",
            "INCLUDE_GSM8K": "1",
            "CAPABILITIES_SUBDIR": "capabilities-4gpu",
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
    for index, name in enumerate(["model one", "model two"]):
        cap, gsm = records[index * stages : index * stages + 2]
        for call, tasks, shots, maximum, folder in [
            (
                cap,
                "mmlu,truthfulqa_mc1,arc_easy,hellaswag",
                "0",
                "128",
                "capabilities-4gpu",
            ),
            (gsm, "gsm8k", "5", "32", "gsm8k-5shot"),
        ]:
            assert call[call.index("--tasks") + 1] == tasks
            assert call[call.index("--num_fewshot") + 1] == shots
            assert call[call.index("--max_batch_size") + 1] == maximum
            assert (
                call[call.index("--output_path") + 1]
                == f"evaluation output/{name}/{folder}"
            )
            assert f"pretrained=models/{name}," in call[call.index("--model_args") + 1]
        if include_harmfulness:
            harm = records[index * stages + 2]
            assert harm[0] == "harmfulness"
            assert (
                harm[harm.index("--output") + 1]
                == f"evaluation output/{name}/harmfulness.json"
            )
