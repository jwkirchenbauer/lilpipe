"""The standalone WMDP trajectory interface preserves baseline and cleanup semantics."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_wmdp_mmlu.sbatch"
)


@pytest.mark.parametrize("cleanup", ["false", "true"])
def test_trajectory_evaluates_baseline_without_deleting_it(tmp_path, cleanup):
    baseline, trajectory, output = [
        tmp_path / name for name in ["defense", "attack", "evals"]
    ]
    baseline.mkdir()
    for step in (16, 128):
        (trajectory / f"checkpoint-{step}").mkdir(parents=True)
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import os,sys,json\nfrom pathlib import Path\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['CALLS'],'a') as f:f.write(json.dumps(args)+'\\n')\n"
        "p=Path(args[args.index('--output_path')+1]);p.mkdir(parents=True,exist_ok=True)\n"
        "model=args[args.index('--model_args')+1]\n"
        "(p/'results_2026.json').write_text(json.dumps({'config':{'model_args':model,'limit':None},'results':{t:{'acc,none':0.5} for t in ['mmlu_no_bio','wmdp_bio_robust']}}))\n"
    )
    commands = tmp_path / "commands"
    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        "module() { :; }\nuv() { :; }\n"
        'python() { "$TEST_PYTHON" "$@"; }\n'
        'accelerate() { "$TEST_PYTHON" "$TEST_LAUNCHER" "$@"; }\n'
    )
    activate = tmp_path / "job/.venv-123/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.write_text(":\n")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--checkpoints",
            str(trajectory),
            str(output),
            cleanup,
            "false",
            str(baseline),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "CALLS": str(commands),
            "BASH_ENV": str(bash_env),
            "TEST_PYTHON": sys.executable,
            "TEST_LAUNCHER": str(launcher),
            "SCRATCH": str(tmp_path),
            "SLURM_TMPDIR": str(tmp_path / "job"),
            "SLURM_JOB_ID": "123",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    import json

    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    assert [call[call.index("--output_path") + 1] for call in calls] == [
        str(output / f"checkpoint-{step}") for step in (0, 16, 128)
    ]
    assert "pretrained=defense,dtype=bfloat16,trust_remote_code=True" in calls[0]
    assert baseline.is_dir()
    for step in (16, 128):
        assert (trajectory / f"checkpoint-{step}").exists() == (cleanup == "false")
