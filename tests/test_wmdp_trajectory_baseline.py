import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/slurm/tamia/eval_wmdp_mmlu_trajectory.sbatch"
)


@pytest.mark.parametrize("cleanup", ["false", "true"])
def test_trajectory_evaluates_baseline_without_deleting_it(tmp_path, cleanup):
    baseline = tmp_path / "defense"
    trajectory = tmp_path / "attack"
    output = tmp_path / "evals"
    baseline.mkdir()
    for step in (16, 128):
        (trajectory / f"checkpoint-{step}").mkdir(parents=True)
    commands = tmp_path / "commands"
    body = SCRIPT.read_text().split("base_model_name_or_path=", 1)[1]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\naccelerate() { printf "%s\\n" "$@" >> "$CALLS"; }\n'
            + "base_model_name_or_path="
            + body,
            "test",
            str(trajectory),
            str(output),
            cleanup,
            "false",
            str(baseline),
        ],
        env={**os.environ, "CALLS": str(commands)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = commands.read_text().splitlines()
    for step in (0, 16, 128):
        assert str(output / f"checkpoint-{step}") in calls
    assert f"pretrained={baseline},dtype=bfloat16,trust_remote_code=True" in calls
    assert baseline.is_dir()
    assert (trajectory / "checkpoint-16").exists() == (cleanup == "false")
    assert (trajectory / "checkpoint-128").exists() == (cleanup == "false")
