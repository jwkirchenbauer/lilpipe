"""Reusable SEAM wrappers live with code; run records preserve submitted copies."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "examples/seam/scripts/verify_checkpoint.py"


@pytest.mark.parametrize("complete", [True, False])
def test_checkpoint_verifier_checks_referenced_shards(tmp_path, complete):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text('{"global_step": 50}')
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model-1.safetensors").write_bytes(b"weights")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}
        )
    )
    if complete:
        (checkpoint / "model-2.safetensors").write_bytes(b"weights")
    output = tmp_path / "validation.json"
    result = subprocess.run(
        [sys.executable, str(VERIFY), str(checkpoint), "50", str(output)],
        capture_output=True,
        text=True,
    )
    if complete:
        assert result.returncode == 0, result.stderr
        assert json.loads(output.read_text())["global_step"] == 50
    else:
        assert result.returncode != 0
        assert not output.exists()
