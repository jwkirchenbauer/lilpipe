import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_dynamics.py"
)
SPEC = importlib.util.spec_from_file_location("tar_dynamics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_atomic_savez_round_trip_preserves_requested_filename(tmp_path):
    target = tmp_path / "minibatch-losses.npz"
    losses = np.arange(12, dtype=np.float64).reshape(3, 4)
    steps = np.array([0, 4, 8])

    MODULE.atomic_savez(target, losses=losses, steps=steps)

    assert target.exists()
    assert not (tmp_path / "minibatch-losses.npz.npz").exists()
    with np.load(target) as archive:
        np.testing.assert_array_equal(archive["losses"], losses)
        np.testing.assert_array_equal(archive["steps"], steps)
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_savez_failure_keeps_previous_archive(tmp_path, monkeypatch):
    target = tmp_path / "minibatch-losses.npz"
    target.write_bytes(b"previous archive")

    def partial_then_fail(file, **arrays):
        file.write(b"partial archive")
        raise RuntimeError("simulated writer failure")

    monkeypatch.setattr(MODULE.np, "savez", partial_then_fail)
    with pytest.raises(RuntimeError, match="simulated writer failure"):
        MODULE.atomic_savez(target, losses=np.arange(3))

    assert target.read_bytes() == b"previous archive"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_dataframe_csv_failure_keeps_previous_csv(tmp_path, monkeypatch):
    target = tmp_path / "random-evolution.csv"
    target.write_text("old,data\n1,2\n")
    frame = pd.DataFrame({"old": [3], "data": [4]})
    frame_type = type(frame)

    def partial_then_fail(self, path, index=False):
        Path(path).write_text("partial,csv\n")
        raise RuntimeError("simulated CSV writer failure")

    monkeypatch.setattr(frame_type, "to_csv", partial_then_fail)
    with pytest.raises(RuntimeError, match="simulated CSV writer failure"):
        MODULE.atomic_dataframe_csv(frame, target)

    assert target.read_text() == "old,data\n1,2\n"
    assert list(tmp_path.glob(".*.tmp")) == []
