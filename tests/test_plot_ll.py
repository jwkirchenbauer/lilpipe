import runpy
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/loss-landscapes/plot_ll_2d.py"
)


@pytest.mark.parametrize("with_trajectory", [False, True])
@pytest.mark.parametrize("columns", [("alpha", "beta"), ("c0", "c1")])
def test_plot_ll_reads_saved_grids(monkeypatch, tmp_path, columns, with_trajectory):
    rows = [
        {"radius": r, columns[0]: x, columns[1]: y, "loss": 2 + x + 3 * y}
        for r in (10, 25, 50, 75)
        for x in (-r, 0, r)
        for y in (-r, 0, r)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "grid.csv", index=False)
    coordinates = [[0, 0], [-0.03, -0.04], [-0.035, -0.034]]
    if with_trajectory:
        pd.DataFrame(coordinates, columns=columns).to_csv(
            tmp_path / "trajectory.csv", index=False
        )
    output = tmp_path / "landscapes.pdf"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--indir", str(tmp_path), "--outpath", str(output)]
    )
    result = runpy.run_path(str(SCRIPT), run_name="__main__")
    assert output.is_file()
    axes = list(result["axes"].flat)
    assert [axis.get_title() for axis in axes] == [
        "Radius 7.50e+01",
        "Radius 5.00e+01",
        "Radius 2.50e+01",
        "Radius 1.00e+01",
    ]
    assert all(axis.name == "rectilinear" and axis.collections for axis in axes)
    assert all((axis.get_xlabel(), axis.get_ylabel()) == ("c0", "c1") for axis in axes)
    if with_trajectory:
        for axis in axes:
            dots = axis.collections[-1]
            assert dots.get_offsets().tolist() == coordinates
            assert dots.get_facecolors()[0].tolist() == [1, 0, 0, 1]
