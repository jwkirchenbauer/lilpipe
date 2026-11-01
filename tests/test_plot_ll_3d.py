import runpy
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/loss-landscapes/plot_ll_3d.py"
)


@pytest.mark.parametrize("columns", [("alpha", "beta"), ("c0", "c1")])
def test_3d_csv_plot_preserves_small_loss_ranges(monkeypatch, tmp_path, columns):
    radii = (0.1, 0.01, 0.001)
    pd.DataFrame(
        [
            {"radius": r, columns[0]: x, columns[1]: y, "loss": 2 + 1e-5 * (x + 3 * y)}
            for r in radii
            for x in (-r, 0, r)
            for y in (-r, 0, r)
        ]
    ).to_csv(tmp_path / "grid.csv", index=False)
    output = tmp_path / "surfaces.pdf"
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "--indir", str(tmp_path), "--outpath", str(output)]
    )
    result = runpy.run_path(str(SCRIPT), run_name="__main__")
    assert output.is_file()
    axes = result["fig"].axes
    assert len(axes) == 3
    for axis, radius in zip(axes, radii, strict=True):
        assert axis.name == "3d"
        assert axis.collections
        assert len(axis.child_axes) == 1
        assert axis.get_title() == f"Radius {radius:.2e}"
        low, high = axis.get_zlim()
        assert high - low == pytest.approx(8e-5 * radius, abs=1e-12)
