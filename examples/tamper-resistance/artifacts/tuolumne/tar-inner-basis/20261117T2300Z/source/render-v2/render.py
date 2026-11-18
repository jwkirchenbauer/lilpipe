"""Correct one figure layout using validated, unchanged cached analysis tables."""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import tar_inner_geometry as geometry


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(sys.argv[1]).resolve()
    folder = run / "geometry"
    proof = folder / "render-v2-provenance.json"
    if proof.exists():
        raise FileExistsError(proof)
    summary = json.loads((folder / "summary.json").read_text())
    status = json.loads((folder / "status/geometry.json").read_text())
    assert status["state"] == "complete"
    assert status["signature"] == summary["signature"]
    tables = {name: [] for name in geometry.TABLES}
    inputs = {}
    numeric = {str(path): digest(path) for path in folder.glob("*.csv")}
    for call in summary["outer_calls"]:
        path = folder / "per-call" / f"outer-{call:06d}.json"
        record = json.loads(path.read_text())
        assert record["signature"] == summary["signature"]
        inputs[str(path)] = digest(path)
        for name in tables:
            tables[name].extend(record["tables"][name])
    with tempfile.TemporaryDirectory(
        prefix="capture-layout-", dir=run / "cache"
    ) as tmp:
        geometry.figures(
            Path(tmp),
            tables,
            summary["inner_steps"],
            summary["actual_views"],
            summary["outer_calls"],
        )
        published = {}
        for extension in ("png", "pdf"):
            target = folder / f"capture-by-step.{extension}"
            os.replace(Path(tmp) / target.name, target)
            published[str(target)] = digest(target)
    assert numeric == {str(path): digest(path) for path in folder.glob("*.csv")}
    result = {
        "state": "complete",
        "change": "Move capture-plot legend above the panels to separate it from the x-axis label",
        "analysis_signature": summary["signature"],
        "cached_tables": inputs,
        "numeric_csv_sha256_unchanged": numeric,
        "published": published,
        "source_sha256": {
            str(path): digest(path) for path in Path(__file__).parent.glob("*.py")
        },
    }
    proof.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
