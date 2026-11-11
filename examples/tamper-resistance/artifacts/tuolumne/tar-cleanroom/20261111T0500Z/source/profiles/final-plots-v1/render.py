"""Render complete study grids with all-case and utility-passing shared scales."""

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ce_values(result):
    return [
        point[group]["ce"]
        for plane in result["planes"].values()
        for point in plane["grid"]
        for group in ("forget_train", "forget_heldout_subset")
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    plot_source = run / "source/profiles/geometry-plot-v4/tar_cleanroom_geometry.py"
    spec = importlib.util.spec_from_file_location("sealed_cleanroom_plot", plot_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = read(run / "report/summary.json")
    assert not summary["warnings"], summary["warnings"]
    assert len(summary["benchmarks"]) == 162
    assert all(row["state"] == "complete" for row in summary["benchmarks"])
    cases = ["control"] + [row["case"] for row in summary["defenses"]]
    assert len(cases) == len(set(cases)) == 9
    results = {}
    for case in cases:
        result = read(run / "geometry" / case / "geometry.json")
        assert result["case"] == case
        assert tuple(result["planes"]) == module.PLANE_ORDER
        assert all(len(plane["grid"]) == 81 for plane in result["planes"].values())
        assert all(math.isfinite(value) for value in ce_values(result))
        results[case] = result
    identity = {
        r["planes"]["reference"]["reference_definition"]["identity_sha256"]
        for r in results.values()
    }
    assert len(identity) == 1, identity
    gate = summary["utility"]["native_fp32_gate_percent"]
    passing = []
    for row in summary["benchmarks"]:
        point = row["point"]
        if point["variant"] is None and point["case"] in cases:
            final_step = 0 if point["case"] == "control" else 125
            if point["step"] == final_step and row["native_gate_pass"] is True:
                passing.append(point["case"])
    passing = [case for case in cases if case in passing]
    assert passing == [
        "control",
        "full-maxent-terminal-fresh-retain",
        "full-ce-worst-fresh-retain",
    ], passing
    all_limits = [
        0.0,
        float(math.ceil(max(v for r in results.values() for v in ce_values(r)))),
    ]
    passing_limits = [
        0.0,
        float(math.ceil(max(v for case in passing for v in ce_values(results[case])))),
    ]
    control_path = run / "geometry/control/geometry.json"
    archived = run / "revisions/control-geometry-plot-snapshots"
    archived.mkdir(parents=True, exist_ok=True)
    destination = archived / (sha(control_path) + ".json")
    if not destination.exists():
        shutil.copy2(control_path, destination)
    clean = module.load_cleanroom(run)
    rendering = {
        "driver_sha256": sha(Path(__file__)),
        "plot_source": str(plot_source),
        "plot_source_sha256": sha(plot_source),
        "all_cases": cases,
        "all_case_ce_limits": all_limits,
        "utility_gate_percent": gate,
        "utility_passing_cases": passing,
        "utility_passing_ce_limits": passing_limits,
        "policy": "One shared full data range per declared comparison group; same measured grids and physical axes in both exports.",
        "files": [],
    }
    for case in cases:
        module.plot_saved(SimpleNamespace(run=run, case=case, ce_limits=all_limits))
        folder = run / "geometry" / case
        for name in ("slices", "slices-3d", "matched-radius-sections"):
            for suffix in ("png", "pdf"):
                path = folder / (name + "." + suffix)
                assert path.is_file() and path.stat().st_size > 0
                rendering["files"].append(
                    {"file": str(path.relative_to(run)), "sha256": sha(path)}
                )
        if case in passing:
            result = read(folder / "geometry.json")
            target = folder / "utility-comparison"
            target.mkdir(exist_ok=True)
            scale = module.plot(result, target, passing_limits)
            clean.dump(
                target / "rendering.json",
                {
                    "case": case,
                    "comparison_cases": passing,
                    "measured_native_utility_gate_percent": gate,
                    "ce_scale": scale,
                    "plot_source_sha256": sha(plot_source),
                    "input_geometry_sha256": sha(folder / "geometry.json"),
                },
            )
            for name in ("slices", "slices-3d", "matched-radius-sections"):
                for suffix in ("png", "pdf"):
                    path = target / (name + "." + suffix)
                    assert path.is_file() and path.stat().st_size > 0
                    rendering["files"].append(
                        {"file": str(path.relative_to(run)), "sha256": sha(path)}
                    )
    clean.dump(run / "geometry/rendering.json", rendering)
    print(
        json.dumps({key: value for key, value in rendering.items() if key != "files"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
