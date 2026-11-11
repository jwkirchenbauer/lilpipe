"""Render the stronger attack and retain an explicit shared utility-group scale."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

CASE = "full-maxent-terminal-fresh-retain"
VARIANT = "full-new"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(result):
    groups = result.get("probes", {"forget_train": [], "forget_heldout_subset": []})
    return [
        point[group]["ce"]
        for plane in result["planes"].values()
        for point in plane["grid"]
        for group in groups
    ]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    run = parser.parse_args().run.resolve()
    canonical_source = (
        run / "source/profiles/geometry-plot-v4/tar_cleanroom_geometry.py"
    )
    variant_source = (
        run / "source/profiles/geometry-variant-v1/tar_cleanroom_geometry.py"
    )
    canonical = load(canonical_source, "canonical_cleanroom_plot")
    alternate = load(variant_source, "alternate_cleanroom_plot")
    rendering_path = run / "geometry/rendering.json"
    rendering = read(rendering_path)
    folder = run / "geometry" / CASE / "attack-variants" / VARIANT
    result_path = folder / "geometry.json"
    result = read(result_path)
    base_result = read(run / "geometry" / CASE / "geometry.json")
    assert result["case"] == CASE and result["attack_variant"] == VARIANT
    assert set(result["planes"]) == {"first", "fit"}
    assert set(result["probes"]) == {
        "forget_train",
        "forget_heldout_subset",
        "forget_new_attack",
    }
    assert result["data_content_sha256"] == base_result["data_content_sha256"]
    assert (
        result["checkpoint_files"]["anchor"]
        == base_result["checkpoint_files"]["anchor"]
    )
    assert all(len(plane["grid"]) == 81 for plane in result["planes"].values())
    for group in ("forget_train", "forget_heldout_subset"):
        assert (
            abs(
                result["anchor_probes"][group]["ce"]
                - base_result["anchor_probes"][group]["ce"]
            )
            < 1e-5
        )
    assert all(math.isfinite(value) for value in values(result))
    passing = rendering["utility_passing_cases"]
    passing_results = {
        case: read(run / "geometry" / case / "geometry.json") for case in passing
    }
    shared_limits = [
        0.0,
        float(
            math.ceil(
                max(
                    max(values(result)),
                    max(value for r in passing_results.values() for value in values(r)),
                )
            )
        ),
    ]
    alternate_limits = [
        0.0,
        max(rendering["all_case_ce_limits"][1], float(math.ceil(max(values(result))))),
    ]
    alternate.plot_saved(
        SimpleNamespace(
            run=run, case=CASE, attack_variant=VARIANT, ce_limits=alternate_limits
        )
    )
    clean = alternate.load_cleanroom(run)
    common = {
        "comparison_cases": passing,
        "additional_attacker": {"case": CASE, "variant": VARIANT},
        "ce_scale": {"ce_min": shared_limits[0], "ce_max": shared_limits[1]},
        "canonical_plot_source_sha256": sha(canonical_source),
        "variant_plot_source_sha256": sha(variant_source),
        "driver_sha256": sha(Path(__file__)),
        "policy": "A single shared CE range across the three utility-passing models and the additional attacker view; measured native grids remain unchanged.",
    }
    for case, data in passing_results.items():
        target = run / "geometry" / case / "utility-comparison"
        canonical.plot(data, target, shared_limits)
        clean.dump(
            target / "rendering.json",
            {
                **common,
                "case": case,
                "input_geometry_sha256": sha(run / "geometry" / case / "geometry.json"),
            },
        )
    target = folder / "utility-comparison"
    target.mkdir(exist_ok=True)
    alternate.plot(read(result_path), target, shared_limits)
    clean.dump(
        target / "rendering.json",
        {
            **common,
            "case": CASE,
            "attack_variant": VARIANT,
            "input_geometry_sha256": sha(result_path),
        },
    )
    records = {entry["file"]: entry for entry in rendering["files"]}
    for target in (folder, folder / "utility-comparison"):
        for name in ("slices", "slices-3d"):
            for suffix in ("png", "pdf"):
                path = target / (name + "." + suffix)
                assert path.is_file() and path.stat().st_size > 0
                records[str(path.relative_to(run))] = {
                    "file": str(path.relative_to(run))
                }
    for name, record in records.items():
        record["sha256"] = sha(run / name)
    assert len(records) == 80
    rendering.update(
        additional_driver_sha256=sha(Path(__file__)),
        utility_passing_ce_limits=shared_limits,
        additional_geometries=[
            {
                "case": CASE,
                "attack_variant": VARIANT,
                "geometry": str(result_path.relative_to(run)),
                "planes": ["first", "fit"],
                "probe_groups": list(result["probes"]),
                "full_range_ce_limits": alternate_limits,
                "source_sha256": sha(variant_source),
            }
        ],
        files=list(records.values()),
    )
    clean.dump(rendering_path, rendering)
    print(
        json.dumps(
            {
                "shared_utility_ce_limits": shared_limits,
                "additional_full_ce_limits": alternate_limits,
                "files": len(records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
