#!/usr/bin/env python3
"""Aggregate fixed TAR landscape caches across B, R, I, O, and RIO."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS = ("B", "R", "I", "O", "RIO")
LOCAL_RADIUS = 0.15713339880469646
LOCAL_FRAMES = (0, 1, 8, 32, 128, 512, 1000)
BROAD_FRAMES = (0, 128, 1000)
DEFENSE_FRAMES = (0, 8, 32, 125)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_compare_module(run):
    path = Path(run) / "source/misc/tar_compute_compare.py"
    spec = importlib.util.spec_from_file_location("frozen_tar_compute_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_state(path, expected="complete"):
    payload = read_json(path)
    if payload.get("state") != expected:
        raise ValueError(
            f"{path}: expected state={expected}, got {payload.get('state')!r}"
        )
    return payload


def validate_training_guard(folder, case):
    """Require completed training plus the frozen optimizer-input-order proof."""
    folder = Path(folder)
    require_state(folder / "status/train.json")
    proof = read_json(folder / "training-validation.json")
    maximum = int(case["max_steps"])
    order = proof.get("input_order_validation", {})
    if proof.get("state") != "passed" or proof.get("updates") != maximum:
        raise ValueError(f"{folder}: training validation did not pass for all updates")
    if (
        order.get("state") != "passed"
        or order.get("batches") != maximum
        or len(str(order.get("canonical_step_batch_sha256", ""))) != 64
    ):
        raise ValueError(f"{folder}: input-order guard did not pass")
    return proof


def validate_dynamics(record, run, program, probes, compare):
    folder = Path(record["folder"])
    for name in ("train", "evaluation", "analysis", "step-profiles"):
        require_state(folder / f"status/{name}.json")
    surface_validation = read_json(folder / "surfaces/validation.json")
    random_validation = read_json(folder / "surfaces/random-evolution-validation.json")
    if surface_validation.get("state") != "passed":
        raise ValueError(f"{record['name']}: surface validation did not pass")
    if random_validation.get("state") != "passed":
        raise ValueError(f"{record['name']}: broad-random validation did not pass")
    center_validation = read_json(folder / "center-validation.json")
    if not center_validation or any(
        int(row.get("count", 0)) <= 0
        or float(row.get("maximum_absolute_difference", math.inf)) >= 1e-5
        for row in center_validation
    ):
        raise ValueError(f"{record['name']}: center validation is absent or failed")
    validation = compare.checkpoint_validation(record, program, probes)
    if validation["issues"] or not validation.get("physical_comparable"):
        raise ValueError(
            f"{record['name']}: checkpoint validation failed: {validation['issues']}"
        )
    record["validation"] = validation
    return validation


def grid_from_rows(rows, radius, points, *, phase=None, step=None, plane=None):
    coordinates = np.linspace(-radius, radius, points)
    values = np.full((points, points), np.nan)
    for row in rows:
        if phase is not None and row.get("phase") != phase:
            continue
        if step is not None and float(row.get("step", "nan")) != step:
            continue
        if plane is not None and row.get("plane") != plane:
            continue
        x, y, value = (float(row[key]) for key in ("c0", "c1", "probe_loss"))
        ix, iy = int(np.argmin(abs(coordinates - x))), int(
            np.argmin(abs(coordinates - y))
        )
        if max(abs(coordinates[ix] - x), abs(coordinates[iy] - y)) > max(
            1e-10, radius * 1e-9
        ):
            raise ValueError("Observed grid differs from declared physical lattice")
        if np.isfinite(values[iy, ix]) and not math.isclose(
            values[iy, ix], value, abs_tol=1e-10, rel_tol=0
        ):
            raise ValueError("Conflicting duplicate lattice cell")
        values[iy, ix] = value
    if not np.isfinite(values).all():
        raise ValueError("Requested frame is incomplete")
    return coordinates, values


def shared_limits(values):
    flattened = np.concatenate([np.asarray(value).ravel() for value in values])
    low, high = float(np.min(flattened)), float(np.max(flattened))
    margin = max((high - low) * 0.04, 1e-3)
    return low - margin, high + margin


def matrix_limits(records, specs):
    """Require the same physical coordinate axes and a fully populated matrix."""
    frames = sorted({key[1] for key in specs})
    conditions = [record["condition"] for record in records]
    expected = {(condition, frame) for condition in conditions for frame in frames}
    if len(conditions) != len(set(conditions)) or set(specs) != expected:
        raise ValueError("Landscape matrix has missing or extra condition/frame cells")
    all_values = []
    for frame in frames:
        reference = np.asarray(
            specs[(conditions[0], frame)]["coordinates"], dtype=float
        )
        for condition in conditions:
            spec = specs[(condition, frame)]
            coordinates = np.asarray(spec["coordinates"], dtype=float)
            values = np.asarray(spec["values"], dtype=float)
            if coordinates.shape != reference.shape or not np.array_equal(
                coordinates, reference
            ):
                raise ValueError(f"Physical landscape axes differ at frame {frame}")
            if (
                values.shape != (len(coordinates), len(coordinates))
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"Landscape matrix contains invalid values at {(condition, frame)}"
                )
            all_values.append(values)
    return frames, shared_limits(all_values)


def finish_figure_layout(figure, axes, image, title):
    """Reserve physical margins for headers, panel labels and a separate cax."""
    nrows, ncols = axes.shape
    width = 3.35 * ncols + 1.5
    panel_height, row_gap = 2.45, 0.8
    bottom, top_margin = 0.65, 0.95
    height = nrows * panel_height + (nrows - 1) * row_gap + bottom + top_margin
    figure.set_size_inches(width, height, forward=True)
    figure.subplots_adjust(
        left=0.75 / width,
        right=(width - 1.8) / width,
        bottom=bottom / height,
        top=1 - top_margin / height,
        hspace=row_gap / panel_height,
        wspace=0.32,
    )
    content_height = height - bottom - top_margin
    bar_height = min(5.5, 0.84 * content_height)
    color_axis = figure.add_axes(
        (
            (width - 0.9) / width,
            (bottom + (content_height - bar_height) / 2) / height,
            0.14 / width,
            bar_height / height,
        )
    )
    figure.colorbar(image, cax=color_axis, label="Fixed-probe CE (nats)")
    figure.suptitle(title, fontsize=12, y=1 - 0.1 / height, va="top")


def draw_grid_matrix(plt, records, specs, output, *, title, cmap="viridis", xlim=None):
    """Plot one fully measured condition-by-frame matrix on a shared CE scale."""
    frames, (low, high) = matrix_limits(records, specs)
    figure, axes = plt.subplots(
        len(frames),
        len(records),
        figsize=(3.35 * len(records), 2.65 * len(frames)),
        squeeze=False,
    )
    for ri, step in enumerate(frames):
        for ci, record in enumerate(records):
            axis = axes[ri, ci]
            spec = specs.get((record["condition"], step))
            if spec is None:
                axis.set_axis_off()
                continue
            coordinates, matrix = spec["coordinates"], spec["values"]
            image = axis.contourf(
                coordinates,
                coordinates,
                matrix,
                levels=np.linspace(low, high, 18),
                cmap=cmap,
                vmin=low,
                vmax=high,
                extend="both",
            )
            axis.set_aspect("equal")
            if xlim is not None:
                axis.set_xlim(*xlim)
                axis.set_ylim(*xlim)
            axis.set_title(f"{record['condition']} · step {step}", fontsize=9)
            axis.set_xlabel("c0 (global L2)")
            if ci == 0:
                axis.set_ylabel("c1 (global L2)")
    finish_figure_layout(
        figure,
        axes,
        image,
        title + f"\nShared absolute CE scale [{low:.5g}, {high:.5g}]",
    )
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return low, high, len(specs), frames


def write_validation_assets(
    run, output, program, probes, compare, records, defense_history
):
    rows = []
    for record in records:
        validation = validate_dynamics(record, run, program, probes, compare)
        rows.append(
            {
                "condition": record["condition"],
                "case": record["name"],
                "folder": str(record["folder"]),
                "issues": len(validation["issues"]),
                "physical_comparable": validation["physical_comparable"],
                "local_frames": ";".join(map(str, validation["local_frames"])),
                "broad_frames": ";".join(map(str, validation["broad_frames"])),
                "fixed_documents": record["probes"].get("fixed_documents"),
            }
        )
    write_csv(output / "numeric-validation-coverage.csv", rows, list(rows[0]))
    history = read_json(defense_history / "center-validation.json")
    if not history or any(
        int(row.get("count", 0)) <= 0
        or float(row.get("maximum_absolute_difference", math.inf)) >= 1e-5
        for row in history
    ):
        raise ValueError(
            "Baseline defense-history center validation is absent or failed"
        )
    for name in ("defense", "analysis", "random-frames", "report"):
        require_state(defense_history / f"status/{name}.json")
    history_surfaces = read_json(defense_history / "surfaces/validation.json")
    if history_surfaces.get("state") != "passed":
        raise ValueError("Baseline defense-history surface validation did not pass")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (args.output or (run / "aggregate-comparison-v2/final")).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace immutable output: {output}")
    program, probes = read_json(run / "study.json"), read_json(run / "probes.json")
    if probes.get("state") not in (None, "complete"):
        raise ValueError("Shared probe preparation is incomplete")
    compare = load_compare_module(run)
    controls = program["comparison_controls"]
    control_name, control = next(iter(controls.items()))
    records = []
    base = compare.checkpoint_record(
        "B", control["folder"], int(control["schedule_steps"]), control
    )
    base["condition"] = "B"
    records.append(base)
    by_condition = {"B": base}
    for condition in ("R", "I", "O", "RIO"):
        name = f"hparam-{condition.lower()}-lr1e-6"
        declaration = program["cases"].get(name)
        if not declaration or declaration.get("role") != "attack":
            raise ValueError(f"Missing planned attack case {name}")
        record = compare.checkpoint_record(
            name, run / "cases" / name, declaration["max_steps"]
        )
        record["label"] = f"{condition}: {declaration['defense_hyperparameters']}"
        record["condition"] = condition
        defense = run / "cases" / f"hparam-{condition.lower()}-defense"
        defense_case = program["cases"][f"hparam-{condition.lower()}-defense"]
        validate_training_guard(defense, defense_case)
        records.append(record)
        by_condition[condition] = record

    # Finish all provenance/status/validation checks before creating any output.
    for record in records:
        validate_dynamics(record, run, program, probes, compare)
    defense_history = Path(program["baseline_dynamics"])
    history = read_json(defense_history / "center-validation.json")
    if not history or any(
        int(row.get("count", 0)) <= 0
        or float(row.get("maximum_absolute_difference", math.inf)) >= 1e-5
        for row in history
    ):
        raise ValueError(
            "Baseline defense-history center validation is absent or failed"
        )
    for name in ("defense", "analysis", "random-frames", "report"):
        require_state(defense_history / f"status/{name}.json")
    if read_json(defense_history / "surfaces/validation.json").get("state") != "passed":
        raise ValueError("Baseline defense-history surface validation did not pass")

    output.mkdir(parents=True)
    coverage = write_validation_assets(
        run, output, program, probes, compare, records, defense_history
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_rows = []
    figures = output / "figures"
    figures.mkdir()

    # Fixed end-defense endpoint: shared random/reference rows share physical axes;
    # fitted attack planes are condition-specific and explicitly labeled by radius.
    fixed_records = []
    fixed_limits = {}
    fixed_rows = []
    for plane in ("random", "reference", "attack"):
        grids = {}
        all_values = []
        for record in records:
            folder = Path(record["folder"])
            rows = compare.read_rows(folder / "surfaces/fixed-grids.csv")
            radius = float(
                record["settings"]
                .get("plane_radii", {})
                .get(plane, record["geometry"].get("radius", 1))
            )
            plane_rows = [row for row in rows if row.get("plane") == plane]
            points = int(record["settings"].get("surface_points", 17))
            coords, vals = grid_from_rows(plane_rows, radius, points, plane=plane)
            grids[record["condition"]] = (coords, vals, radius)
            all_values.append(vals)
            fixed_rows.append(
                {
                    "condition": record["condition"],
                    "plane": plane,
                    "half_width_global_l2": radius,
                    "grid_points": points,
                    "measured_values": int(vals.size),
                }
            )
        low, high = shared_limits(all_values)
        fixed_limits[plane] = (low, high)
        figure, axes = plt.subplots(
            1, len(records), figsize=(3.3 * len(records), 3.5), squeeze=False
        )
        for axis, record in zip(axes[0], records):
            coords, vals, radius = grids[record["condition"]]
            image = axis.contourf(
                coords,
                coords,
                vals,
                levels=np.linspace(low, high, 18),
                cmap="viridis",
                vmin=low,
                vmax=high,
                extend="both",
            )
            axis.set_aspect("equal")
            axis.set_title(f"{record['condition']} · ±{radius:.5g} L2", fontsize=9)
            axis.set_xlabel("c0 (global L2)")
            if axis is axes[0, 0]:
                axis.set_ylabel("c1 (global L2)")
        finish_figure_layout(
            figure,
            axes,
            image,
            (
                f"End-defense fixed {plane} plane; shared absolute CE scale [{low:.5g}, {high:.5g}]\nAttack-plane directions are fitted separately per condition"
                if plane == "attack"
                else f"End-defense fixed {plane} plane; shared absolute CE scale [{low:.5g}, {high:.5g}]"
            ),
        )
        path = figures / f"end-defense-fixed-{plane}"
        figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        figure_rows.append(
            {
                "figure": path.name,
                "plane": plane,
                "frames": "attack-start/defense-endpoint",
                "color_min": low,
                "color_max": high,
                "panels": len(records),
                "status": "complete",
            }
        )
    write_csv(output / "fixed-plane-coverage.csv", fixed_rows, list(fixed_rows[0]))

    # Shared-random broad surfaces and shared-reference local slices, each on a
    # common numerical CE scale across all conditions and frames.
    broad_frames = sorted(
        set(BROAD_FRAMES).intersection(
            *(set(record["broad_frames"]) for record in records)
        )
    )
    if 0 not in broad_frames or 1000 not in broad_frames:
        raise ValueError(f"Common broad frames omit endpoints: {broad_frames}")
    broad_specs = {}
    for record in records:
        rows = compare.read_rows(
            Path(record["folder"]) / "surfaces/random-evolution.csv"
        )
        points = int(record["settings"].get("broad_points", 9))
        for step in broad_frames:
            coordinates, values = grid_from_rows(
                rows, 125, points, phase="attack", step=step
            )
            broad_specs[(record["condition"], step)] = {
                "coordinates": coordinates,
                "values": values,
            }
    stats = draw_grid_matrix(
        plt,
        records,
        broad_specs,
        figures / "broad-random-r125",
        title="Shared-random broad attack slices; half-width 125 global L2",
    )
    figure_rows.append(
        {
            "figure": "broad-random-r125",
            "plane": "shared-random",
            "frames": ";".join(map(str, stats[3])),
            "color_min": stats[0],
            "color_max": stats[1],
            "panels": stats[2],
            "status": "complete",
        }
    )

    local_frames = sorted(
        set(LOCAL_FRAMES).intersection(
            *(set(record["local_frames"]) for record in records)
        )
    )
    if 0 not in local_frames or 1000 not in local_frames:
        raise ValueError(f"Common local frames omit endpoints: {local_frames}")
    local_specs = {}
    for record in records:
        rows = compare.read_rows(Path(record["folder"]) / "surfaces/local-grids.csv")
        points = int(record["settings"].get("local_surface_points", 9))
        for step in local_frames:
            coordinates, values = grid_from_rows(
                rows, LOCAL_RADIUS, points, phase="attack", step=step
            )
            local_specs[(record["condition"], step)] = {
                "coordinates": coordinates,
                "values": values,
            }
    stats = draw_grid_matrix(
        plt,
        records,
        local_specs,
        figures / "local-reference-matched-attack",
        title=f"Matched shared-reference local slices; half-width {LOCAL_RADIUS:.12g} global L2",
    )
    figure_rows.append(
        {
            "figure": "local-reference-matched-attack",
            "plane": "shared-reference",
            "frames": ";".join(map(str, stats[3])),
            "color_min": stats[0],
            "color_max": stats[1],
            "panels": stats[2],
            "status": "complete",
        }
    )

    # Defense histories use the actual saved baseline defense run for B, because
    # the 1000-step B attack case did not retain that history.
    history_records = []
    for record in records:
        history_records.append(
            (
                record,
                (
                    defense_history
                    if record["condition"] == "B"
                    else Path(record["folder"])
                ),
            )
        )
    defense_specs = {}
    for record, folder in history_records:
        rows = compare.read_rows(folder / "surfaces/local-grids.csv")
        settings = read_json(folder / "study.json")
        points = int(settings.get("local_surface_points", 9))
        for step in DEFENSE_FRAMES:
            coordinates, values = grid_from_rows(
                rows, LOCAL_RADIUS, points, phase="defense", step=step
            )
            defense_specs[(record["condition"], step)] = {
                "coordinates": coordinates,
                "values": values,
            }
    stats = draw_grid_matrix(
        plt,
        records,
        defense_specs,
        figures / "defense-history-local-reference",
        title=f"Saved defense-history shared-reference slices; half-width {LOCAL_RADIUS:.12g} global L2",
    )
    figure_rows.append(
        {
            "figure": "defense-history-local-reference",
            "plane": "shared-reference",
            "frames": ";".join(map(str, stats[3])),
            "color_min": stats[0],
            "color_max": stats[1],
            "panels": stats[2],
            "status": "complete",
        }
    )

    write_csv(output / "figure-index.csv", figure_rows, list(figure_rows[0]))
    (output / "aggregate-validation.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "conditions": [record["condition"] for record in records],
                "control": control_name,
                "numeric_validation_rows": len(coverage),
                "figures": figure_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"state": "complete", "output": str(output), "figures": len(figure_rows)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
