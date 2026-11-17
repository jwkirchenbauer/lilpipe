"""Validate and compare eight controlled original-recipe static grids."""

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path

from coordinates import canonicalize, key as coordinate_key
from static_recipe import RADII, dump, file_identity, read, sha, validate_grid


FACTORS = ("R", "I", "O")
CELLS = ("B", "R", "I", "RI", "O", "RO", "IO", "RIO")
METRICS = ("center_ce", "rim_rise", "center_elevation")
LOW = (2.5, 1e-6, 5e-6)
HIGH = (4.0, 1e-5, 1e-5)
DEFINITIONS = {
    "center_ce": "L[4,4] on the 9x9 grid; CE in nats.",
    "rim_rise": "max(L[4+di,4+dj] for di,dj in {-1,0,1} except di=dj=0) - L[4,4]. The 8 immediate neighbors have coordinate spacing radius/4. No clipping at zero.",
    "center_elevation": "L[4,4] - mean(L[i,j] where i or j is 0 or 8). The perimeter contains 32 cells. No clipping at zero.",
    "interpretation": "Finite-grid diagnostics of these sampled slices; not global barriers, Hessian curvature or attack robustness. One seed and no replication; no p-values.",
    "factor_coding": "R/I/O are binary 0/1. R: retain 2.5/4; I: inner LR 1e-6/1e-5; O: outer LR 5e-6/1e-5.",
    "anchored_coefficients": "For subset S, c_S=sum(T subset S) (-1)^(|S|-|T|) f(1_T,0_else). No divisor. The exact 0/1 multilinear expansion is f(r,i,o)=c_empty+c_R*r+c_I*i+c_O*o+c_RI*r*i+c_RO*r*o+c_IO*i*o+c_RIO*r*i*o.",
    "conditional_main": "At each fixed combination of the other 2 factors, f(factor=1)-f(factor=0). No divisor; the marginal main contrast is the mean of 4 such differences, divisor 4.",
    "conditional_pair": "At each fixed setting of the third factor, f11-f10-f01+f00. No divisor; the marginal pair contrast is the mean of 2 such differences, divisor 2.",
    "three_way": "f111-f110-f101-f011+f100+f010+f001-f000. No divisor; also the difference between conditional pair contrasts at third-factor settings 1 and 0.",
}


class InvalidInput(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidInput(message)


def bits(cell):
    return (
        tuple(int(factor in cell) for factor in FACTORS) if cell != "B" else (0, 0, 0)
    )


def label_for(values):
    return (
        "".join(factor for factor, value in zip(FACTORS, values, strict=True) if value)
        or "B"
    )


def parameters(cell):
    values = [
        HIGH[index] if value else LOW[index] for index, value in enumerate(bits(cell))
    ]
    return dict(
        zip(
            ("retain_coefficient", "inner_learning_rate", "outer_learning_rate"),
            values,
            strict=True,
        )
    )


def grid_metrics(rows, radius):
    selected = [row for row in rows if row["radius"] == radius]
    coordinates = sorted({row["c0"] for row in selected})
    require(
        len(selected) == 81 and len(coordinates) == 9,
        "Metric requires complete 9x9 grid",
    )
    lookup = {(row["c0"], row["c1"]): row["loss"] for row in selected}
    require(len(lookup) == 81, "Duplicate/missing metric coordinates")
    values = [[lookup[(x, y)] for x in coordinates] for y in coordinates]
    center = values[4][4]
    require(coordinates[4] == 0, "Grid center is not zero")
    neighbors = [
        values[4 + di][4 + dj]
        for di, dj in itertools.product((-1, 0, 1), repeat=2)
        if di or dj
    ]
    perimeter = [
        values[i][j]
        for i, j in itertools.product(range(9), repeat=2)
        if i in (0, 8) or j in (0, 8)
    ]
    require(len(neighbors) == 8 and len(perimeter) == 32, "Wrong diagnostic cell count")
    return {
        "center_ce": center,
        "rim_rise": max(neighbors) - center,
        "center_elevation": center - math.fsum(perimeter) / 32,
    }


def factorial_contrasts(values):
    require(set(values) == set(CELLS), "All 8 factorial cells are required")
    require(
        all(math.isfinite(value) for value in values.values()),
        "Nonfinite factorial response",
    )
    rows = []
    for size in range(1, 4):
        for subset in itertools.combinations(range(3), size):
            factors = "".join(FACTORS[index] for index in subset)
            coefficient = 0.0
            for settings in itertools.product((0, 1), repeat=size):
                point = [0, 0, 0]
                for index, setting in zip(subset, settings, strict=True):
                    point[index] = setting
                coefficient += (-1) ** (size - sum(settings)) * values[label_for(point)]
            rows.append(
                {
                    "kind": "anchored_coefficient",
                    "factors": factors,
                    "conditioning": "remaining factors0",
                    "normalization": 1,
                    "value": coefficient,
                }
            )
            if size == 3:
                continue
            remaining = [index for index in range(3) if index not in subset]
            conditionals = []
            for fixed in itertools.product((0, 1), repeat=len(remaining)):
                difference = 0.0
                for settings in itertools.product((0, 1), repeat=size):
                    point = [0, 0, 0]
                    for index, setting in zip(remaining, fixed, strict=True):
                        point[index] = setting
                    for index, setting in zip(subset, settings, strict=True):
                        point[index] = setting
                    difference += (-1) ** (size - sum(settings)) * values[
                        label_for(point)
                    ]
                conditioning = ",".join(
                    f"{FACTORS[index]}={setting}"
                    for index, setting in zip(remaining, fixed, strict=True)
                )
                conditionals.append(difference)
                rows.append(
                    {
                        "kind": "conditional_main" if size == 1 else "conditional_pair",
                        "factors": factors,
                        "conditioning": conditioning,
                        "normalization": 1,
                        "value": difference,
                    }
                )
            rows.append(
                {
                    "kind": "marginal_main" if size == 1 else "marginal_pair",
                    "factors": factors,
                    "conditioning": "mean over remaining factor settings",
                    "normalization": len(conditionals),
                    "value": math.fsum(conditionals) / len(conditionals),
                }
            )
    return rows


def normalized_metadata(metadata, legacy=False):
    return {
        "get_ll_sha256": metadata["source_sha256"]["get_ll.py"],
        "torch": metadata["torch"],
        "hip": metadata["hip"],
        "seed": metadata["direction_seed"] if legacy else metadata["seed"],
        "document_sha256": metadata["document_sha256"],
        "dataset_revision": metadata["dataset_revision"],
        "radii": metadata["radii"],
        "n_pts": metadata["n_pts"],
        "batch_size": metadata["batch_size"],
        "max_seq_len": metadata["max_seq_len"],
    }


def validate_records(records):
    require(
        len(records) == 8 and {record["cell"] for record in records} == set(CELLS),
        "Exactly 8 unique controlled cells required",
    )
    reference = next(record for record in records if record["cell"] == "B")
    expected = reference["common"]
    require(
        expected["seed"] == 42
        and expected["radii"] == RADII
        and expected["n_pts"] == 9,
        "Wrong static seed/radii/grid",
    )
    require(
        expected["batch_size"] == 2
        and expected["max_seq_len"] == 512
        and len(expected["document_sha256"]) == 10,
        "Wrong fixed10 inputs",
    )
    keys = {coordinate_key(row) for row in reference["rows"]}
    require(len(keys) == 648, "Reference grid has missing/duplicate axes")
    batch_hashes = None
    order_hash = None
    for record in records:
        cell = record["cell"]
        require(
            record["parameters"] == parameters(cell), f"{cell}: wrong factor values"
        )
        require(
            record["common"] == expected,
            f"{cell}: incompatible source/version/probes/scale protocol",
        )
        require(
            len(record["rows"]) == 648
            and {coordinate_key(row) for row in record["rows"]} == keys,
            f"{cell}: missing or mismatched FP32 grid axes",
        )
        require(
            all(math.isfinite(row["loss"]) for row in record["rows"]),
            f"{cell}: nonfinite loss",
        )
        if cell != "B":
            require(
                len(record["batch_hashes"]) == 5,
                f"{cell}: missing fixed10 token batch hashes",
            )
            batch_hashes = batch_hashes or record["batch_hashes"]
            require(
                record["batch_hashes"] == batch_hashes,
                f"{cell}: token batch inputs differ",
            )
            order_hash = order_hash or record["order_sha256"]
            require(
                record["order_sha256"] == order_hash,
                f"{cell}: defense input-order proof differs",
            )
    return {
        "common_protocol": expected,
        "evaluated_batch_sha256": batch_hashes,
        "defense_order_sha256": order_hash,
        "legacy_B_token_evidence": "All7 guarded workers recorded equality to B's actual tokenizer/collation; legacy B stores text hashes and original source, not token batch hash fields.",
    }


def load_standard(entry):
    folder, static = Path(entry["case_folder"]), Path(entry["static_folder"])
    require(
        read(folder / "status/train.json")["state"] == "complete",
        f"{entry['cell']}: defense training pending",
    )
    require(
        read(folder / "status/static.json")["state"] == "complete",
        f"{entry['cell']}: static measurements pending",
    )
    case = read(folder / "case.json")
    require(
        case["role"] == "defense"
        and case["max_steps"] == 125
        and case["defense_hyperparameters"] == parameters(entry["cell"]),
        f"{entry['cell']}: invalid defense declaration",
    )
    proof = read(folder / "training-validation.json")
    require(
        proof["state"] == "passed" and proof["updates"] == 125,
        f"{entry['cell']}: defense proof failed",
    )
    order = proof["input_order_validation"]
    require(
        order["state"] == "passed"
        and order["batches"] == 125
        and len(order["canonical_step_batch_sha256"]) == 64,
        f"{entry['cell']}: exact batch guard missing",
    )
    require(
        Path(order["reference_observations"]).resolve()
        == Path(entry["baseline_observations"]).resolve(),
        f"{entry['cell']}: wrong batch guard reference",
    )
    protocol, validation = read(static / "protocol.json"), read(
        static / "validation.json"
    )
    require(
        validation["state"] == "passed" and validation["points"] == 648,
        f"{entry['cell']}: invalid static validation",
    )
    require(
        validation["grid_sha256"] == sha(static / "grid.csv")
        and validation["protocol_sha256"] == sha(static / "protocol.json"),
        f"{entry['cell']}: stale grid/protocol",
    )
    require(
        protocol["training_validation_sha256"]
        == sha(folder / "training-validation.json")
        and protocol["defense_config_sha256"] == sha(folder / "config.yml"),
        f"{entry['cell']}: stale training/config proof",
    )
    require(
        protocol["baseline_tokenizer_and_collation_match"]
        and protocol["default_dataset_fixed10_matches_pinned"],
        f"{entry['cell']}: baseline fixed inputs not verified",
    )
    require(
        len(protocol["evaluated_batch_sha256"]) == 5
        and all(len(value) == 64 for value in protocol["evaluated_batch_sha256"]),
        f"{entry['cell']}: incomplete token batch hashes",
    )
    checkpoint = (folder / "models/defense").resolve()
    require(
        Path(protocol["checkpoint"]).resolve() == checkpoint
        and protocol["checkpoint_files"] == file_identity(checkpoint),
        f"{entry['cell']}: checkpoint identity changed",
    )
    require(
        protocol["source_sha256"]["get_ll.py"] == sha(Path(entry["get_ll_source"])),
        f"{entry['cell']}: static source changed",
    )
    rows = validate_grid(static / "grid.csv")
    return {
        "cell": entry["cell"],
        "parameters": parameters(entry["cell"]),
        "common": normalized_metadata(protocol),
        "rows": rows,
        "batch_hashes": protocol["evaluated_batch_sha256"],
        "order_sha256": order["canonical_step_batch_sha256"],
        "proof": {
            "case_folder": str(folder),
            "static_folder": str(static),
            "protocol_sha256": sha(static / "protocol.json"),
            "grid_sha256": sha(static / "grid.csv"),
            "training_validation_sha256": sha(folder / "training-validation.json"),
            "checkpoint_files": protocol["checkpoint_files"],
        },
    }


def load_legacy(entry):
    original, static, recorded = (
        Path(entry[key])
        for key in ("original_run", "static_folder", "baseline_dynamics")
    )
    require(
        read(original / "stage-status/defense.json")["state"] == "complete",
        "B: legacy defense incomplete",
    )
    training = read(original / "training_validation.json")["defense"]
    require(
        training["global_step"] == 125
        and training["inner_steps"] == 2000
        and training["all_finite"],
        "B: legacy defense proof failed",
    )
    recorded_proof = read(recorded / "training-validation.json")["defense"]
    require(
        recorded_proof["global_step"] == recorded_proof["observed_batches"] == 125
        and recorded_proof["all_metrics_finite"]
        and recorded_proof["max_metric_difference_from_original"] == 0,
        "B: recorded legacy defense proof failed",
    )
    require(
        read(recorded / "status/defense.json")["state"] == "complete",
        "B: recorded defense incomplete",
    )
    require(
        read(recorded / "geometry/geometry.json")["rerun_distance_from_original"][
            "defense"
        ]
        == 0,
        "B: recorded endpoint does not equal original",
    )
    require(
        read(static / "status.json")["state"] == "complete", "B: static grid incomplete"
    )
    metadata, validation = read(static / "provenance.json"), read(
        static / "validation.json"
    )
    require(
        validation["state"] == "passed" and validation["points"] == 648,
        "B: legacy static validation failed",
    )
    require(
        metadata["artifact_sha256"]["grid.csv"] == sha(static / "grid.csv"),
        "B: static grid changed",
    )
    require(
        file_identity(metadata["checkpoint"]) == entry["checkpoint_files"],
        "B: checkpoint identity changed since comparison design",
    )
    require(
        metadata["source_sha256"]["get_ll.py"] == sha(Path(entry["get_ll_source"])),
        "B: original static source differs",
    )
    return {
        "cell": "B",
        "parameters": parameters("B"),
        "common": normalized_metadata(metadata, legacy=True),
        "rows": validate_grid(static / "grid.csv"),
        "proof": {
            "static_folder": str(static),
            "grid_sha256": sha(static / "grid.csv"),
            "checkpoint_files": entry["checkpoint_files"],
            "legacy_training_validation": str(original / "training_validation.json"),
            "recorded_validation": str(recorded / "training-validation.json"),
            "recorded_endpoint_distance_from_original": 0,
        },
    }


def load_records(spec):
    for path, expected in spec.get("pinned_files", {}).items():
        require(sha(path) == expected, f"Previously completed input changed: {path}")
    records = [
        load_legacy(entry) if entry["cell"] == "B" else load_standard(entry)
        for entry in spec["cells"]
    ]
    original_rows = next(record["rows"] for record in records if record["cell"] == "B")
    for record in records:
        record["rows"], record["proof"]["coordinate_validation"] = canonicalize(
            record["rows"], original_rows
        )
    validation = validate_records(records)
    return sorted(records, key=lambda record: CELLS.index(record["cell"])), validation


def common_limits(values, symmetric=False):
    if symmetric:
        bound = max(max(map(abs, values)) * 1.05, 1e-5)
        return -bound, bound
    low, high = min(values), max(values)
    pad = max((high - low) * 0.05, 1e-5)
    return low - pad, high + pad


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, colors

    scales = {}
    for radius in RADII:
        subsets = [
            [row for row in record["rows"] if row["radius"] == radius]
            for record in records
        ]
        centers = [grid_metrics(rows, radius)["center_ce"] for rows in subsets]
        absolute = common_limits([row["loss"] for rows in subsets for row in rows])
        relative = common_limits(
            [
                row["loss"] - center
                for rows, center in zip(subsets, centers, strict=True)
                for row in rows
            ],
            symmetric=True,
        )
        scales[str(radius)] = {
            "absolute_ce": absolute,
            "center_relative_ce": relative,
            "coordinate_limits": [-radius, radius],
            "camera": {"elevation": 27, "azimuth": -58},
        }
        for kind, limits, cmap in (
            ("absolute", absolute, "viridis"),
            ("delta", relative, "coolwarm"),
        ):
            figure, axes = plt.subplots(
                2,
                4,
                figsize=(24, 12),
                subplot_kw={"projection": "3d"} if kind == "absolute" else {},
            )
            figure.subplots_adjust(
                left=0.035, right=0.92, bottom=0.09, top=0.84, wspace=0.26, hspace=0.36
            )
            for axis, record, rows, center in zip(
                axes.flat, records, subsets, centers, strict=True
            ):
                coordinates = sorted({row["c0"] for row in rows})
                lookup = {(row["c0"], row["c1"]): row["loss"] for row in rows}
                x, y = np.meshgrid(coordinates, coordinates)
                z = np.array(
                    [[lookup[(c0, c1)] for c0 in coordinates] for c1 in coordinates]
                )
                if kind == "absolute":
                    axis.plot_surface(
                        x,
                        y,
                        z,
                        cmap=cmap,
                        vmin=limits[0],
                        vmax=limits[1],
                        linewidth=0,
                        antialiased=True,
                    )
                    axis.view_init(elev=27, azim=-58)
                    axis.set_zlim(*limits)
                    axis.set_zlabel("Forget CE (nats)", labelpad=8)
                    axis.set_box_aspect((1, 1, 0.72))
                else:
                    z -= center
                    levels = np.linspace(*limits, 17)
                    axis.contourf(
                        x,
                        y,
                        z,
                        levels=levels,
                        cmap=cmap,
                        vmin=limits[0],
                        vmax=limits[1],
                    )
                    if np.ptp(z) > 1e-8:
                        axis.contour(
                            x,
                            y,
                            z,
                            levels=levels,
                            colors="black",
                            linewidths=0.35,
                            alpha=0.4,
                        )
                    axis.set_aspect("equal")
                values = record["parameters"]
                axis.set_title(
                    f"{record['cell']} | retain {values['retain_coefficient']:g}\ninner {values['inner_learning_rate']:g} | outer {values['outer_learning_rate']:g}\ncenter CE {center:.4f}",
                    fontsize=10,
                    pad=13,
                )
                axis.set_xlim(-radius, radius)
                axis.set_ylim(-radius, radius)
                axis.set_xlabel("Static random coordinate 1", labelpad=7)
                axis.set_ylabel("Static random coordinate 2", labelpad=7)
                axis.tick_params(labelsize=8, pad=2)
            title = (
                "Absolute CE | common z and color limits across all 8 controlled cells"
                if kind == "absolute"
                else "CE minus each actual center | common symmetric color limits across all 8 cells"
            )
            figure.suptitle(
                f"Original BF16 static recipe | half-width {radius:g}\n{title}",
                fontsize=17,
                y=0.965,
            )
            figure.text(
                0.48,
                0.025,
                "Top row: outer LR 5e-6. Bottom row: outer LR 1e-5. Same seed 42 directions/coordinates; independent from FP32 orthonormal dynamic slices.",
                ha="center",
                fontsize=11,
            )
            color_axis = figure.add_axes([0.945, 0.19, 0.012, 0.55])
            figure.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(*limits), cmap=cmap),
                cax=color_axis,
                label=(
                    "CE (nats)" if kind == "absolute" else "Center-relative CE (nats)"
                ),
            )
            for extension in ("png", "pdf"):
                figure.savefig(
                    output / f"radius-{radius:g}-{kind}.{extension}",
                    dpi=160,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
            plt.close(figure)
    return scales


def report(run, spec):
    records, validation = load_records(spec)
    output = run / "static/full-factorial"
    output.mkdir(parents=True, exist_ok=True)
    metric_rows, contrast_rows = [], []
    for radius in RADII:
        metrics = {}
        for record in records:
            values = grid_metrics(record["rows"], radius)
            metrics[record["cell"]] = values
            metric_rows.append(
                {
                    "cell": record["cell"],
                    "radius": radius,
                    **record["parameters"],
                    **values,
                }
            )
        for metric in METRICS:
            for row in factorial_contrasts(
                {cell: values[metric] for cell, values in metrics.items()}
            ):
                contrast_rows.append({"radius": radius, "metric": metric, **row})
    write_csv(output / "grid-metrics.csv", metric_rows)
    write_csv(output / "factorial-contrasts.csv", contrast_rows)
    scales = render(records, output)
    lines = [
        "# Eight-cell controlled static comparison",
        "",
        "R changes retain coefficient 2.5→4, I changes inner learning rate 1e-6→1e-5, and O changes outer learning rate 5e-6→1e-5. All 8 cells keep the original local sampler, K16/125 defense updates, source, data order and seeds fixed. Only defense endpoints and cached static grids are required; no new attacks or benchmarks enter this report.",
        "",
        "These figures use the exact original BF16 random recipe and default attention: seed 42, the fixed set of 10 texts, truncation 512, batch size 2, independent globally normalized BF16 directions and sequential BF16 perturbations. They are separate from the newer FP32 orthonormal dynamic views. Each radius shares physical coordinates, orientation, absolute CE/z limits and camera across all 8 controlled cells. Center-relative views subtract each cell's measured center and share a symmetric color scale. Radii have separate scales. The earlier uploaded HF contextual comparison is outside this controlled factorial.",
        "",
        "The rim/elevation names below refer only to finite sampled grids. These translated slices evaluate a fixed forget CE objective; they do not establish global barriers, Hessian curvature or robustness. One seed and no replication support descriptive contrasts only; there are no p-values.",
        "",
        "## Prespecified metrics and formulas",
        "",
    ]
    lines += [f"- **{key}**: {value}" for key, value in DEFINITIONS.items()]
    lines += [
        "",
        "All reported differences have units of CE nats. Anchored coefficients and conditional differences have divisor 1. Marginal main contrasts average 4 conditional differences; marginal pair contrasts average 2. They therefore use distinct, explicitly recorded normalizations. The three-way coefficient is also the change of a conditional pair difference between the two settings of its third factor.",
        "",
        "## Focus views",
        "",
        "| Half-width | Absolute CE | Center-relative CE |",
        "|---:|---|---|",
    ]
    for radius in (125, 100, *[item for item in RADII if item not in (125, 100)]):
        lines.append(
            f"| {radius:g} | [PNG](radius-{radius:g}-absolute.png) / [PDF](radius-{radius:g}-absolute.pdf) | [PNG](radius-{radius:g}-delta.png) / [PDF](radius-{radius:g}-delta.pdf) |"
        )
    for radius in (125, 100):
        lines += [
            "",
            f"## Finite-grid diagnostics at half-width {radius:g}",
            "",
            "| Cell | Retain | Inner LR | Outer LR | Center CE | Rim rise | Center elevation |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metric_rows:
            if row["radius"] == radius:
                lines.append(
                    f"| {row['cell']} | {row['retain_coefficient']:g} | {row['inner_learning_rate']:g} | {row['outer_learning_rate']:g} | {row['center_ce']:.6f} | {row['rim_rise']:.6f} | {row['center_elevation']:.6f} |"
                )
        lines += [
            "",
            "Anchored polynomial coefficients (other factors at 0; raw differences, divisor 1):",
            "",
            "| Term | Center CE | Rim rise | Center elevation |",
            "|---|---:|---:|---:|",
        ]
        for factors in ("R", "I", "O", "RI", "RO", "IO", "RIO"):
            selected = {
                row["metric"]: row["value"]
                for row in contrast_rows
                if row["radius"] == radius
                and row["kind"] == "anchored_coefficient"
                and row["factors"] == factors
            }
            lines.append(
                f"| {factors} | "
                + " | ".join(f"{selected[metric]:.6f}" for metric in METRICS)
                + " |"
            )
    lines += [
        "",
        "[All radii grid metrics](grid-metrics.csv) and [all anchored, conditional and marginal contrasts](factorial-contrasts.csv) retain the measured values. The normalization column is the divisor used for averaging conditional differences; anchored and conditional differences are unscaled. Signs describe differences at these two chosen levels, not a fitted dose response.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    dump(
        output / "validation.json",
        {
            "state": "complete",
            "cells": 8,
            "points_per_cell": 648,
            **validation,
            "definitions": DEFINITIONS,
            "scales": scales,
            "source_sha256": sha(Path(__file__)),
            "input_spec_sha256": sha(Path(__file__).parent / "inputs.json"),
            "proofs": {record["cell"]: record["proof"] for record in records},
            "artifacts": {
                path.name: sha(path)
                for path in sorted(output.iterdir())
                if path.suffix in {".csv", ".png", ".pdf", ".md"}
            },
        },
    )
    dump(
        run / "status/static-factorial-report.json",
        {"state": "complete", "cells": 8, "figures": 16, "points_per_cell": 648},
    )
    print(f"Complete controlled 8-cell static comparison: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "report"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    spec = read(Path(__file__).parent / "inputs.json")
    require(
        Path(spec["run"]).resolve() == args.run.resolve(),
        "Input specification belongs to another run",
    )
    if args.mode == "validate":
        records, validation = load_records(spec)
        print(
            json.dumps(
                {"state": "passed", "cells": len(records), **validation}, indent=2
            )
        )
    else:
        require(
            bool(os.environ.get("FLUX_JOB_ID")),
            "Allocated launcher job required for rendering",
        )
        report(args.run.resolve(), spec)


if __name__ == "__main__":
    main()
