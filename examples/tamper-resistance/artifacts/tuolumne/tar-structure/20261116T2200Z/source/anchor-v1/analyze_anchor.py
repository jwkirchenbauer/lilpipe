"""Postprocess local trajectory Grams under alternate defense anchors.

This script reads only the small aggregate Gram matrices and per-parameter scan
caches created by tar_basis_structure.py. It never opens model checkpoint
tensors. All reported subspaces describe parameter displacement geometry, not
protected-loss directions.
"""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text())


def publish_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_helpers(run):
    frozen = Path(run) / "source/v1/tar_basis_structure.py"
    spec = importlib.util.spec_from_file_location("frozen_tar_basis_structure", frozen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fit_subspace, module.overlap, module.classify_parameter


def embed_coefficients(coefficients, transform, total_snapshots):
    """Map transformed-snapshot coefficients back to original snapshot space."""
    result = transform.T @ np.asarray(coefficients, dtype=np.float64)
    if result.shape[0] != total_snapshots:
        raise ValueError("Transformed coefficient dimension mismatch")
    return result


def transformed_subspace(gram, transform, fit_subspace, rank=None):
    """Fit in an algebraically transformed snapshot basis without tensor reads."""
    gram = np.asarray(gram, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    transformed_gram = transform @ gram @ transform.T
    coefficients, info = fit_subspace(
        transformed_gram, np.arange(transform.shape[0]), rank
    )
    mapped = embed_coefficients(coefficients, transform, gram.shape[0])
    np.testing.assert_allclose(
        mapped.T @ gram @ mapped, np.eye(mapped.shape[1]), atol=1e-6
    )
    return mapped, info


def anchor_transforms(total_snapshots, defense_indices):
    """Return start, endpoint, and last-three-interval coordinate maps."""
    if len(defense_indices) != 9:
        raise ValueError("Expected the nine local defense snapshots")
    if list(defense_indices) != list(range(9)):
        raise ValueError("Local defense snapshots must occupy indices 0 through 8")
    endpoint = np.zeros((9, total_snapshots), dtype=np.float64)
    endpoint[0, 8] = -1.0
    for row, source in enumerate(range(8), start=1):
        endpoint[row, source] = 1.0
        endpoint[row, 8] = -1.0
    late = np.zeros((3, total_snapshots), dtype=np.float64)
    for row, (left, right) in enumerate(((5, 6), (6, 7), (7, 8))):
        late[row, left] = -1.0
        late[row, right] = 1.0
    return endpoint, late


def make_views(gram, fit_subspace, defense_indices, attack_indices):
    endpoint_transform, late_transform = anchor_transforms(
        gram.shape[0], defense_indices
    )
    views = {}
    infos = {}
    for rank, suffix in ((2, "plane"), (None, "span")):
        views[f"start-defense-{suffix}"], infos[f"start-defense-{suffix}"] = (
            fit_subspace(gram, defense_indices, rank)
        )
        views[f"endpoint-defense-{suffix}"], infos[f"endpoint-defense-{suffix}"] = (
            transformed_subspace(gram, endpoint_transform, fit_subspace, rank)
        )
        views[f"late-interval-{suffix}"], infos[f"late-interval-{suffix}"] = (
            transformed_subspace(gram, late_transform, fit_subspace, rank)
        )
    for rank, suffix in ((2, "plane"), (None, "span")):
        views[f"local-attack-{suffix}"], infos[f"local-attack-{suffix}"] = fit_subspace(
            gram, attack_indices, rank
        )
    return views, infos


def projected_fraction(gram, basis, snapshot_index):
    vector = np.asarray(gram[snapshot_index], dtype=np.float64)
    norm = float(gram[snapshot_index, snapshot_index])
    if norm <= 0:
        return 1.0
    metric = basis.T @ gram @ basis
    cross = vector @ basis
    power = float(cross @ np.linalg.pinv(metric) @ cross)
    return float(np.clip(power / norm, 0.0, 1.0))


def compute_overlaps(gram, overlap, views, precision):
    rows = []
    target_names = ("local-attack-plane", "local-attack-span")
    anchor_names = (
        "start-defense-plane",
        "start-defense-span",
        "endpoint-defense-plane",
        "endpoint-defense-span",
        "late-interval-plane",
        "late-interval-span",
    )
    for anchor in anchor_names:
        for target in target_names:
            value = overlap(gram, views[anchor], views[target])
            rows.append(
                {
                    "precision": precision,
                    "anchor_subspace": anchor,
                    "attack_subspace": target,
                    "left_rank": value["left_rank"],
                    "right_rank": value["right_rank"],
                    "angles_degrees": json.dumps(value["angles_degrees"]),
                    "mean_cosine_squared": value["mean_cosine_squared"],
                }
            )
    invariant = overlap(
        gram, views["start-defense-span"], views["endpoint-defense-span"]
    )
    return rows, invariant


def category_energy_rows(
    run, case_dir, layout, gram, views, precision, classify_parameter
):
    rows = []
    totals = {
        name: float(np.trace(coeff.T @ gram @ coeff))
        for name, coeff in views.items()
        if name.endswith("-plane")
    }
    category_accumulator = {name: {} for name in totals}
    total_parameters = int(layout["parameters"])
    cache_dir = Path(case_dir) / "cache/scan"
    for index, name in enumerate(layout["names"]):
        category, layer, component = classify_parameter(name)
        count = math.prod(layout["shapes"][name])
        cache_path = cache_dir / f"{index:03}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path) as block:
            parameter_gram = np.asarray(block[precision], dtype=np.float64)
        if parameter_gram.shape != gram.shape:
            raise ValueError(f"Cached Gram shape mismatch for {name} ({precision})")
        for view_name, coefficient in views.items():
            if not view_name.endswith("-plane"):
                continue
            energy = float(np.trace(coefficient.T @ parameter_gram @ coefficient))
            total = totals[view_name]
            fraction = energy / total if total else 0.0
            enrichment = fraction / (count / total_parameters) if total else 0.0
            rows.append(
                {
                    "precision": precision,
                    "view": view_name,
                    "scope": "parameter",
                    "name": name,
                    "category": category,
                    "layer": layer,
                    "component": component,
                    "parameters": count,
                    "energy": energy,
                    "energy_fraction": fraction,
                    "parameter_fraction": count / total_parameters,
                    "enrichment": enrichment,
                    "rms_displacement": math.sqrt(energy / count) if count else 0.0,
                }
            )
            aggregate = category_accumulator[view_name].setdefault(
                category, {"energy": 0.0, "parameters": 0}
            )
            aggregate["energy"] += energy
            aggregate["parameters"] += count
    for view_name, categories in category_accumulator.items():
        total = totals[view_name]
        for category, value in sorted(categories.items()):
            count, energy = value["parameters"], value["energy"]
            fraction = energy / total if total else 0.0
            rows.append(
                {
                    "precision": precision,
                    "view": view_name,
                    "scope": "category",
                    "name": category,
                    "category": category,
                    "layer": "all",
                    "component": "all",
                    "parameters": count,
                    "energy": energy,
                    "energy_fraction": fraction,
                    "parameter_fraction": count / total_parameters,
                    "enrichment": (
                        fraction / (count / total_parameters)
                        if total and count
                        else 0.0
                    ),
                    "rms_displacement": math.sqrt(energy / count) if count else 0.0,
                }
            )
    return rows


def self_test(fit_subspace, overlap):
    deltas = np.random.default_rng(17).normal(size=(9, 6))
    gram = deltas @ deltas.T
    endpoint_map, late_map = anchor_transforms(9, range(9))
    start, start_info = fit_subspace(gram, np.arange(9), None)
    end, end_info = transformed_subspace(gram, endpoint_map, fit_subspace, None)
    result = overlap(gram, start, end)
    assert start_info["rank"] == end_info["rank"]
    assert result["mean_cosine_squared"] > 1.0 - 1e-10
    assert max(result["angles_degrees"]) < 1e-3
    np.testing.assert_allclose(end.T @ gram @ end, np.eye(end.shape[1]), atol=1e-8)
    late, _ = transformed_subspace(gram, late_map, fit_subspace, None)
    assert late.shape[1] == 3
    np.testing.assert_allclose(late.T @ gram @ late, np.eye(3), atol=1e-8)
    print("PASS transformed anchor span preserves full observed span")
    print("PASS late-interval transform returns an orthonormal restricted span")


def make_figure(output, overlap_rows, capture_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precisions = ("raw", "evaluated")
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    anchor_rows = [
        "start-defense-plane",
        "endpoint-defense-plane",
        "late-interval-plane",
        "start-defense-span",
        "endpoint-defense-span",
        "late-interval-span",
    ]
    for axis, precision in zip(axes[:2], precisions):
        values = np.full((len(anchor_rows), 2), np.nan)
        for row in overlap_rows:
            if row["precision"] != precision:
                continue
            col = 0 if row["attack_subspace"] == "local-attack-plane" else 1
            values[anchor_rows.index(row["anchor_subspace"]), col] = row[
                "mean_cosine_squared"
            ]
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_yticks(range(len(anchor_rows)), anchor_rows)
        axis.set_xticks([0, 1], ["attack plane", "attack span"])
        axis.set_title(f"{precision.title()} principal-subspace overlap")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(
                    j, i, f"{values[i,j]:.3f}", ha="center", va="center", color="white"
                )
    fig.colorbar(image, ax=list(axes[:2]), label="mean cos² principal angle")
    axis = axes[2]
    candidates = (
        "start-defense-plane",
        "endpoint-defense-plane",
        "late-interval-plane",
        "start-defense-span",
        "endpoint-defense-span",
        "late-interval-span",
    )
    for name in candidates:
        points = [row for row in capture_rows if row["anchor_subspace"] == name]
        axis.plot(
            [row["attack_step"] for row in points],
            [row["captured_fraction"] for row in points],
            marker="o",
            markersize=3,
            label=name,
        )
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Saved attack update")
    axis.set_ylabel("Captured displacement energy")
    axis.set_title("Attack snapshots projected onto defense spans")
    axis.legend(fontsize=7, loc="best")
    fig.suptitle("Parameter-displacement geometry only; not protected-loss subspaces")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def analyze(run):
    run = Path(run).resolve()
    case_dir = run / "cases/local-defense-attack"
    status = read_json(case_dir / "status/scan.json")
    if status.get("state") != "complete":
        raise RuntimeError("Local status/scan.json must be complete before analysis")
    job = read_json(case_dir / "job.json")
    layout = read_json(job["layout"])
    with np.load(case_dir / "gram.npz") as data:
        grams = {
            key: np.asarray(data[key], dtype=np.float64) for key in ("raw", "evaluated")
        }
    if grams["raw"].shape[0] != 25 or grams["evaluated"].shape[0] != 23:
        raise ValueError(
            "Unexpected local Gram dimensions; expected raw 25x25/evaluated 23x23"
        )
    snapshots = job["snapshots"]
    if [item["step"] for item in snapshots[:9]] != [1, 2, 4, 8, 16, 32, 64, 96, 125]:
        raise ValueError("Unexpected local defense snapshot schedule")
    if [item["step"] for item in snapshots[9:22]] != [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1000,
    ]:
        raise ValueError("Unexpected local attack snapshot schedule")
    if job.get("require_zero_snapshots") != [22]:
        raise ValueError("Expected the zero anchor bridge at raw/evaluated row 22")
    fit_subspace, overlap, classify_parameter = load_frozen_helpers(run)
    defense_indices, attack_indices = list(range(9)), list(range(9, 22))
    output_dir = run / "anchor-sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_rows, capture_rows, invariant_rows, view_infos = [], [], [], {}
    parameter_rows = []
    for precision, gram in grams.items():
        views, infos = make_views(gram, fit_subspace, defense_indices, attack_indices)
        view_infos[precision] = infos
        rows, invariant = compute_overlaps(gram, overlap, views, precision)
        overlap_rows.extend(rows)
        invariant_rows.append(
            {
                "precision": precision,
                "start_rank": invariant["left_rank"],
                "endpoint_rank": invariant["right_rank"],
                "mean_cosine_squared": invariant["mean_cosine_squared"],
                "angles_degrees": invariant["angles_degrees"],
                "verified": invariant["left_rank"] == invariant["right_rank"]
                and invariant["mean_cosine_squared"] > 1.0 - 1e-8
                and max(invariant["angles_degrees"], default=0.0) < 0.01,
            }
        )
        for view_name, coefficient in views.items():
            if not (view_name.endswith("-plane") or view_name.endswith("-span")):
                continue
            for snapshot_index in attack_indices:
                capture_rows.append(
                    {
                        "precision": precision,
                        "anchor_subspace": view_name,
                        "attack_step": snapshots[snapshot_index]["step"],
                        "captured_fraction": projected_fraction(
                            gram, coefficient, snapshot_index
                        ),
                    }
                )
        parameter_rows.extend(
            category_energy_rows(
                run,
                case_dir,
                layout,
                gram,
                views,
                precision,
                classify_parameter,
            )
        )
    write_csv(output_dir / "attack-overlaps.csv", overlap_rows)
    write_csv(output_dir / "attack-capture.csv", capture_rows)
    write_csv(output_dir / "parameter-plane-energy.csv", parameter_rows)
    publish_json(
        output_dir / "anchor-analysis.json",
        {
            "state": "complete",
            "input": str(case_dir / "gram.npz"),
            "cache_scope": "small aggregate Gram and per-parameter Gram caches only; no checkpoint tensors opened",
            "source": str(Path(__file__).resolve()),
            "definitions": {
                "start_anchor": "Uncentered saved displacement snapshots D0..D8 from the original phase start.",
                "endpoint_anchor": "Nine vectors -D8 and D_i-D8 for i=0..7, including the original phase-start base point.",
                "late_intervals": "Only D6-D5, D7-D6, D8-D7 (32->64, 64->96, 96->125); not the whole defense trajectory.",
                "attack": "Uncentered local K16 attack displacement snapshots D9..D21 (13 saved updates).",
                "interpretation": "These are parameter-displacement subspaces, not protected-loss subspaces or a robustness guarantee.",
                "full_span_invariance": "The full observed defense span is expected to be invariant under start/end anchoring because the endpoint transform is invertible; late intervals are a restricted local subspace.",
            },
            "spectra": view_infos,
            "start_endpoint_full_span_invariance": invariant_rows,
        },
    )
    make_figure(output_dir / "anchor-sensitivity.png", overlap_rows, capture_rows)
    if not all(row["verified"] for row in invariant_rows):
        raise AssertionError("Full defense span did not verify as anchor invariant")
    print(f"Anchor sensitivity written to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run = Path(__file__).resolve().parents[2]
        fit_subspace, overlap, _ = load_frozen_helpers(run)
        self_test(fit_subspace, overlap)
        return
    if not args.run:
        parser.error("run directory required")
    analyze(args.run)


if __name__ == "__main__":
    main()
