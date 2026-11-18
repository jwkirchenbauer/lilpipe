import importlib.util
import json
import math
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_basis_structure.py"
)
SPEC = importlib.util.spec_from_file_location("tar_basis_structure", MODULE_PATH)
BASIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASIS)


def test_fit_subspace_returns_orthonormal_coefficients_and_restricted_rank():
    # Three snapshots span two directions; the fourth snapshot is excluded.
    deltas = np.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [5.0, 5.0, 5.0]]
    )
    gram = deltas @ deltas.T

    full, info = BASIS.fit_subspace(gram, [0, 1, 2])
    restricted, restricted_info = BASIS.fit_subspace(gram, [0, 1, 2], rank=1)

    assert info["rank"] == 2
    assert info["used_rank"] == 2
    assert info["snapshot_count"] == 3
    assert math.isclose(info["captured_snapshot_energy"], 1.0, abs_tol=1e-12)
    np.testing.assert_allclose(full.T @ gram @ full, np.eye(2), atol=1e-10)
    assert np.all(full[3] == 0)
    assert restricted.shape == (4, 1)
    assert restricted_info["rank"] == 2
    assert restricted_info["used_rank"] == 1
    assert 0 < restricted_info["captured_snapshot_energy"] < 1
    np.testing.assert_allclose(restricted.T @ gram @ restricted, np.eye(1), atol=1e-10)


def test_overlap_parallel_orthogonal_and_rotated_subspaces():
    gram = np.eye(4)
    left = np.eye(4, dtype=np.float64)[:, :2]
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    parallel_rotated = left @ rotation
    orthogonal = np.eye(4, dtype=np.float64)[:, 2:]

    same = BASIS.overlap(gram, left, left)
    rotated = BASIS.overlap(gram, left, parallel_rotated)
    disjoint = BASIS.overlap(gram, left, orthogonal)

    assert same["left_rank"] == same["right_rank"] == 2
    np.testing.assert_allclose(same["angles_degrees"], [0.0, 0.0], atol=1e-6)
    assert math.isclose(same["mean_cosine_squared"], 1.0, abs_tol=1e-12)
    np.testing.assert_allclose(rotated["angles_degrees"], [0.0, 0.0], atol=1e-6)
    assert math.isclose(rotated["mean_cosine_squared"], 1.0, abs_tol=1e-12)
    np.testing.assert_allclose(disjoint["angles_degrees"], [90.0, 90.0], atol=1e-6)
    assert math.isclose(disjoint["mean_cosine_squared"], 0.0, abs_tol=1e-12)


def test_overlap_whitens_restricted_rank_one_directions():
    gram = np.eye(3)
    theta = np.deg2rad(60.0)
    left = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    right = np.array([[np.cos(theta), 0.0], [np.sin(theta), 0.0], [0.0, 0.0]])

    result = BASIS.overlap(gram, left, right)

    assert result["left_rank"] == result["right_rank"] == 1
    np.testing.assert_allclose(result["angles_degrees"], [60.0], atol=1e-6)
    assert math.isclose(result["mean_cosine_squared"], 0.25, abs_tol=1e-12)


def test_concentration_histogram_bounds_contain_exact_sorted_support():
    # Each positive synthetic energy falls inside its stated half-open bin.
    energies = np.array([0.5, 1.5, 3.0, 7.0, 15.0, 0.0, 0.0])
    edges = np.array([0.1, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    positive = energies[energies > 0]
    bins = np.searchsorted(edges, positive, side="right") - 1
    counts = np.bincount(bins, minlength=len(edges) - 1)
    masses = np.bincount(bins, weights=positive, minlength=len(edges) - 1)
    result = BASIS.concentration(
        counts, masses, edges, len(energies), float(np.square(energies).sum())
    )
    total = float(positive.sum())
    sorted_energy = np.sort(positive)[::-1]

    for target in (0.5, 0.9, 0.99, 0.999):
        exact_count = int(np.searchsorted(np.cumsum(sorted_energy), target * total) + 1)
        exact_fraction = exact_count / len(energies)
        bounds = result["energy_targets"][str(target)]
        assert bounds["coordinate_fraction_lower"] <= exact_fraction
        assert exact_fraction <= bounds["coordinate_fraction_upper"]

    for fraction in (1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0):
        take = min(math.ceil(fraction * len(energies)), len(positive))
        exact_energy_fraction = (
            float(sorted_energy[:take].sum() / total) if take else 0.0
        )
        bounds = result["top_fractions"][str(fraction)]
        assert bounds["energy_fraction_lower"] <= exact_energy_fraction + 1e-12
        assert exact_energy_fraction <= bounds["energy_fraction_upper"] + 1e-12

    assert result["nonzero_coordinates"] == 5
    assert math.isclose(result["nonzero_fraction"], 5 / 7)


def test_temporal_rank_and_coordinate_support_are_distinct():
    dense_rank_one = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    sparse_rank_two = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    assert np.linalg.matrix_rank(dense_rank_one) == 1
    assert np.count_nonzero(dense_rank_one) / dense_rank_one.size == 1.0
    assert np.linalg.matrix_rank(sparse_rank_two) == 2
    assert np.count_nonzero(sparse_rank_two) / sparse_rank_two.size == 0.25


def test_saved_displacement_energy_preserves_magnitudes_and_excursions():
    import torch

    # The final checkpoint has returned to coordinate zero, after a large excursion.
    # An unrelated phase in row 3 must not contribute to this trajectory.
    snapshots = torch.tensor(
        [[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 100.0]],
        dtype=torch.float64,
    )
    trajectory = BASIS.displacement_energy(snapshots, [0, 1, 2]).numpy()
    endpoint = BASIS.displacement_energy(snapshots, [2]).numpy()
    np.testing.assert_array_equal(trajectory, [18.0, 5.0, 0.0])
    np.testing.assert_array_equal(endpoint, [0.0, 4.0, 0.0])
    assert math.isclose(trajectory.sum(), np.trace((snapshots @ snapshots.T)[:3, :3]))
    # Scaling the dominant excursion changes path weighting, but not its span.
    scaled = snapshots.clone()
    scaled[0] *= 10
    np.testing.assert_array_equal(
        BASIS.displacement_energy(scaled, [0, 1, 2]).numpy(), [909.0, 5.0, 0.0]
    )


def test_displacement_energy_casts_endpoints_before_subtraction():
    import torch

    anchor = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    saved = torch.tensor([[1.001, 1.01], [1.002, 1.02]], dtype=torch.float64)
    raw = saved - anchor
    evaluated = saved.to(torch.bfloat16).double() - anchor.to(torch.bfloat16).double()
    assert BASIS.displacement_energy(raw, [0, 1])[0] > 0
    assert BASIS.displacement_energy(evaluated, [0, 1])[0] == 0
    np.testing.assert_array_equal(
        BASIS.displacement_energy(evaluated, [0, 1]).numpy(),
        np.square(evaluated.numpy()).sum(axis=0),
    )


def test_concentration_comparison_aggregates_dense_trajectory_and_endpoint(
    tmp_path, monkeypatch
):
    import contextlib
    import csv

    import torch

    folder = tmp_path / "case"
    folder.mkdir()
    anchor = torch.zeros(4, dtype=torch.float64)
    raw = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [2.0, 1.0, 1.0, 0.0], [1.0, 0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )
    evaluated = raw.to(torch.bfloat16).double()
    snapshots = [
        {"name": f"save-{step}", "step": step, "anchor": "anchor", "path": f"p{step}"}
        for step in (1, 2, 4)
    ]
    job = {
        "layout": str(folder / "layout.json"),
        "snapshots": snapshots,
        "groups": [{"name": "attack", "indices": [0, 1, 2]}],
    }
    (folder / "job.json").write_text(json.dumps(job))
    original_declaration = {"job": job, "files": []}
    (folder / "input-identity.json").write_text(
        json.dumps({"declaration": original_declaration})
    )
    (folder / "layout.json").write_text(
        json.dumps({"names": ["weight"], "shapes": {"weight": [4]}, "parameters": 4})
    )
    (folder / "gram.npz").write_bytes(b"fixture")

    edges = np.logspace(-100, 100, 8193)

    def histogram(values):
        energy = np.square(values).sum(axis=0)
        positive = energy[energy > 0]
        bins = np.floor((np.log10(positive) + 100) * (8192 / 200)).astype(int)
        counts = np.bincount(bins, minlength=8192)
        masses = np.bincount(bins, weights=positive, minlength=8192)
        summary = BASIS.concentration(counts, masses, edges, 4, float(energy @ energy))
        return counts, masses, summary

    endpoint_data = {}
    view_names = []
    all_counts, all_masses = [], []
    rows = []
    for precision, values in (("raw", raw.numpy()), ("evaluated", evaluated.numpy())):
        name = f"save-4-{precision}-displacement"
        counts, masses, summary = histogram(values[-1:])
        endpoint_data[name] = summary
        view_names.append(name)
        all_counts.append(counts)
        all_masses.append(masses)
        rows.append({"view": name, "category": "mlp", "energy": summary["energy"]})
    (folder / "concentration.json").write_text(json.dumps({"views": endpoint_data}))
    np.savez(
        folder / "coordinate-histograms.npz",
        names=np.array(view_names),
        counts=np.array(all_counts),
        masses=np.array(all_masses),
    )
    with (folder / "parameter-structure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view", "category", "energy"))
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(
        BASIS,
        "identity",
        lambda current: ("signature", {"job": current, "files": []}),
    )
    monkeypatch.setattr(BASIS, "classify_parameter", lambda name: ("mlp",))
    monkeypatch.setattr(BASIS, "plot_concentration_comparison", lambda *args: None)

    @contextlib.contextmanager
    def synthetic_blocks(current, layout, start_parameter=0, include_external=True):
        assert start_parameter == 0
        yield iter([(0, "weight", 0, 4, 4, raw, evaluated, {})])

    monkeypatch.setattr(BASIS, "blocks", synthetic_blocks)
    BASIS.concentration_comparison(folder, job)

    result = json.loads(
        (folder / "displacement-concentration/results.json").read_text()
    )
    raw_trajectory = next(
        row
        for row in result["records"]
        if row["precision"] == "raw" and row["measure"] == "trajectory"
    )
    direct = np.square(raw.numpy()).sum(axis=0)
    ordered = np.sort(direct)[::-1]
    exact = int(np.searchsorted(np.cumsum(ordered), 0.9 * ordered.sum()) + 1) / 4
    bounds = raw_trajectory["energy_targets"]["0.9"]
    assert raw_trajectory["nonzero_coordinates"] == int(np.count_nonzero(direct))
    assert bounds["coordinate_fraction_lower"] <= exact
    assert exact <= bounds["coordinate_fraction_upper"]
    assert bounds["coordinate_fraction_upper"] <= 1
    validation = json.loads(
        (folder / "displacement-concentration/validation.json").read_text()
    )["checks"]
    assert all(
        item["endpoint_expected_energy"] == item["endpoint_scanned_energy"]
        for item in validation
    )


def test_fp32_plane_energy_is_invariant_to_rotation_within_plane():
    # Rows are sampled snapshots; columns 0 and 2 form the restricted parameter block.
    deltas = np.array(
        [[1.2, 0.3, -0.7, 0.1], [0.1, 1.1, 0.2, 0.5], [0.8, 1.4, -0.5, 0.9]],
        dtype=np.float32,
    )
    gram = deltas.astype(np.float64) @ deltas.astype(np.float64).T
    plane, _ = BASIS.fit_subspace(gram, [0, 1, 2], rank=2)
    angle = np.deg2rad(23.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    block_gram = (
        deltas[:, [0, 2]].astype(np.float64) @ deltas[:, [0, 2]].astype(np.float64).T
    )

    def fp32_plane_energy(coefficients):
        # Match the analyzer's FP32 projection before accumulating energy in FP64.
        projected = (coefficients.T @ deltas.astype(np.float64)).astype(np.float32)
        return float(np.square(projected[:, [0, 2]].astype(np.float64)).sum())

    direct = float(np.trace(plane.T @ block_gram @ plane))
    rotated = plane @ rotation
    rotated_energy = float(np.trace(rotated.T @ block_gram @ rotated))

    assert math.isclose(direct, rotated_energy, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        fp32_plane_energy(plane), fp32_plane_energy(rotated), rel_tol=1e-6, abs_tol=1e-6
    )


def _batch_row(ids, mask, labels, metadata_id):
    return ids, mask, labels, metadata_id


def test_token_frequencies_counts_masks_shifted_labels_and_role_filter(tmp_path):
    first = [
        _batch_row([1, 3, 0, 0], [1, 1, 0, 0], [-100, 3, -100, -100], 0),
        _batch_row([1, 4, 5, 0], [1, 1, 1, 0], [-100, 4, 5, -100], 1),
    ]
    second = [_batch_row([2, 2, 0], [1, 1, 0], [-100, 2, -100], 0)]
    batches = []
    for rows in (first, second):
        batches.append(
            {
                "batch": {
                    "input_ids": [row[0] for row in rows],
                    "attention_mask": [row[1] for row in rows],
                    "labels": [row[2] for row in rows],
                    "metadata_ids": [row[3] for row in rows],
                }
            }
        )
    path = tmp_path / "observed.jsonl"
    path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))

    all_counts, all_batches = BASIS.token_frequencies(path, vocabulary=8)
    forget_counts, forget_batches = BASIS.token_frequencies(path, vocabulary=8, role=0)
    retain_counts, retain_batches = BASIS.token_frequencies(path, vocabulary=8, role=1)

    assert all_batches == forget_batches == retain_batches == 2
    np.testing.assert_array_equal(all_counts["all_input"], [4, 2, 2, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(
        all_counts["attended_input"], [0, 2, 2, 1, 1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        all_counts["supervised_target"], [0, 0, 1, 1, 1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        all_counts["attended_target"], [0, 0, 1, 1, 1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        forget_counts["attended_input"], [0, 1, 2, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        forget_counts["supervised_target"], [0, 0, 1, 1, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        retain_counts["attended_input"], [0, 1, 0, 0, 1, 1, 0, 0]
    )
    np.testing.assert_array_equal(
        retain_counts["supervised_target"], [0, 0, 0, 0, 1, 1, 0, 0]
    )
