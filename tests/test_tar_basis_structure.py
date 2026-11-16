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
