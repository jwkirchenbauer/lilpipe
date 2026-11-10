"""Check comparison math, partial endpoints, and evidence for shared slices."""

import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_compute_compare.py"
)
SPEC = importlib.util.spec_from_file_location("tar_compute_compare", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ComparisonMathTests(unittest.TestCase):
    def test_budget_maximum_includes_stronger_rates_and_earlier_checkpoints(self):
        records = {}
        for rate, score in (("1e-5", 60), ("1e-6", 35), ("5e-6", 58)):
            records[f"baseline-lr{rate}"] = {
                "case": {
                    "save_steps": [16, 128, 256, 512, 1000],
                    "attack_learning_rate": rate,
                },
                "scores": [
                    {"task": "wmdp_bio_robust", "attack_step": 0, "jwk_percent": 33},
                    {
                        "task": "wmdp_bio_robust",
                        "attack_step": 16,
                        "jwk_percent": score,
                    },
                    {
                        "task": "wmdp_bio_robust",
                        "attack_step": 128,
                        "jwk_percent": score - 1,
                    },
                    {"task": "mmlu_no_bio", "attack_step": 128, "jwk_percent": 99},
                ],
            }
        records["baseline-lr5e-6"]["scores"].append(
            {"task": "wmdp_bio_robust", "attack_step": 256, "jwk_percent": 61}
        )
        result = COMPARE.strongest_attacks(records)
        baseline = [row for row in result if row["defense_family"] == "baseline"]
        self.assertEqual(baseline[0]["peak_measured_wmdp"], 60)
        self.assertEqual(baseline[0]["peak_learning_rate"], "1e-5")
        self.assertEqual(baseline[0]["peak_attack_step"], 16)
        self.assertEqual(baseline[0]["measured_case_checkpoints"], 9)
        self.assertEqual(baseline[0]["expected_case_checkpoints"], 9)
        self.assertTrue(baseline[0]["complete_through_budget"])
        self.assertEqual(baseline[1]["peak_measured_wmdp"], 61)
        self.assertEqual(baseline[1]["peak_case"], "baseline-lr5e-6")
        self.assertEqual(baseline[1]["measured_case_checkpoints"], 10)
        self.assertEqual(baseline[1]["expected_case_checkpoints"], 12)
        self.assertFalse(baseline[1]["complete_through_budget"])
        long = next(row for row in result if row["defense_family"] == "long")
        self.assertIsNone(long["peak_measured_wmdp"])
        self.assertEqual(long["measured_learning_rates"], 0)
        self.assertFalse(long["complete_through_budget"])

    def test_overlap_uses_recorded_plane_order_and_marks_rank_zero(self):
        gram = np.eye(6)
        gram[:2, 4:6] = np.diag([0.8, 0.6])
        gram[4:6, :2] = gram[:2, 4:6].T
        geometry = {
            "attack_rank": 3,
            "planes": {"attack": [], "random": [], "reference": []},
            "basis_actual_gram": gram.tolist(),
        }
        value, note = COMPARE.plane_overlap(geometry)
        self.assertAlmostEqual(value, 0.5)
        self.assertEqual(note, "")
        geometry["attack_rank"] = 0
        value, note = COMPARE.plane_overlap(geometry)
        self.assertIsNone(value)
        self.assertIn("rank-zero", note)
        geometry["attack_rank"] = 1
        self.assertIn("random completion", COMPARE.plane_overlap(geometry)[1])

    def test_partial_surface_keeps_physical_coordinates_and_missing_points(self):
        radius = COMPARE.LOCAL_RADIUS
        rows = [
            {
                "phase": "attack",
                "step": 0,
                "c0": -radius,
                "c1": radius,
                "probe_loss": 9.0,
            },
            {"phase": "attack", "step": 0, "c0": 0, "c1": 0, "probe_loss": 10.0},
            {"phase": "attack", "step": 1000, "c0": 0, "c1": 0, "probe_loss": 4.0},
        ]
        frame = COMPARE.surface_grid(rows, 0, radius, 3)
        np.testing.assert_allclose(frame["coordinates"], [-radius, 0, radius])
        self.assertEqual(frame["measured"], 2)
        self.assertEqual(frame["expected"], 9)
        self.assertEqual(frame["values"][2, 0], 9.0)
        self.assertEqual(frame["values"][1, 1], 10.0)
        self.assertTrue(np.isnan(frame["values"][0, 0]))
        self.assertEqual(frame["issues"], [])
        rows.append({"phase": "attack", "step": 0, "c0": 1.0, "c1": 0, "probe_loss": 3})
        invalid = COMPARE.surface_grid(rows, 0, radius, 3)
        self.assertIn("physical lattice", invalid["issues"][0])

    def test_family_color_limits_do_not_hide_height_or_shape_differences(self):
        low, high = COMPARE.color_limits([3, 4, 12, 15])
        self.assertLess(low, 3)
        self.assertGreater(high, 15)
        self.assertEqual(
            COMPARE.color_limits([-0.1, 0.2, -3, 1], relative=True), (-3, 3)
        )
        low, high = COMPARE.color_limits([0, 0], relative=True)
        self.assertGreater(high, low)
        self.assertEqual(low, -high)


class AggregateFixtureTests(unittest.TestCase):
    def test_partial_endpoints_and_ambiguous_source_rows_remain_honest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run / "cases/noise-1e-4"
            analysis = folder
            case = {
                "role": "attack",
                "max_steps": 1000,
                "save_steps": [128, 1000],
                "bootstrap_seed": None,
            }
            write_json(
                folder / "source-dataset.json",
                {
                    "population_size": 3,
                    "draw_size": 3,
                    "unique_source_rows": 2,
                    "batch_sources": {"same": [0, 1], "different": [2]},
                    "drawn_source_rows": [0, 0, 1],
                },
            )
            write_json(
                folder / "runtime.json",
                {"wall_seconds": 3600, "allocated_gpu_seconds": 14400},
            )
            write_json(
                folder / "models/attack/noise.json",
                {"actual_l2_after_cast": 0.25, "changed_fraction": 0.01},
            )
            write_json(
                analysis / "summary.json",
                {
                    "attack_plane_energy_fraction": 0.8,
                    "planes": {
                        "reference": {
                            "minimum_capture": 0.2,
                            "maximum_projection_loss_gap": 3.5,
                        }
                    },
                },
            )
            write_json(
                analysis / "geometry/geometry.json",
                {
                    "steps": [128],
                    "attack_rank": 2,
                    "planes": {"attack": [{"step": 128, "distance": 0.4}]},
                },
            )
            write_csv(
                analysis / "surfaces/actual-losses.csv",
                [
                    {
                        "phase": "attack",
                        "step": 0,
                        "probe_loss": 11,
                        "heldout_loss": 12,
                    },
                    {
                        "phase": "attack",
                        "step": 128,
                        "probe_loss": 4,
                        "heldout_loss": 5,
                    },
                ],
            )
            write_csv(
                folder / "evaluation-scores.csv",
                [
                    {"attack_step": 0, "task": "mmlu_no_bio", "jwk_percent": 54},
                    {"attack_step": 0, "task": "wmdp_bio_robust", "jwk_percent": 33},
                    {"attack_step": 128, "task": "wmdp_bio_robust", "jwk_percent": 45},
                ],
            )
            trace = folder / "models/attack/observed-batches.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "step": step,
                            "batch_sha256": "same",
                            "source_rows": [0, 1],
                            "forward_backward_seconds": 2,
                        }
                    )
                    for step in (1, 2)
                )
            )
            record = COMPARE.collect_case(run, "noise-1e-4", case)
            metrics = record["metrics"]
            self.assertEqual(metrics["fixed_ce_start"], 11)
            self.assertIsNone(metrics["fixed_ce_final"])
            self.assertIsNone(metrics["heldout_ce_final"])
            self.assertIsNone(metrics["actual_displacement"])
            self.assertEqual(metrics["latest_actual_step"], 128)
            self.assertEqual(metrics["latest_geometry_step"], 128)
            self.assertEqual(metrics["peak_measured_wmdp"], 45)
            self.assertEqual(metrics["peak_measured_wmdp_step"], 128)
            self.assertEqual(metrics["initial_mmlu"], 54)
            self.assertTrue(metrics["initial_within_5pp_base"])
            self.assertFalse(metrics["evaluation_complete"])
            self.assertEqual(metrics["observed_distinct_content_hashes"], 1)
            self.assertEqual(metrics["prepared_unique_source_rows"], 2)
            self.assertEqual(metrics["recorded_forward_backward_hours"], 4 / 3600)
            self.assertEqual(metrics["wall_hours"], 1)
            self.assertEqual(metrics["allocated_gpu_hours"], 4)
            self.assertEqual(metrics["maximum_functional_projection_gap"], 3.5)
            self.assertEqual(metrics["noise_actual_l2"], 0.25)

    def test_partial_trace_is_reported_without_inventing_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"step":1,"batch_sha256":"a"}\n{"step":')
            result = COMPARE.observations(path, 2)
            self.assertEqual(result["observed_steps"], 1)
            self.assertFalse(result["observations_complete"])
            self.assertIsNone(result["forward_backward_seconds"])
            self.assertIn("Incomplete observation JSON", result["issues"][0])


class ComparabilityTests(unittest.TestCase):
    def fixture(self, root):
        baseline, analysis = root / "baseline", root / "case/analysis"
        (baseline / "geometry").mkdir(parents=True)
        (analysis / "geometry").mkdir(parents=True)
        for plane, filename in (
            ("random", "random-basis.npy"),
            ("reference", "attack-basis.npy"),
        ):
            source = baseline / "geometry" / filename
            source.write_bytes(b"directions are never loaded by the report")
            (analysis / "geometry" / f"{plane}-basis.npy").symlink_to(source)
        probes = {
            "fixed_source_indices": list(range(10)),
            "heldout_source_indices": list(range(100, 132)),
            "heldout_sha256": [str(index) for index in range(32)],
            "forget_revision": "frozen-revision",
            "expected_nonbootstrap_order_sha256": "clean-order",
        }
        record = {
            "name": "clean",
            "analysis": analysis,
            "case": {"bootstrap_seed": None},
            "trace": {"issues": []},
            "metrics": {
                "observations_complete": True,
                "observed_order_sha256": "clean-order",
            },
            "settings": {
                "shared_random_basis": str(baseline / "geometry/random-basis.npy"),
                "reference_bases": {
                    "reference": str(baseline / "geometry/attack-basis.npy")
                },
                "forget_revision": "frozen-revision",
                "local_plane": "reference",
                "local_radius": COMPARE.LOCAL_RADIUS,
                "plane_radii": {"random": 1, "reference": 1},
                "planes": ["random", "reference", "attack"],
                "broad_radius": 125,
            },
            "probes": {
                **copy.deepcopy(probes),
                "fixed_documents": 10,
                "heldout_excludes_observed_attack_documents": True,
            },
        }
        return record, {"baseline_dynamics": str(baseline)}, probes

    def test_matching_metadata_and_physical_sources_enable_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = self.fixture(Path(directory))
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertTrue(result["local_comparable"])
            self.assertTrue(result["broad_comparable"])
            self.assertTrue(result["shared_probes_verified"])
            self.assertEqual(result["issues"], [])
            record["probes"]["heldout_sha256"][0] = "different"
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertFalse(result["broad_comparable"])
            self.assertFalse(result["shared_probes_verified"])

    def test_scale_and_data_order_failures_are_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = self.fixture(Path(directory))
            record["settings"]["local_radius"] *= 2
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertTrue(result["broad_comparable"])
            record["settings"]["local_radius"] = COMPARE.LOCAL_RADIUS
            record["settings"]["broad_radius"] = 99
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertTrue(result["local_comparable"])
            self.assertFalse(result["broad_comparable"])
            record["metrics"]["observed_order_sha256"] = "different"
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertIn("different", record["metrics"]["observed_order_sha256"])
            self.assertTrue(
                any("data order differs" in issue for issue in result["issues"])
            )


if __name__ == "__main__":
    unittest.main()
