import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = (
    Path(__file__).resolve().parents[1] / "examples/tamper-resistance/scripts/misc"
)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    import tar_inner_geometry as geometry


def blocks(inner, actual):
    return {
        "inner_gram": (inner @ inner.T).tolist(),
        "cross_gram": (inner @ actual.T).tolist(),
        "reference_gram": (actual @ actual.T).tolist(),
    }


def example():
    # Actual state 2 is projection-only. Its vector is orthogonal to the full inner span.
    inner = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    actual = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    states = [{"name": f"ref{i}", "step": i} for i in (1, 2, 4, 17)]
    views = {
        "matched": [states[0], states[2]],
        "long": [states[0], states[2], states[3]],
    }
    record = {
        "schema_version": 1,
        "status": "complete",
        "outer_call": 1,
        "coordinates": 3,
        "inner_steps": [1, 4],
        "layout": [{"name": "toy", "shape": [3], "dtype": "float32"}],
        "references": states,
        "reference_names": [s["name"] for s in states],
        "representations": {
            "raw": blocks(inner, actual),
            "bf16": blocks(inner * 0, actual),
        },
    }
    return record, views, states


class TestInnerGeometry(unittest.TestCase):
    def test_intervals_and_full_rank_normalization(self):
        record, views, states = example()
        rows = geometry.analyze_call(record, views, states)

        def select(table, **filters):
            return [
                r
                for r in rows[table]
                if r.get("representation") == "raw_saved"
                and r.get("normalization", "norm_weighted") == "norm_weighted"
                and all(r[k] == v for k, v in filters.items())
            ]

        (r,) = select(
            "comparisons",
            motion="displacement",
            actual_view="long",
            requested_rank="full",
        )
        self.assertEqual((r["inner_rank"], r["actual_rank"]), (2, 3))
        self.assertAlmostEqual(r["inner_overlap_fraction"], 1)
        self.assertAlmostEqual(r["actual_overlap_fraction"], 2 / 3)
        self.assertAlmostEqual(r["actual_energy_captured"], 6 / 15)
        self.assertAlmostEqual(r["mean_actual_vector_capture"], 2 / 3)
        self.assertAlmostEqual(r["first_step_cosine"], -1)
        all_rows = select(
            "capture",
            motion="displacement",
            actual_view="all_states",
            requested_rank="full",
        )
        self.assertEqual([r["end_step"] for r in all_rows], [1, 2, 4, 17])
        np.testing.assert_allclose(
            [r["capture_fraction"] for r in all_rows], [1, 0, 1, 0], atol=1e-12
        )
        (r,) = select(
            "capture",
            motion="interval_update",
            actual_view="all_states",
            requested_rank="full",
            end_step=4,
        )
        self.assertEqual(r["start_step"], 2)
        self.assertAlmostEqual(r["squared_norm"], 6)
        self.assertAlmostEqual(r["capture_fraction"], 5 / 6)
        (r,) = select(
            "capture",
            motion="interval_update",
            actual_view="matched",
            requested_rank="full",
            end_step=4,
        )
        self.assertEqual(r["start_step"], 1)
        self.assertAlmostEqual(r["squared_norm"], 8)
        self.assertAlmostEqual(r["capture_fraction"], 1)
        (r,) = select(
            "inner_similarity", motion="interval_update", left_end=1, right_end=4
        )
        self.assertAlmostEqual(r["cosine"], 0)
        self.assertEqual(
            {r["view"] for r in rows["spectrum"]}, {"inner", "matched", "long"}
        )
        zero = [
            r
            for r in rows["comparisons"]
            if r["representation"] == "bf16_evaluated" and r["requested_rank"] == "full"
        ]
        self.assertTrue(
            all(
                r["inner_rank"] == 0
                and r["actual_energy_captured"] == 0
                and r["inner_overlap_fraction"] is None
                and r["first_step_cosine"] is None
                for r in zero
            )
        )

    def test_projection_matches_dense_svd_at_both_normalizations(self):
        rng = np.random.default_rng(205)
        for equal_unit in (False, True):
            x = rng.normal(size=(4, 7)) * np.array([1, 2, 5, 3])[:, None]
            y = rng.normal(size=(5, 7)) * np.array([1, 1, 3, 1, 5])[:, None]
            a, _ = geometry.fit(x @ x.T, equal_unit)
            b, _ = geometry.fit(y @ y.T, equal_unit)
            vals, angles, energies, fractions = geometry.capture_and_overlap(
                a, b, x @ y.T, y @ y.T, "2"
            )
            x_fit = x / np.linalg.norm(x, axis=1)[:, None] if equal_unit else x
            y_fit = y / np.linalg.norm(y, axis=1)[:, None] if equal_unit else y
            ux = np.linalg.svd(x_fit, full_matrices=False)[2][:2]
            uy = np.linalg.svd(y_fit, full_matrices=False)[2][:2]
            expected = np.square(y @ ux.T).sum(axis=1)
            np.testing.assert_allclose(fractions, expected / np.square(y).sum(axis=1))
            np.testing.assert_allclose(
                angles,
                np.rad2deg(np.arccos(np.linalg.svd(ux @ uy.T, compute_uv=False))),
            )
            self.assertAlmostEqual(
                vals["actual_energy_captured"], expected.sum() / energies.sum()
            )

    def test_spec_and_family_validation(self):
        record, views, states = example()
        record["representations"]["raw"]["anchor_gap_squared_norm"] = 9.0
        record["families"] = {
            "toy": {"coordinates": 3, "representations": record["representations"]}
        }
        tables = geometry.analyze_call(record, views, states)
        self.assertEqual(tables["anchor_distance"][0]["distance"], 3)
        self.assertEqual(len(tables["family_comparisons"]), len(tables["comparisons"]))
        self.assertAlmostEqual(
            tables["family_comparisons"][0]["actual_energy_captured"],
            tables["comparisons"][0]["actual_energy_captured"],
        )
        record["families"]["toy"]["coordinates"] = 2
        with self.assertRaisesRegex(ValueError, "counts"):
            geometry.analyze_call(record, views, states)
        with self.assertRaisesRegex(ValueError, "references"):
            geometry.validate_record(record, views, states[1:])
        record["representations"]["raw"]["cross_gram"][0][0] = 1000
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            geometry.combined_gram(record, "raw_saved")

    def test_report_resumes_from_atomic_per_call_results(self):
        record, views, states = example()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input.json"
            path.write_text(json.dumps(record))
            spec = {
                "version": 1,
                "aggregates": [str(path)],
                "actual_views": views,
                "actual_states": states,
            }
            output = root / "output"
            geometry.run("analyze", spec, output)
            self.assertEqual(
                json.loads((output / "status/geometry.json").read_text())["state"],
                "analysis_complete",
            )
            with patch.object(
                geometry,
                "analyze_call",
                side_effect=AssertionError("repeated analysis"),
            ):
                geometry.run("report", spec, output)
                status = json.loads((output / "status/geometry.json").read_text())
                self.assertEqual(status["state"], "complete")
                self.assertTrue(all((output / p).is_file() for p in status["files"]))
                with (output / "capture.csv").open() as stream:
                    self.assertIn(
                        "all_states", {r["actual_view"] for r in csv.DictReader(stream)}
                    )
                with patch.object(
                    geometry, "figures", side_effect=AssertionError("repeated figures")
                ):
                    geometry.run("report", spec, output)
            record["outer_call"] = 2
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "different inputs"):
                geometry.run("report", spec, output)


if __name__ == "__main__":
    unittest.main()
