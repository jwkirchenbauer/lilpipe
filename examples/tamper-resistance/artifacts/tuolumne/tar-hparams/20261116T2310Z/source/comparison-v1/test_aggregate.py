"""Small CPU-only tests for aggregate landscape validation and plotting scales."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).with_name("aggregate.py")
SPEC = importlib.util.spec_from_file_location("aggregate_under_test", SOURCE)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"condition": name} for name in ("B", "R")]
        coordinates = np.array([-1.0, 0.0, 1.0])
        self.specs = {
            (condition, frame): {
                "coordinates": coordinates.copy(),
                "values": np.full((3, 3), value, dtype=float),
            }
            for condition, frame, value in (
                ("B", 0, 2.0),
                ("R", 0, 5.0),
                ("B", 1, -1.0),
                ("R", 1, 3.0),
            )
        }

    def test_global_limits_and_shared_axes_cover_all_conditions_and_frames(self):
        frames, (low, high) = aggregate.matrix_limits(self.records, self.specs)
        self.assertEqual(frames, [0, 1])
        self.assertLess(low, -1.0)
        self.assertGreater(high, 5.0)
        self.assertAlmostEqual(high - low, 6.48)

    def test_missing_condition_frame_is_rejected(self):
        incomplete = dict(self.specs)
        del incomplete[("R", 1)]
        with self.assertRaisesRegex(ValueError, "missing or extra"):
            aggregate.matrix_limits(self.records, incomplete)

    def test_mismatched_physical_axes_are_rejected(self):
        mismatched = {key: dict(value) for key, value in self.specs.items()}
        mismatched[("R", 0)]["coordinates"] = np.array([-2.0, 0.0, 2.0])
        with self.assertRaisesRegex(ValueError, "axes differ"):
            aggregate.matrix_limits(self.records, mismatched)

    def test_nonfinite_or_wrong_shaped_grid_is_rejected(self):
        invalid = {key: dict(value) for key, value in self.specs.items()}
        invalid[("B", 0)]["values"] = np.full((2, 2), np.nan)
        with self.assertRaisesRegex(ValueError, "invalid values"):
            aggregate.matrix_limits(self.records, invalid)

    def test_missing_grid_point_is_rejected(self):
        rows = [
            {"phase": "attack", "step": "0", "c0": str(x), "c1": str(y), "probe_loss": "1.5"}
            for x in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            if (x, y) != (1.0, 1.0)
        ]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            aggregate.grid_from_rows(rows, 1.0, 3, phase="attack", step=0)

    def test_wrong_physical_grid_coordinate_is_rejected(self):
        rows = [
            {"phase": "attack", "step": "0", "c0": "0.2", "c1": "0", "probe_loss": "1.5"}
        ]
        with self.assertRaisesRegex(ValueError, "physical lattice"):
            aggregate.grid_from_rows(rows, 1.0, 3, phase="attack", step=0)

    def test_defense_training_guard_does_not_require_benchmark_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status").mkdir()
            (folder / "status/train.json").write_text('{"state":"complete"}\n')
            (folder / "training-validation.json").write_text(
                json.dumps({
                    "state": "passed",
                    "updates": 2,
                    "input_order_validation": {
                        "state": "passed",
                        "batches": 2,
                        "canonical_step_batch_sha256": "a" * 64,
                    },
                })
            )
            result = aggregate.validate_training_guard(folder, {"max_steps": 2})
            self.assertEqual(result["state"], "passed")
            self.assertFalse((folder / "status/evaluation.json").exists())

    def test_failed_input_guard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "status").mkdir()
            (folder / "status/train.json").write_text('{"state":"complete"}\n')
            (folder / "training-validation.json").write_text(
                json.dumps({
                    "state": "passed",
                    "updates": 2,
                    "input_order_validation": {"state": "failed", "batches": 1},
                })
            )
            with self.assertRaisesRegex(ValueError, "input-order guard"):
                aggregate.validate_training_guard(folder, {"max_steps": 2})


if __name__ == "__main__":
    unittest.main()
