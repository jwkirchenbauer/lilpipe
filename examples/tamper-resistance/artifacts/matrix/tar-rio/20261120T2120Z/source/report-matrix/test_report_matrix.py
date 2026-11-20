import importlib.util
import tempfile
import unittest
from pathlib import Path


WORKER = Path(__file__).with_name("finish_report.py")
spec = importlib.util.spec_from_file_location("finish_report", WORKER)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def valid_parity_record():
    return {
        "state": "complete",
        "actual_updates": 16,
        "actual_horizon": 1000,
        "actual_loss_errors": [0.0] * 16,
        "trajectory_parity": "exact",
        "parameter_checks": [
            {
                "step": step,
                "simulated_vs_actual": {
                    "exact_equal": True,
                    "dtype_mismatches": {},
                    "difference_l2": 0.0,
                    "maximum_absolute_difference": 0.0,
                },
                "archived_vs_actual": {
                    "relative_update_error": 0.125,
                    "exact_equal": False,
                },
            }
            for step in (1, 2, 4, 8, 16)
        ],
    }


class ReportMatrixTests(unittest.TestCase):
    def test_accepts_publisher_dtype_map_and_archived_drift_diagnostic(self):
        archived = report.validate_parity_record(valid_parity_record())
        self.assertEqual(archived, [0.125] * 5)

    def test_rejects_nonempty_dtype_mismatch_map(self):
        record = valid_parity_record()
        record["parameter_checks"][0]["simulated_vs_actual"]["dtype_mismatches"] = {
            "layer.weight": ["torch.float16", "torch.bfloat16"]
        }
        with self.assertRaisesRegex(ValueError, "not exact"):
            report.validate_parity_record(record)

    def test_cuda_thresholds_follow_fresh_base(self):
        self.assertEqual(
            report.trajectory_thresholds(55.06912442396313),
            (45.0, 50.0, 49.56221198156682),
        )
        self.assertEqual(report.trajectory_thresholds(55.0, (43.0,)), (43.0,))
        scores = [
            {"task": "wmdp_bio_robust", "attack_step": 8, "jwk_percent": 49.7},
            {"task": "wmdp_bio_robust", "attack_step": 16, "jwk_percent": 49.5},
        ]
        self.assertEqual(
            report.first_wmdp_crossing(scores, 49.56221198156682, 1000), 16.0
        )

    def test_environment_prefix_accepts_symlink_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            physical = root / "physical"
            physical.mkdir()
            alias = root / "alias"
            alias.symlink_to(physical, target_is_directory=True)
            self.assertTrue(report.same_path(alias, physical))


if __name__ == "__main__":
    unittest.main()
