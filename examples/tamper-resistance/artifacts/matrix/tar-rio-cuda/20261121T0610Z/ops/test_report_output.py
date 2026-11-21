"""Distinguish an observed utility failure from an incomplete measurement."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "cuda_report", Path(__file__).with_name("finish_report.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def records(mmlu):
    scores = [
        {"attack_step": 0, "task": "wmdp_bio_robust", "jwk_percent": 30.0},
        {"attack_step": 1000, "task": "wmdp_bio_robust", "jwk_percent": 48.0},
    ]
    if mmlu is not None:
        scores.append({"attack_step": 0, "task": "mmlu_no_bio", "jwk_percent": mmlu})
    return {16: {"scores": scores}}


class ReportOutputTests(unittest.TestCase):
    def test_recovery_note_renders_literal_percentages(self):
        text = report.recovery_note(55.0691244, 58.5797928)
        for value in ("45%", "50%", "90%", "55.0691%", "53.5798%"):
            self.assertIn(value, text)
        self.assertIn("same-backend-k-summary.csv", text)

    def test_measured_below_gate_is_failure_without_missing_label(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(report, "ROOT", Path(directory)),
        ):
            summary = report.write_score_tables(records(52.0), 58.5797928, 55.0691244)
            self.assertEqual(summary[0]["utility gate"], "fail")
            self.assertNotIn(
                "missing", (Path(directory) / "same-backend-k-summary.csv").read_text()
            )

    def test_measured_above_gate_passes(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(report, "ROOT", Path(directory)),
        ):
            summary = report.write_score_tables(records(54.0), 58.5797928, 55.0691244)
            self.assertEqual(summary[0]["utility gate"], "pass")

    def test_missing_measurement_stops_reporting(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(report, "ROOT", Path(directory)),
        ):
            with self.assertRaisesRegex(ValueError, "Missing checkpoint-zero MMLU"):
                report.write_score_tables(records(None), 58.5797928, 55.0691244)
            self.assertFalse((Path(directory) / "same-backend-k-summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
