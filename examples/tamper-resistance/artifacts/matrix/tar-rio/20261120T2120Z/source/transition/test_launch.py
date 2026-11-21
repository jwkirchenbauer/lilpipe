"""Reject incomplete case preparation before requesting an allocation."""

import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "matrix_launch", Path(__file__).with_name("launch.py")
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


class CasePreparationTests(unittest.TestCase):
    def test_missing_status_directory_fails_before_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "cases/control"
            for name in ("logs", "cache", "models"):
                (folder / name).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "status"):
                launch.require_case_directories(root, "control")
            (folder / "status").mkdir()
            launch.require_case_directories(root, "control")


class RetrySubmissionTests(unittest.TestCase):
    def test_retry_attempt_has_unique_ledger_and_names_without_changing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "cases" / "rio-k64"
            for name in ("status", "logs", "cache", "models"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            (root / "logs").mkdir(exist_ok=True)
            ledger_path = root / "matrix-jobs.json"
            ledger_path.write_text('{"profiles:rio-k64":{"job_id":"510001"}}\n')
            launch_result = mock.Mock(stdout="Submitted with job_id=510099.\n")

            with (
                mock.patch.object(launch, "ROOT", root),
                mock.patch.object(
                    launch.subprocess, "run", return_value=launch_result
                ) as run_mock,
                mock.patch(
                    "sys.argv",
                    [
                        "launch.py",
                        "profiles",
                        "rio-k64",
                        "--run",
                        str(root),
                        "--attempt",
                        "retry-v2",
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch.main()

            ledger = json.loads(ledger_path.read_text())
            self.assertEqual(ledger["profiles:rio-k64"]["job_id"], "510001")
            self.assertEqual(ledger["profiles:rio-k64:retry-v2"]["job_id"], "510099")
            command = run_mock.call_args.args[0]
            self.assertIn("--run_name=tar-rio-profiles-rio-k64-retry-v2", command)
            self.assertIn("--sub_output_dir_name=profiles-rio-k64-retry-v2", command)
            self.assertEqual(
                command[-6:],
                [
                    "bash",
                    "-i",
                    str(root / "source/transition/entry.sh"),
                    "profiles",
                    str(root),
                    "rio-k64",
                ],
            )

    def test_retry_dryrun_switch_precedes_target_delimiter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "cases" / "rio-k64"
            for name in ("status", "logs", "cache", "models"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            (root / "logs").mkdir(exist_ok=True)
            ledger_path = root / "matrix-jobs.json"
            ledger_path.write_text('{"profiles:rio-k64":{"job_id":"510001"}}\n')
            original = ledger_path.read_bytes()
            launch_result = mock.Mock(stdout="dryrun\n")

            with (
                mock.patch.object(launch, "ROOT", root),
                mock.patch.object(
                    launch.subprocess, "run", return_value=launch_result
                ) as run_mock,
                mock.patch(
                    "sys.argv",
                    [
                        "launch.py",
                        "profiles",
                        "rio-k64",
                        "--run",
                        str(root),
                        "--attempt",
                        "retry-v2",
                        "--dryrun",
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch.main()

            command = run_mock.call_args.args[0]
            self.assertLess(command.index("--dryrun"), command.index("--"))
            self.assertEqual(ledger_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
