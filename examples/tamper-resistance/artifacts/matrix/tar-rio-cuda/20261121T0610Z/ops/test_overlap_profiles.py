"""Protect shared-reference ordering when releasing profile allocations."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "overlap_profiles", Path(__file__).with_name("overlap_profiles.py")
)
overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlap)
REFERENCE = "rio-k16-cuda-lr1e-6"
OTHER = "rio-k64-cuda-lr1e-6"


class ProfileReferenceTests(unittest.TestCase):
    def fixture(self, root):
        (root / "study.json").write_text(
            json.dumps({"profile_reference_case": REFERENCE})
        )

    def test_reference_can_measure_its_own_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            overlap.require_profile_reference(root, REFERENCE)

    def test_pending_reference_blocks_cli_before_scheduler_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with (
                patch.object(overlap, "ROOT", root),
                patch("sys.argv", ["overlap_profiles.py", OTHER, "--apply"]),
                patch.object(overlap.subprocess, "check_output") as inspect,
                patch.object(overlap.subprocess, "run") as mutate,
            ):
                with self.assertRaisesRegex(ValueError, "must finish"):
                    overlap.main()
                inspect.assert_not_called()
                mutate.assert_not_called()

    def test_completed_reference_allows_independent_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            status = root / "cases" / REFERENCE / "status/step-profiles.json"
            status.parent.mkdir(parents=True)
            status.write_text(json.dumps({"state": "complete"}))
            overlap.require_profile_reference(root, OTHER)


if __name__ == "__main__":
    unittest.main()
