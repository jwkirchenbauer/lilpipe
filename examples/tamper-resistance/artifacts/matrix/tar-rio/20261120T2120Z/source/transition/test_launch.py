"""Reject incomplete case preparation before requesting an allocation."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
