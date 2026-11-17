"""Small CPU fixtures only; no model or scientific rendering."""

import csv
import tempfile
import unittest
from pathlib import Path

from static_report import common_limits
from static_worker import RADII, check_protocol, validate_grid


class StaticTests(unittest.TestCase):
    def grid(self, path):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("radius", "c0", "c1", "loss"))
            writer.writeheader()
            for radius in RADII:
                for first in range(9):
                    for second in range(9):
                        x, y = (
                            -radius + first * radius / 4,
                            -radius + second * radius / 4,
                        )
                        writer.writerow(
                            {
                                "radius": radius,
                                "c0": x,
                                "c1": y,
                                "loss": 12 + (x * x + y * y) * 1e-6,
                            }
                        )

    def test_complete_grid_and_measured_centers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.csv"
            self.grid(path)
            self.assertEqual(len(validate_grid(path)), 648)

    def test_partial_duplicate_or_changed_center_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.csv"
            self.grid(path)
            rows = path.read_text().splitlines()
            for replacement in (
                rows[:-1],
                [*rows[:-1], rows[-2]],
                [
                    (
                        row.replace("0.0,0.0,12.0", "0.0,0.0,12.5", 1)
                        if index == 41
                        else row
                    )
                    for index, row in enumerate(rows)
                ],
            ):
                path.write_text("\n".join(replacement) + "\n")
                with self.assertRaises(AssertionError):
                    validate_grid(path)

    def test_protocol_reuse_and_stale_input_rejection(self):
        protocol = {
            "files": [{"path": "/frozen/a", "size": 10, "mtime_ns": 20}],
            "source": "abc",
            "seed": 42,
        }
        check_protocol(protocol, dict(protocol))
        for changed in (
            {**protocol, "seed": 43},
            {**protocol, "source": "def"},
            {**protocol, "files": [{"path": "/frozen/a", "size": 10, "mtime_ns": 21}]},
        ):
            with self.assertRaisesRegex(ValueError, "changed"):
                check_protocol(protocol, changed)

    def test_shared_absolute_and_symmetric_delta_limits(self):
        low, high = common_limits([10, 11, 100, 103])
        self.assertLess(low, 10)
        self.assertGreater(high, 103)
        self.assertEqual(common_limits([-2, 1], symmetric=True), (-2.1, 2.1))
        self.assertEqual(common_limits([0, 0], symmetric=True), (-1e-5, 1e-5))


if __name__ == "__main__":
    unittest.main()
