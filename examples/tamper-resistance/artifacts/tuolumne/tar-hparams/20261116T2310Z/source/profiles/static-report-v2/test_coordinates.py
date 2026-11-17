"""Regression for actual original CSV decimal strings and real mismatches."""

import csv
import io
import math
import struct
import unittest

from coordinates import canonicalize, fp32


def row(text):
    result = next(csv.DictReader(io.StringIO("radius,c0,c1,loss\n" + text)))
    return {key: float(value) for key, value in result.items()}


class CoordinateTests(unittest.TestCase):
    def test_actual_original_and_new_csv_decimals_recover_same_fp32(self):
        original = row(
            "0.1,-0.1000000014901161,-0.0750000029802322,11.238433265686036\n"
        )
        current = row(
            "0.1,-0.10000000149011612,-0.07500000298023224,10.95722942352295\n"
        )
        self.assertNotEqual(original["c0"], current["c0"])
        values, proof = canonicalize([current], [original])
        self.assertTrue(proof["fp32_execution_coefficients_bitwise_match_reference"])
        self.assertGreater(
            proof["maximum_serialized_coordinate_difference_from_reference"], 0
        )
        self.assertLess(
            proof["maximum_serialized_coordinate_difference_from_reference"], 1e-15
        )
        self.assertEqual(values[0]["loss"], current["loss"])
        self.assertEqual(values[0]["c0"], fp32(original["c0"]))

    def test_one_fp32_ulp_coordinate_change_is_rejected(self):
        original = row("0.1,-0.1000000014901161,0,11\n")
        packed = struct.unpack("!I", struct.pack("!f", original["c0"]))[0]
        shifted = struct.unpack("!f", struct.pack("!I", packed + 1))[0]
        with self.assertRaisesRegex(ValueError, "bitwise"):
            canonicalize([{**original, "c0": shifted}], [original])

    def test_missing_keys_and_fp32_collisions_are_rejected(self):
        original = row("0.1,-0.1000000014901161,0,11\n")
        current = row("0.1,-0.10000000149011612,0,11\n")
        with self.assertRaisesRegex(ValueError, "collide"):
            canonicalize([original, current])
        with self.assertRaisesRegex(ValueError, "bitwise"):
            canonicalize([], [original])

    def test_nonfinite_coordinates_or_losses_are_rejected(self):
        original = row("0.1,0,0,11\n")
        for field in ("c0", "c1", "loss"):
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                canonicalize([{**original, field: math.nan}])


if __name__ == "__main__":
    unittest.main()
