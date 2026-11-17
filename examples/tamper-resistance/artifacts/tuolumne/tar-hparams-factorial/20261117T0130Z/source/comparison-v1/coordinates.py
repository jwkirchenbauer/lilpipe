"""Recover the original torch.linspace FP32 coefficients from CSV decimals."""

import math
import struct


def fp32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def key(row):
    return (row["radius"], struct.pack("!f", row["c0"]), struct.pack("!f", row["c1"]))


def canonicalize(rows, reference=None):
    """Require exact FP32 coefficient identity; losses remain untouched."""
    if not all(
        math.isfinite(row[field])
        for row in rows
        for field in ("radius", "c0", "c1", "loss")
    ):
        raise ValueError("Nonfinite grid data")
    keys = [key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Coordinate keys collide after FP32 recovery")
    maximum_difference = 0.0
    if reference is not None:
        prior = {key(row): row for row in reference}
        if set(keys) != set(prior) or len(prior) != len(reference):
            raise ValueError("Grid FP32 execution coefficients differ bitwise")
        maximum_difference = max(
            (
                abs(row[field] - prior[key(row)][field])
                for row in rows
                for field in ("c0", "c1")
            ),
            default=0.0,
        )
    result = [{**row, "c0": fp32(row["c0"]), "c1": fp32(row["c1"])} for row in rows]
    return result, {
        "method": "Recover original FP32 torch.linspace coordinates; compare radius and FP32 coefficient bits exactly. No interpolation; no loss values changed.",
        "unique_fp32_keys": len(keys),
        "fp32_execution_coefficients_bitwise_match_reference": reference is not None,
        "maximum_serialized_coordinate_difference_from_reference": maximum_difference,
        "maximum_coordinate_decimal_to_fp32_difference": max(
            (
                abs(row[field] - transformed[field])
                for row, transformed in zip(rows, result, strict=True)
                for field in ("c0", "c1")
            ),
            default=0.0,
        ),
    }
