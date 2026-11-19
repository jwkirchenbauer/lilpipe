"""Coordinate checks for the real-model simulated/actual attack parity pilot."""

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")
EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
sys.path[:0] = [str(EXAMPLE / "scripts/misc"), str(EXAMPLE / "src")]
from tar_inner_recording import parity_parameter_statistics


def test_parity_error_normalizes_by_attack_update_not_total_weight_norm():
    current = {"weight": torch.tensor([1002.0, 1004.0])}
    reference = {"weight": torch.tensor([1001.0, 1005.0])}
    anchor = {"weight": torch.tensor([1000.0, 1001.0])}
    measured = parity_parameter_statistics(current, reference.__getitem__, anchor, 1)
    assert measured["coordinates"] == 2
    assert measured["difference_l2"] == math.sqrt(2)
    assert measured["expected_update_l2"] == math.sqrt(17)
    assert measured["relative_update_error"] == math.sqrt(2 / 17)
    assert measured["maximum_absolute_difference"] == 1
    assert not measured["exact_equal"]


def test_parity_distinguishes_exact_values_from_dtype_and_zero_update_cases():
    reference = {"weight": torch.tensor([1.0, 2.0])}
    anchor = {"weight": torch.zeros(2)}
    exact = parity_parameter_statistics(reference, reference.__getitem__, anchor)
    assert exact["exact_equal"]
    assert exact["relative_update_error"] == 0
    mismatched = {"weight": reference["weight"].double()}
    casting = parity_parameter_statistics(mismatched, reference.__getitem__, anchor)
    assert casting["difference_l2"] == 0
    assert not casting["exact_equal"]
    assert casting["dtype_mismatches"] == {"weight": ["torch.float64", "torch.float32"]}
    no_reference_update = parity_parameter_statistics(
        {"weight": torch.tensor([1.0, 3.0])}, reference.__getitem__, reference
    )
    assert no_reference_update["relative_update_error"] is None
