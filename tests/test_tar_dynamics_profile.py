"""CPU checks for the memory-bounded early-update profile arithmetic."""

import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_dynamics.py"
)
SPEC = importlib.util.spec_from_file_location("tar_dynamics_profile_under_test", SCRIPT)
tar_dynamics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tar_dynamics)


class ProfileArithmeticTests(unittest.TestCase):
    def test_cpu_anchor_interpolation_preserves_raw_and_evaluated_endpoints(self):
        anchor = torch.tensor([1.0, -2.0, 3.0], dtype=torch.bfloat16)
        after = torch.tensor([1.125, -1.75, 2.5], dtype=torch.bfloat16)
        delta = after.float() - anchor.float()

        start = tar_dynamics._profile_interpolate(
            anchor, delta, 0, dtype=torch.bfloat16
        )
        end = tar_dynamics._profile_interpolate(anchor, delta, 1, dtype=torch.bfloat16)
        middle = tar_dynamics._profile_interpolate(
            anchor, delta, 0.5, dtype=delta.dtype
        )

        self.assertEqual(start.device.type, "cpu")
        self.assertTrue(torch.equal(start, anchor))
        self.assertTrue(torch.equal(end, after))
        self.assertEqual(middle.dtype, torch.float32)
        self.assertTrue(torch.equal(middle, anchor.float() + delta * 0.5))

    def test_cpu_update_statistics_retain_chunked_fp64_norm_semantics(self):
        before = torch.tensor([0.0, 1.0, -2.0, 4.0], dtype=torch.bfloat16)
        after = torch.tensor([0.25, 1.0, -1.5, 3.0], dtype=torch.bfloat16)

        squared, changed = tar_dynamics._profile_update_statistics(
            before, after, chunk=2
        )
        expected = (after.double() - before.double()).square().sum().item()
        self.assertEqual(squared, expected)
        self.assertEqual(changed, 3)

    def test_fp32_saved_motion_can_vanish_after_bf16_evaluation_cast(self):
        before = torch.tensor([1.0, 16.0, -1.0])
        after = before + torch.tensor([1e-6, 1e-5, 0.0])
        raw_squared, raw_changed = tar_dynamics._profile_update_statistics(
            before, after
        )
        cast_squared, cast_changed = tar_dynamics._profile_update_statistics(
            before, after, dtype=torch.bfloat16
        )
        self.assertGreater(raw_squared, 0)
        self.assertEqual(raw_changed, 2)
        self.assertEqual((cast_squared, cast_changed), (0, 0))

    def test_cpu_first_update_plane_retains_orthogonal_completion(self):
        delta = {"weight": torch.tensor([1.0, 2.0, -1.0])}
        random = {"weight": torch.tensor([0.3, -0.2, 0.1])}
        proof = tar_dynamics._profile_first_plane(delta, random)
        self.assertEqual(proof["status"], "fitted")
        self.assertAlmostEqual(proof["q0_norm"], 1, places=6)
        self.assertAlmostEqual(proof["q1_norm"], 1, places=6)
        self.assertAlmostEqual(proof["q0_dot_q1"], 0, places=6)


if __name__ == "__main__":
    unittest.main()
