"""CPU-only checks for actual-model gradient diagnostic orchestration."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("requires torch") from error

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_cleanroom_llm_gradients.py"
)
SPEC = importlib.util.spec_from_file_location("tar_cleanroom_llm_gradients", SCRIPT)
LLM = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LLM
SPEC.loader.exec_module(LLM)
_, GRADIENTS = LLM.load_helpers()


class LLMGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_streamed_comparison_matches_known_vectors(self):
        result = LLM.gradient_statistics(
            {"x": torch.tensor([3.0, 4.0])},
            {"x": torch.tensor([3.0, -4.0])},
            chunk_elements=1,
        )
        self.assertEqual(result["exact_gradient_norm"], 5)
        self.assertEqual(result["first_order_gradient_norm"], 5)
        self.assertEqual(result["gradient_dot_product"], -7)
        self.assertAlmostEqual(result["gradient_cosine"], -0.28)
        self.assertEqual(result["gradient_difference_l2"], 8)
        self.assertEqual(result["first_order_relative_l2_error"], 1.6)

    def test_chunked_shifts_are_global_normalized_and_preserve_initialization(self):
        params = {
            "a": torch.tensor([0.2, -0.5], dtype=torch.float64, requires_grad=True),
            "b": torch.tensor([0.7], dtype=torch.float64, requires_grad=True),
        }
        direction = {
            "a": torch.tensor([3.0, 4.0]).double(),
            "b": torch.tensor([12.0]).double(),
        }
        originals = {name: value.detach().clone() for name, value in params.items()}
        shifted, geometry = LLM.shifted_parameters(
            params, direction, 13, -0.03, chunk_elements=1
        )
        self.assertAlmostEqual(geometry["realized_l2_step"], 0.03, places=14)
        self.assertAlmostEqual(
            geometry["projection_along_unit_gradient"], -0.03, places=14
        )
        self.assertAlmostEqual(
            LLM.distance(shifted, params, chunk_elements=1), 0.03, places=14
        )
        self.assertEqual(geometry["changed_parameter_fraction"], 1)
        for name, value in params.items():
            torch.testing.assert_close(value, originals[name], rtol=0, atol=0)
            self.assertTrue(shifted[name].is_leaf)
            self.assertTrue(shifted[name].requires_grad)

    def test_gradient_aligned_differences_and_outer_step_profiles(self):
        model, batches, outer, loss_fn = GRADIENTS.tiny_problem()
        params = dict(model.named_parameters())
        original = {name: value.detach().clone() for name, value in params.items()}
        config = GRADIENTS.OptimizerConfig(name="sgd", lr=0.08, max_grad_norm=0.25)
        report = LLM.diagnostic(
            GRADIENTS,
            model,
            params,
            batches[:1],
            loss_fn,
            outer,
            loss_fn,
            config,
            epsilons=(1e-3, 1e-4),
        )
        self.assertGreater(report["gradient_difference_l2"], 1e-4)
        self.assertLess(report["gradient_cosine"], 0.9999999)
        self.assertAlmostEqual(
            report["exact_outer_loss"], report["first_order_outer_loss"], places=13
        )
        self.assertAlmostEqual(
            report["exact_baseline_replay_abs_difference"], 0, places=13
        )
        self.assertAlmostEqual(
            report["baseline_replay"]["attack_update_l2"], 0.08 * 0.25, delta=1e-7
        )
        for row in report["finite_differences"]:
            self.assertLess(row["exact_relative_error"], 2e-5)
            self.assertEqual(row["state"], "complete")
            profile = next(
                profile
                for profile in report["negative_gradient_step_profiles"]
                if profile["gradient"] == row["direction"]
                and profile["requested_step_global_l2"] == row["epsilon_global_l2"]
            )
            self.assertEqual(
                profile["outer_loss_after_step_and_fresh_attack"],
                row["minus"]["outer_loss"],
            )
            self.assertTrue(profile["reduces_objective"])
            self.assertTrue(profile["reuses_minus_side_finite_difference_evaluation"])
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            self.assertIsNone(value.grad)

    def test_adamw_path_uses_same_orchestration_with_fresh_state(self):
        model, batches, outer, loss_fn = GRADIENTS.tiny_problem()
        config = GRADIENTS.OptimizerConfig(
            name="adamw", lr=0.02, eps=0.03, max_grad_norm=0.25
        )
        report = LLM.diagnostic(
            GRADIENTS,
            model,
            dict(model.named_parameters()),
            batches[:1],
            loss_fn,
            outer,
            loss_fn,
            config,
            directions=("first-order",),
            epsilons=(1e-4,),
        )
        self.assertEqual(report["optimizer"]["name"], "adamw")
        self.assertAlmostEqual(
            report["exact_first_order_outer_loss_abs_difference"], 0, places=13
        )
        self.assertLess(report["finite_differences"][0]["exact_relative_error"], 2e-6)
        self.assertEqual(len(report["negative_gradient_step_profiles"]), 1)

    def test_lm_losses_use_shifted_valid_tokens_and_negative_signs(self):
        logits = torch.tensor([[[2.0, -1.0], [-0.5, 1.0], [0.1, 0.3]]])
        batch = {
            "input_ids": torch.tensor([[0, 1, 0]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[0, 1, -100]]),
        }

        def forward(input_ids, attention_mask):
            return SimpleNamespace(logits=logits)

        expected_ce = -torch.log_softmax(logits[0, 0], -1)[1]
        log_probs = torch.log_softmax(logits[0, 0], -1)
        expected_entropy = -(log_probs.exp() * log_probs).sum()
        torch.testing.assert_close(LLM.cross_entropy(forward, batch), expected_ce)
        torch.testing.assert_close(
            LLM.negative_entropy(forward, batch), -expected_entropy
        )
        for objective, expected in (
            ("ce", -expected_ce),
            ("entropy", -expected_entropy),
        ):
            result = LLM.whole_batch_metrics(forward, batch, objective)
            torch.testing.assert_close(result["outer_loss"], expected)
            torch.testing.assert_close(result["outer_ce"], expected_ce)
            torch.testing.assert_close(result["outer_entropy"], expected_entropy)

    def test_invalid_direction_and_radius_are_rejected(self):
        model, batches, outer, loss_fn = GRADIENTS.tiny_problem()
        arguments = (
            GRADIENTS,
            model,
            dict(model.named_parameters()),
            batches[:1],
            loss_fn,
            outer,
            loss_fn,
            GRADIENTS.OptimizerConfig(),
        )
        with self.assertRaises(ValueError):
            LLM.diagnostic(*arguments, epsilons=(math.nan,))
        with self.assertRaises(ValueError):
            LLM.diagnostic(*arguments, directions=("random",))


if __name__ == "__main__":
    unittest.main()
