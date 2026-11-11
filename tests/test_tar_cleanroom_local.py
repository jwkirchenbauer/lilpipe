"""CPU-only checks for norm-bounded local CE counterexample search."""

import contextlib
import io
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_tar_cleanroom as fixtures

CLEAN, torch = fixtures.CLEAN, fixtures.torch


class ConstantLogitLM(torch.nn.Module):
    def __init__(self, zero_gradient=False):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(vocab_size=2)
        self.zero_gradient = zero_gradient

    @property
    def device(self):
        return self.value.device

    def forward(self, input_ids, attention_mask=None, labels=None):
        value = self.value * (0 if self.zero_gradient else 1)
        logit = value.expand_as(input_ids)
        return SimpleNamespace(logits=torch.stack((logit, -logit), -1))


class NarrowLogitLM(ConstantLogitLM):
    """CE decreases only in a narrow neighborhood of the initialization."""

    def forward(self, input_ids, attention_mask=None, labels=None):
        score = -((self.value - 1e-4) / 1e-4).square()
        logit = score.expand_as(input_ids)
        return SimpleNamespace(logits=torch.stack((logit, torch.zeros_like(logit)), -1))


def cohort(label):
    return [
        {"input_ids": [label] * 3, "attention_mask": [1] * 3, "labels": [label] * 3}
        for _ in range(2)
    ]


class LocalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_projection_uses_fixed_anchor_and_actual_global_distance(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameters = {"p": parameter}
        anchor = {"p": torch.zeros(2)}
        displacement = {"p": torch.tensor([0.3, 0.4])}
        gradients = {"p": torch.tensor([1.0, 0.0])}
        expected = torch.tensor([0.2, 0.4])
        expected = expected * (0.2 / torch.linalg.vector_norm(expected))
        geometry = CLEAN.local_project_candidate(
            parameters, anchor, displacement, gradients, 0.1, 0.2, 2e-7
        )
        torch.testing.assert_close(parameter, expected, rtol=2e-6, atol=2e-8)
        self.assertLessEqual(geometry["distance_l2"], 0.2 + 2e-7)
        self.assertAlmostEqual(
            CLEAN.local_distance(parameters, anchor), geometry["distance_l2"]
        )
        torch.testing.assert_close(anchor["p"], torch.zeros(2), atol=0, rtol=0)
        torch.testing.assert_close(
            displacement["p"], torch.tensor([0.3, 0.4]), atol=0, rtol=0
        )

    def test_accepted_ce_is_monotone_inside_ball_and_heldout_never_selects(self):
        model = ConstantLogitLM()
        data = {"forget_train": cohort(0), "forget_heldout": cohort(1)}
        anchor = model.value.detach().clone()
        result = CLEAN.local_projected_search(model, data, radius=0.05, steps=2)
        values = [row["ce"] for row in result["trace"]]
        self.assertTrue(all(right <= left for left, right in zip(values, values[1:])))
        self.assertLess(values[-1], values[0])
        self.assertGreater(float(model.value.detach()), float(anchor))
        self.assertGreater(
            result["trace"][-1]["probes"]["forget_heldout"]["ce"],
            result["trace"][0]["probes"]["forget_heldout"]["ce"],
        )
        for row in result["trace"]:
            self.assertLessEqual(
                row["distance_l2"], 0.05 + result["radius_tolerance_l2"]
            )
            if row["step"]:
                self.assertEqual(len(row["candidates"]), 8)
                self.assertEqual(row["candidates"][0]["step_length"], 0)
                self.assertEqual(row["candidates"][0]["forward_calls"], 0)
        distance = float((model.value.detach().double() - anchor.double()).abs())
        self.assertAlmostEqual(distance, result["final_distance_l2"])
        self.assertTrue(result["ce_floor_counterexample_with_radius_tolerance"])
        self.assertIsNone(model.value.grad)

    def test_forward_backward_and_valid_token_budgets_include_candidate_search(self):
        model = ConstantLogitLM()
        result = CLEAN.local_projected_search(
            model,
            {"forget_train": cohort(0), "forget_heldout": cohort(1)},
            radius=0.05,
            steps=2,
        )
        budget = result["budget"]
        self.assertEqual(budget["gradient_forward_calls"], 2)
        self.assertEqual(budget["backward_calls"], 2)
        self.assertEqual(budget["gradient_forward_valid_tokens"], 8)
        self.assertEqual(budget["backward_valid_tokens"], 8)
        self.assertEqual(budget["candidate_forward_calls"], 14)
        self.assertEqual(budget["candidate_forward_valid_tokens"], 56)
        # Two groups are probed at states 0, 1, and the requested final state2.
        self.assertEqual(budget["probe_forward_calls"], 6)
        self.assertEqual(budget["probe_forward_valid_tokens"], 24)
        self.assertEqual(budget["forward_calls"], 22)
        self.assertEqual(budget["forward_valid_tokens"], 88)

    def test_zero_gradient_retains_current_point_without_extra_candidate_forwards(self):
        model = ConstantLogitLM(zero_gradient=True)
        initial = model.value.detach().clone()
        result = CLEAN.local_projected_search(
            model, {"forget_train": cohort(0)}, radius=0.1, steps=3
        )
        torch.testing.assert_close(model.value, initial, rtol=0, atol=0)
        self.assertEqual(result["final_distance_l2"], 0)
        self.assertAlmostEqual(result["final_ce"], math.log(2), places=6)
        self.assertEqual(result["budget"]["candidate_forward_calls"], 0)
        self.assertEqual(result["budget"]["backward_calls"], 3)
        for row in result["trace"][1:]:
            self.assertTrue(row["zero_gradient"])
            self.assertEqual(row["selected_step_length"], 0)
            self.assertEqual(len(row["candidates"]), 1)

    def test_chunked_gradient_matches_whole_batch_token_mean_with_ignored_labels(self):
        torch.manual_seed(13)
        model = fixtures.TinyLM()
        other = CLEAN.clone_model(model)
        gradients, record = CLEAN.local_ce_gradient(model, fixtures.rows())
        rows = CLEAN.batch(fixtures.rows(), other.device)
        ce, _ = CLEAN.loss_statistics(other, rows)
        ce.backward()
        norm = CLEAN.local_l2(p.grad for p in other.parameters())
        self.assertAlmostEqual(record["ce_before"], float(ce.detach()), delta=3e-7)
        self.assertEqual(record["forward_calls"], 2)
        self.assertEqual(record["backward_calls"], 2)
        self.assertEqual(record["backward_valid_tokens"], 9)
        self.assertAlmostEqual(record["gradient_norm"], norm, delta=2e-7)
        for name, parameter in other.named_parameters():
            torch.testing.assert_close(
                gradients[name], parameter.grad / norm, atol=2e-7, rtol=2e-6
            )

    def test_invalid_radius_and_non_fp32_parameters_are_rejected(self):
        for radius in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                CLEAN.local_projected_search(
                    ConstantLogitLM(), {"forget_train": cohort(0)}, radius=radius
                )
        with self.assertRaisesRegex(ValueError, "FP32"):
            CLEAN.local_projected_search(
                ConstantLogitLM().double(), {"forget_train": cohort(0)}
            )

    def test_fine_declared_grid_finds_descent_missed_by_default_grid(self):
        data = {"forget_train": cohort(0)}
        coarse = CLEAN.local_projected_search(NarrowLogitLM(), data, steps=1)
        fine_grid = (3e-5, 1e-4, 0.001)
        fine = CLEAN.local_projected_search(
            NarrowLogitLM(), data, steps=1, step_lengths=fine_grid
        )
        self.assertEqual(coarse["trace"][1]["selected_step_length"], 0)
        self.assertEqual(coarse["final_ce"], coarse["trace"][0]["ce"])
        self.assertEqual(fine["trace"][1]["selected_step_length"], 1e-4)
        self.assertLess(fine["final_ce"], coarse["final_ce"] - 0.5)
        self.assertEqual(fine["candidate_step_lengths"], [0.0, *fine_grid])
        self.assertEqual(
            [row["step_length"] for row in fine["trace"][1]["candidates"]],
            [0.0, *fine_grid],
        )
        self.assertEqual(fine["budget"]["candidate_forward_calls"], len(fine_grid))
        self.assertEqual(
            coarse["candidate_step_lengths"], [0.0, *CLEAN.LOCAL_STEP_LENGTHS]
        )

    def test_invalid_custom_grids_and_unsafe_output_names_are_rejected(self):
        for values in (
            (),
            (math.nan,),
            (math.inf,),
            (0,),
            (-0.1,),
            (0.003, 0.001),
            (0.001, 0.001),
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CLEAN.local_projected_search(
                    ConstantLogitLM(),
                    {"forget_train": cohort(0)},
                    steps=1,
                    step_lengths=values,
                )
        for name in (
            "",
            ".",
            "..",
            "../escape",
            "/absolute",
            "a/b",
            "a\\b",
            "two words",
            "-option",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CLEAN.local_attack(Path("/unused"), {}, {}, "control", output_name=name)
        self.assertEqual(
            CLEAN.validated_local_output_name("local-search-fine"), "local-search-fine"
        )
        self.assertEqual(CLEAN.validated_local_output_name("fine_2.0"), "fine_2.0")

    def test_cli_validates_before_io_and_threads_declared_grid_and_output(self):
        for option in (
            ("--local-step-lengths", "nan"),
            ("--local-output-name", "../escape"),
        ):
            with (
                patch(
                    "sys.argv",
                    [
                        "tar_cleanroom.py",
                        "local-attack",
                        "/unused",
                        "--case",
                        "control",
                        *option,
                    ],
                ),
                patch.object(CLEAN, "load_study") as load,
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    CLEAN.main()
                self.assertEqual(error.exception.code, 2)
                load.assert_not_called()
        study, data = {"fixture": "study"}, {"fixture": "data"}
        with (
            patch(
                "sys.argv",
                [
                    "tar_cleanroom.py",
                    "local-attack",
                    "/unused",
                    "--case",
                    "control",
                    "--local-step-lengths",
                    "0.00003",
                    "0.0001",
                    "0.001",
                    "--local-output-name",
                    "local-search-fine",
                ],
            ),
            patch.object(CLEAN, "load_study", return_value=study),
            patch.object(CLEAN, "prepare_data", return_value=data),
            patch.object(CLEAN, "local_attack") as attack,
        ):
            CLEAN.main()
        attack.assert_called_once_with(
            Path("/unused").resolve(),
            study,
            data,
            "control",
            0.75,
            8,
            False,
            step_lengths=(3e-5, 1e-4, 0.001),
            output_name="local-search-fine",
        )


if __name__ == "__main__":
    unittest.main()
