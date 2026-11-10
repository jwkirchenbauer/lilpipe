"""Check early-update interpolation and restartable diagnostics on tiny tensors."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_dynamics.py"
)
SPEC = importlib.util.spec_from_file_location("tar_step_profiles", SCRIPT)
DYNAMICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DYNAMICS)


class ProfileMathTests(unittest.TestCase):
    def test_fp32_anchor_is_immutable_and_raw_motion_can_vanish_after_cast(self):
        before = torch.tensor([1.0, 2.0], dtype=torch.float32)
        after = torch.tensor([1.0001, 2.0], dtype=torch.float32)
        saved_before, saved_after = before.clone(), after.clone()
        delta = after.float() - before.float()
        for fraction in (-1, 0, 0.25, 1, 2, -0.5):
            DYNAMICS._profile_interpolate(before, delta, fraction, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(before, saved_before))
        self.assertTrue(torch.equal(after, saved_after))
        self.assertTrue(
            torch.equal(
                DYNAMICS._profile_interpolate(before, delta, 0, dtype=torch.bfloat16),
                before.bfloat16(),
            )
        )
        self.assertTrue(
            torch.equal(
                DYNAMICS._profile_interpolate(before, delta, 1, dtype=torch.bfloat16),
                after.bfloat16(),
            )
        )
        raw_squared, raw_changed = DYNAMICS._profile_update_statistics(
            before, after, chunk=1
        )
        evaluated_squared, evaluated_changed = DYNAMICS._profile_update_statistics(
            before, after, dtype=torch.bfloat16, chunk=1
        )
        self.assertGreater(raw_squared, 0)
        self.assertEqual(raw_changed, 1)
        self.assertEqual(evaluated_squared, 0)
        self.assertEqual(evaluated_changed, 0)

    def test_bf16_interpolation_has_exact_saved_endpoints_and_restores_anchor(self):
        before = torch.tensor([0, 0.125, -2, 8, 1e-5], dtype=torch.bfloat16)
        after = torch.tensor([0.25, -0.25, -1.9921875, 9, -1e-5], dtype=torch.bfloat16)
        anchor = before.clone()
        delta = after.float() - before.float()
        DYNAMICS._profile_interpolate(before, delta, -1)
        self.assertTrue(
            torch.equal(DYNAMICS._profile_interpolate(before, delta, 0), before)
        )
        self.assertTrue(
            torch.equal(DYNAMICS._profile_interpolate(before, delta, 1), after)
        )
        self.assertTrue(torch.equal(before, anchor))

    def test_first_plane_contains_update_and_is_orthonormal(self):
        delta = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 0.0])}
        original = {name: value.clone() for name, value in delta.items()}
        random = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
        proof = DYNAMICS._profile_first_plane(delta, random)
        self.assertEqual(proof["first_update_l2"], 5)
        self.assertEqual(proof["status"], "fitted")
        self.assertAlmostEqual(
            DYNAMICS._profile_inner(delta, random, chunk=1), 0, places=6
        )
        self.assertAlmostEqual(
            DYNAMICS._profile_inner(delta, delta, chunk=1), 1, places=6
        )
        self.assertAlmostEqual(
            DYNAMICS._profile_inner(random, random, chunk=1), 1, places=6
        )
        for name, value in delta.items():
            torch.testing.assert_close(value * 5, original[name])

    def test_zero_update_never_creates_a_direction(self):
        delta = {"a": torch.zeros(3)}
        random = {"a": torch.tensor([1.0, 2.0, 3.0])}
        proof = DYNAMICS._profile_first_plane(delta, random)
        self.assertEqual(proof["status"], "skipped_zero_first_update")
        self.assertNotIn("q0_norm", proof)
        self.assertTrue(torch.equal(random["a"], torch.tensor([1.0, 2.0, 3.0])))

    def test_sampled_barrier_uses_segment_and_higher_endpoint(self):
        rows = [
            {"fraction": t, "probe_loss": loss}
            for t, loss in ((-1, 100), (0, 5), (0.5, 7), (1, 6), (2, 100))
        ]
        self.assertEqual(DYNAMICS._profile_barrier(rows, "probe_loss"), 1)
        self.assertIsNone(DYNAMICS._profile_barrier(rows[:3], "probe_loss"))
        rows[2]["probe_loss"] = 4
        self.assertEqual(DYNAMICS._profile_barrier(rows, "probe_loss"), 0)

    def test_plot_limits_include_both_objectives_without_separate_normalization(self):
        low, high = DYNAMICS._profile_loss_limits(
            [
                {"probe_loss": 2, "minibatch_loss": 100},
                {"probe_loss": 3, "minibatch_loss": 40},
            ]
        )
        self.assertLess(low, 2)
        self.assertGreater(high, 100)


class ProfileCacheTests(unittest.TestCase):
    @staticmethod
    def row(fraction=0):
        return {
            "update_step": 1,
            "fraction": fraction,
            "signed_displacement": fraction,
            "probe_loss": 3,
            "minibatch_loss": 2,
        }

    def test_partial_points_resume_and_changed_input_clears_both_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            source = study / "checkpoint"
            source.write_bytes(b"first")
            protocol = {
                "source_sha256": "frozen",
                "inputs": [DYNAMICS._profile_file_identity(source)],
            }
            cache = DYNAMICS._ProfileCache(study, protocol)
            cache.append("profiles", self.row())
            cache.append(
                "grid", {"c0": 0, "c1": 0, "probe_loss": 3, "minibatch_loss": 2}
            )
            resumed = DYNAMICS._ProfileCache(study, protocol)
            self.assertEqual(len(resumed.rows["profiles"]), 1)
            self.assertEqual(len(resumed.rows["grid"]), 1)
            source.write_bytes(b"replacement weights")
            protocol["inputs"] = [DYNAMICS._profile_file_identity(source)]
            changed = DYNAMICS._ProfileCache(study, protocol)
            self.assertNotEqual(
                changed.metadata["identity"], cache.metadata["identity"]
            )
            self.assertEqual(changed.rows, {"profiles": [], "grid": []})

    def test_interrupted_atomic_publication_retains_previous_points(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            cache = DYNAMICS._ProfileCache(study, {"version": 1})
            cache.append("profiles", self.row())
            with patch.object(
                Path, "replace", side_effect=OSError("interrupted rename")
            ):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    cache.append("profiles", self.row(0.125))
            resumed = DYNAMICS._ProfileCache(study, {"version": 1})
            self.assertEqual([row["fraction"] for row in resumed.rows["profiles"]], [0])


class ToyProfileIntegrationTests(unittest.TestCase):
    def test_interrupted_toy_measurements_resume_with_exact_endpoints_and_no_repeated_evaluations(
        self,
    ):
        self.run_toy(False)

    def test_mixed_saved_dtypes_preserve_raw_and_evaluated_update_distinction(self):
        self.run_toy(True)

    def run_toy(self, mixed):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            source = study / "source-geometry"
            source.mkdir()
            np.save(source / "random-basis.npy", np.eye(4, dtype=np.float32)[1:3])
            (source / "geometry.json").write_text(
                json.dumps(
                    {
                        "names": ["weight", "other"],
                        "shapes": {"weight": [2], "other": [2]},
                        "offsets": {"weight": 0, "other": 2},
                    }
                )
            )
            checkpoints = [study / "models/defense"] + [
                study / f"models/attack/checkpoint-{step}" for step in range(1, 9)
            ]
            weights = {}
            tokenizer_sources = []
            for step, path in enumerate(checkpoints):
                path.mkdir(parents=True)
                (path / "model.safetensors").write_bytes(b"tiny immutable fixture")
                weights[path] = {
                    "weight": torch.tensor(
                        [1 + step / 10000 if mixed else step / 8, 0],
                        dtype=torch.float32 if mixed else torch.bfloat16,
                    ),
                    "other": torch.zeros(2, dtype=torch.bfloat16),
                }
            (checkpoints[0] / "config.json").write_text('{"model_type":"toy"}')
            (checkpoints[0] / "tokenizer_config.json").write_text('{"pad_token":"pad"}')
            records = []
            for step in range(1, 9):
                batch = {
                    "input_ids": [[1, 2]],
                    "attention_mask": [[1, 1]],
                    "labels": [[1, 2]],
                }
                records.append(
                    {
                        "step": step,
                        "batch": batch,
                        "batch_sha256": hashlib.sha256(
                            json.dumps(batch, sort_keys=True).encode()
                        ).hexdigest(),
                        "loss_before_update": 0.5,
                    }
                )
            (study / "models/attack/observed-batches.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records)
            )
            settings = {
                "shared_random_basis": str(source / "random-basis.npy"),
                "fixed_source_indices": list(range(10)),
                "forget_revision": "frozen",
            }
            calls = [0]
            interrupt = [True]

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(
                        torch.zeros(2, dtype=torch.bfloat16)
                    )
                    self.other = torch.nn.Parameter(
                        torch.zeros(2, dtype=torch.bfloat16)
                    )

                def forward(self, input_ids, **kwargs):
                    calls[0] += 1
                    if calls[0] == 43 and interrupt[0]:
                        interrupt[0] = False
                        raise RuntimeError("interrupted toy evaluation")
                    loss = (
                        self.weight[0].float() - 0.1 * input_ids.float().mean()
                    ).square() + (self.other[0].float() + 1).square()
                    return SimpleNamespace(loss=loss)

            class Tokenizer:
                pad_token = "pad"

                def __call__(self, text, **kwargs):
                    return {"input_ids": [2, 3], "attention_mask": [1, 1]}

            def collate(rows):
                batch = {
                    key: torch.tensor([row[key] for row in rows]) for key in rows[0]
                }
                batch["labels"] = batch["input_ids"].clone()
                return batch

            @contextlib.contextmanager
            def safe_open(path, **kwargs):
                yield SimpleNamespace(
                    get_tensor=lambda name: weights[Path(path).parent][name]
                )

            with (
                patch.dict(
                    "sys.modules",
                    {
                        "safetensors": SimpleNamespace(safe_open=safe_open),
                        "datasets": SimpleNamespace(
                            load_dataset=lambda *args, **kwargs: [
                                {"text": f"doc {index}"} for index in range(10)
                            ]
                        ),
                        "transformers": SimpleNamespace(
                            AutoModelForCausalLM=SimpleNamespace(
                                from_pretrained=lambda *args, **kwargs: Model()
                            ),
                            AutoTokenizer=SimpleNamespace(
                                from_pretrained=lambda *args: tokenizer_sources.append(
                                    args[0]
                                )
                                or Tokenizer()
                            ),
                            DataCollatorForLanguageModeling=lambda *args, **kwargs: collate,
                        ),
                    },
                ),
                patch.object(torch, "device", return_value=torch.device("cpu")),
                patch.object(torch.cuda, "empty_cache"),
                patch.object(DYNAMICS, "_profile_plots"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted toy"):
                    DYNAMICS.step_profiles(study, settings)
                cache = DYNAMICS._ProfileCache(
                    study,
                    json.loads((study / "surfaces/step-profiles.json").read_text())[
                        "protocol"
                    ],
                )
                self.assertEqual(len(cache.rows["profiles"]), 7)
                DYNAMICS.step_profiles(study, settings)
                metadata = json.loads(
                    (study / "surfaces/step-profiles.json").read_text()
                )
                self.assertEqual(metadata["state"], "complete")
                for checks in metadata["steps"].values():
                    self.assertTrue(checks["t0_weights_exact"])
                    self.assertTrue(checks["t1_weights_exact"])
                    self.assertEqual(checks["changed_scalars"], 1)
                    self.assertEqual(checks["changed_fraction"], 0.25)
                    self.assertEqual(checks["raw_changed_scalars"], 1)
                    self.assertGreater(checks["raw_update_l2"], 0)
                    if mixed:
                        self.assertEqual(checks["evaluated_changed_scalars"], 0)
                        self.assertEqual(checks["evaluated_changed_fraction"], 0)
                        self.assertEqual(checks["evaluated_update_l2"], 0)
                        self.assertEqual(
                            checks["saved_dtype_scalars"]["before"]["torch.float32"], 2
                        )
                    else:
                        self.assertEqual(checks["evaluated_changed_scalars"], 1)
                        self.assertEqual(
                            checks["raw_update_l2"], checks["evaluated_update_l2"]
                        )
                    self.assertEqual(max(checks["endpoint_errors"].values()), 0)
                    self.assertEqual(checks["sampled_barriers"]["probe_loss"], 0)
                self.assertEqual(
                    metadata["first_update_plane"]["plane_endpoint_bf16_mismatches"], 0
                )
                self.assertEqual(
                    metadata["first_update_plane"]["center_errors"],
                    {"probe_loss": 0, "minibatch_loss": 0},
                )
                self.assertEqual(
                    metadata["first_update_plane"]["plane_endpoint_ce_errors"],
                    {"probe_loss": 0, "minibatch_loss": 0},
                )
                self.assertEqual(
                    metadata["evaluation_dtype_scalars"], {"torch.bfloat16": 4}
                )
                prior = calls[0]
                DYNAMICS.step_profiles(study, settings)
                self.assertEqual(calls[0], prior)
                self.assertEqual(tokenizer_sources, [checkpoints[0]] * 3)
                bound = {
                    Path(value["path"]).name for value in metadata["protocol"]["inputs"]
                }
                self.assertTrue({"config.json", "tokenizer_config.json"} <= bound)
            protocol = json.loads((study / "surfaces/step-profiles.json").read_text())[
                "protocol"
            ]
            cache = DYNAMICS._ProfileCache(study, protocol)
            self.assertEqual(len(cache.rows["profiles"]), 200)
            self.assertEqual(len(cache.rows["grid"]), 289)


if __name__ == "__main__":
    unittest.main()
