"""Check experiment controls, bootstrap identities, and BF16 noise measurement."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/tamper-resistance"
MISC = EXAMPLE / "scripts/misc"


def functions(path, names, scope):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


class ConfigTests(unittest.TestCase):
    def test_compute_changes_preserve_unrelated_settings(self):
        import yaml

        spec = importlib.util.spec_from_file_location(
            "tar_compute", MISC / "tar_compute.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = (
            EXAMPLE / "configs/stages-wmdp/0_unlearn" / module.DEFENSE / "selected.yml"
        )
        reference = yaml.safe_load(source.read_text())
        original = copy.deepcopy(reference)
        for updates, epochs, warmup in ((500, 1, 48), (125, 4, 12)):
            actual = module.case_config(
                reference,
                base_model="/base",
                output=Path("/run/case/models/defense"),
                prepared="/prepared",
                role="defense",
                steps=updates,
                epochs=epochs,
                warmup=warmup,
            )
            self.assertEqual(
                updates * 16 * actual["custom_trainer_args"]["attack_epochs"], 8000
            )
            self.assertEqual(actual["warmup_steps"] / updates, 12 / 125)
            for key in (
                "learning_rate",
                "seed",
                "data_seed",
                "micro_batch_size",
                "weight_decay",
                "max_grad_norm",
                "device_map",
                "datasets",
            ):
                self.assertEqual(actual[key], reference[key])
            self.assertFalse(actual["save_only_model"])
        self.assertEqual(reference, original)

    def test_validator_counts_zero_based_inner_updates(self):
        spec = importlib.util.spec_from_file_location(
            "tar_compute", MISC / "tar_compute.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            model = folder / "models/defense"
            checkpoint = model / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            (folder / "case.json").write_text(
                json.dumps(
                    {
                        "role": "defense",
                        "name": "test",
                        "max_steps": 1,
                        "save_steps": [1],
                        "state_steps": [1],
                        "inner_steps_per_update": 4,
                    }
                )
            )
            for filename in (
                "model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
            ):
                (checkpoint / filename).touch()
            state = {
                "global_step": 1,
                "log_history": [{"inner_step_total": step} for step in range(4)],
            }
            (checkpoint / "trainer_state.json").write_text(json.dumps(state))
            (model / "observed-batches.jsonl").write_text(
                json.dumps(
                    {"step": 1, "loss_before_update": 1.0, "elapsed_seconds": 1.0}
                )
                + "\n"
            )
            module.validate_case(folder)
            state["log_history"].pop(0)
            (checkpoint / "trainer_state.json").write_text(json.dumps(state))
            with self.assertRaises(AssertionError):
                module.validate_case(folder)


class DatasetTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_preserves_source_identities(self):
        class Dataset:
            _fingerprint = "fixed-prepared-order"

            def __init__(self, rows):
                self.rows = rows

            def __len__(self):
                return len(self.rows)

            def __iter__(self):
                return iter(self.rows)

            def select(self, indices):
                return Dataset([self.rows[i] for i in indices])

        scope = functions(
            MISC / "tar_dynamics_training.py",
            {"select_attack_dataset"},
            {"torch": torch, "hashlib": hashlib, "json": json, "Path": Path},
        )
        select = scope["select_attack_dataset"]
        rows = [
            {"input_ids": [i, i + 1], "attention_mask": [1, 1], "labels": [i, i + 1]}
            for i in range(20)
        ]
        dataset = Dataset(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            selected, lookup = select(dataset, path, 101)
            manifest = json.loads(path.read_text())
            again, _ = select(dataset, path, 101)
            self.assertEqual(selected.rows, again.rows)
            self.assertEqual(len(selected), len(rows))
            self.assertLess(manifest["unique_source_rows"], len(rows))
            self.assertEqual(sum(manifest["source_multiplicities"].values()), len(rows))
            self.assertEqual(
                manifest["prepared_dataset_fingerprint"], dataset._fingerprint
            )
            for row, source in zip(selected, manifest["drawn_source_rows"]):
                serial = {key: [row[key]] for key in row}
                signature = hashlib.sha256(
                    json.dumps(serial, sort_keys=True).encode()
                ).hexdigest()
                self.assertIn(source, lookup[signature])
            control, _ = select(dataset, path)
            self.assertIs(control, dataset)
            self.assertEqual(rows, dataset.rows)


class LandscapeSamplingTests(unittest.TestCase):
    def test_matrix_positions_and_common_reference_settings(self):
        spec = importlib.util.spec_from_file_location(
            "tar_compute", MISC / "tar_compute.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        positions = module.matrix_positions(1000)
        self.assertEqual(len(positions), 128)
        self.assertEqual(len(set(positions)), 128)
        self.assertTrue(set(range(1, 17)) | {32, 128, 1000} <= set(positions))
        program = {
            "baseline_dynamics": "/baseline",
            "original_run": "/original",
            "base_model": "/model",
        }
        case = {
            "name": "long-lr1e-6",
            "folder": "/study/cases/long-lr1e-6",
            "max_steps": 1000,
            "gaussian_sigma": None,
            "save_steps": module.ATTACK_STEPS,
            "attack_learning_rate": 1e-6,
        }
        probes = {
            "fixed_source_indices": list(range(10)),
            "heldout_source_indices": list(range(1000, 1032)),
            "forget_revision": "fixed",
        }
        settings = module.analysis_settings(Path("/study"), program, case, probes)
        self.assertEqual(settings["local_plane"], "reference")
        self.assertEqual(settings["plane_radii"], {"random": 1.0, "reference": 1.0})
        self.assertEqual(settings["actual_attack_steps"], case["save_steps"])
        self.assertFalse(settings["compare_original_weights"])
        self.assertIn(500, settings["defense_frames"])
        noisy = {**case, "name": "noise-1e-3", "gaussian_sigma": 1e-3}
        settings = module.analysis_settings(Path("/study"), program, noisy, probes)
        self.assertTrue(
            settings["attack_anchor"].endswith("models/attack/checkpoint-0")
        )

    def test_sparse_future_batches_are_distinct_from_unseen_content(self):
        spec = importlib.util.spec_from_file_location(
            "tar_dynamics", MISC / "tar_dynamics.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        records = [
            {
                "step": 1,
                "batch_sha256": "A",
                "loss_before_update": 10,
                "loss_after_update": 9,
            },
            {
                "step": 2,
                "batch_sha256": "B",
                "loss_before_update": 10,
                "loss_after_update": 8,
            },
            {
                "step": 3,
                "batch_sha256": "A",
                "loss_before_update": 9,
                "loss_after_update": 8,
            },
            {
                "step": 4,
                "batch_sha256": "D",
                "loss_before_update": 10,
                "loss_after_update": 7,
            },
        ]
        recovery, observations = module.minibatch_matrix_diagnostics(
            [[10, 10, 10], [9, 9, 10], [8, 8, 10], [5, 5, 7]],
            [0, 1, 3, 4],
            [1, 3, 4],
            records,
        )
        self.assertEqual(recovery[1]["mean_loss_reduction_future_batches"], 0.5)
        self.assertEqual(recovery[1]["mean_loss_reduction_unseen_content"], 0)
        self.assertEqual(recovery[2]["mean_loss_reduction_already_used_batches"], 2)
        self.assertEqual(recovery[2]["sampled_future_positions"], 1)
        self.assertEqual(len(observations), 5)
        self.assertEqual(max(row["absolute_difference"] for row in observations), 0)

    def test_shared_probes_exclude_every_observed_cohort_and_stay_frozen(self):
        spec = importlib.util.spec_from_file_location(
            "tar_compute", MISC / "tar_compute.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [int(text.split()[0])], "attention_mask": [1]}

        corpus = [{"text": str(index) + " " + "A" * 60} for index in range(1100)]
        tokenizer_class = SimpleNamespace(from_pretrained=lambda path: Tokenizer())
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "source").mkdir()
            program = {
                "baseline_dynamics": "/baseline",
                "original_run": "/original",
                "base_model": "/model",
                "cases": {},
            }
            for offset, name in enumerate(
                ("baseline-lr1e-5", "bootstrap-101", "bootstrap-102", "bootstrap-103")
            ):
                folder = run / "cases" / name
                (folder / "models/attack").mkdir(parents=True)
                (folder / "status").mkdir()
                (folder / "status/train.json").write_text('{"state":"complete"}')
                records = [
                    {
                        "step": step,
                        "batch_sha256": str(1000 + offset),
                        "batch": {
                            "input_ids": [[1000 + offset]],
                            "attention_mask": [[1]],
                        },
                    }
                    for step in range(1, 1001)
                ]
                (folder / "models/attack/observed-batches.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in records) + "\n"
                )
                (folder / "config.yml").write_text("datasets:\n- revision: fixed\n")
                program["cases"][name] = {
                    "name": name,
                    "folder": str(folder),
                    "role": "attack",
                    "max_steps": 1000,
                    "save_steps": module.ATTACK_STEPS,
                    "gaussian_sigma": None,
                    "attack_learning_rate": 1e-5,
                }
            (run / "study.json").write_text(json.dumps(program))
            fake = {
                "datasets": SimpleNamespace(
                    load_dataset=lambda *args, **kwargs: corpus
                ),
                "transformers": SimpleNamespace(AutoTokenizer=tokenizer_class),
            }
            with patch.dict("sys.modules", fake):
                module.prepare_analysis(run)
                initial = (run / "probes.json").read_bytes()
                module.prepare_analysis(run)
            self.assertEqual(initial, (run / "probes.json").read_bytes())
            probes = json.loads(initial)
            self.assertEqual(probes["heldout_source_indices"], list(range(1004, 1036)))
            for name in program["cases"]:
                settings = json.loads((run / "cases" / name / "study.json").read_text())
                self.assertEqual(
                    settings["heldout_source_indices"], probes["heldout_source_indices"]
                )


class NoiseTests(unittest.TestCase):
    def test_noise_measurement_records_cast_displacement_and_preserves_global_rng(self):
        scope = functions(
            EXAMPLE / "src/trainers/rl_perturb.py",
            {"add_gaussian_parameter_noise"},
            {"torch": torch, "math": math},
        )
        noise = scope["add_gaussian_parameter_noise"]
        first = torch.nn.Linear(16, 16, bias=False).to(torch.bfloat16)
        with torch.no_grad():
            first.weight.fill_(1)
        second = copy.deepcopy(first)
        before = first.weight.detach().clone()
        random_before = torch.get_rng_state().clone()
        tiny = noise(first, 1e-7, 42)
        self.assertTrue(torch.equal(torch.get_rng_state(), random_before))
        self.assertEqual(tiny["actual_l2_after_cast"], 0)
        self.assertGreater(tiny["draw_l2_before_cast"], 0)
        result = noise(first, 0.1, 42)
        noise(second, 0.1, 42)
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertAlmostEqual(
            result["actual_l2_after_cast"],
            float((first.weight.detach().double() - before.double()).norm()),
        )


class StateSaveTests(unittest.TestCase):
    def test_selected_checkpoints_keep_optimizer_state(self):
        scope = functions(
            MISC / "tar_dynamics_training.py",
            {"DynamicsCallback"},
            {
                "TrainerCallback": object,
                "os": os,
                "torch": torch,
                "time": time,
                "Path": Path,
                "json": json,
                "math": math,
            },
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"DYNAMICS_STATE_STEPS": "2"}),
        ):
            trainer = SimpleNamespace(
                _measure_after=False, _dynamics_started=time.perf_counter()
            )
            callback = scope["DynamicsCallback"](trainer, [1, 2])
            args = SimpleNamespace(output_dir=directory, save_only_model=False)
            for step, weights_only in ((1, True), (2, False)):
                trainer._dynamics_record = {"step": step}
                trainer._dynamics_batch = None
                control = SimpleNamespace(should_save=False)
                callback.on_step_end(args, SimpleNamespace(global_step=step), control)
                self.assertTrue(control.should_save)
                self.assertEqual(args.save_only_model, weights_only)


if __name__ == "__main__":
    unittest.main()
