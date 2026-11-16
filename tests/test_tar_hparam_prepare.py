"""Preparation-only contracts for the original-path TAR hyperparameter study."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
MISC = EXAMPLE / "scripts/misc"
SPEC = importlib.util.spec_from_file_location(
    "tar_hparam_prepare", MISC / "tar_compute.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HparamPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.original = self.root / "original"
        self.baseline = self.root / "baseline"
        self.compute = self.root / "compute"
        self.run = self.root / "new-study"
        self.control = self.compute / "cases/baseline-lr1e-6"
        for path in (
            self.original / "configs",
            self.original / "cache/axolotl/defense",
            self.baseline / "source/misc",
            self.compute / "source/lm_eval_tasks",
            self.compute / "cache/attack",
            self.control / "models/attack",
        ):
            path.mkdir(parents=True)
        source = (
            EXAMPLE / "configs/stages-wmdp/0_unlearn" / MODULE.DEFENSE / "selected.yml"
        )
        self.defense = yaml.safe_load(source.read_text())
        self.defense["base_model"] = str(self.root / "pinned-base-snapshot")
        self.defense["wandb_project"] = "frozen-original-project"
        for dataset in self.defense["datasets"]:
            dataset["revision"] = "frozen-" + dataset["path"].split("/")[-1]
        self.defense_path = self.original / "configs/defense.yml"
        self.defense_path.write_text(yaml.safe_dump(self.defense, sort_keys=False))
        attack_source = (
            EXAMPLE
            / "configs/stages-wmdp/2_relearn"
            / f"{MODULE.DEFENSE}--relearn-fft-lr1e-6-ms1000/selected.yml"
        )
        self.attack = yaml.safe_load(attack_source.read_text())
        self.attack["base_model"] = str(self.original / "models/defense")
        for dataset in self.attack["datasets"]:
            dataset["revision"] = "frozen-" + dataset["path"].split("/")[-1]
        (self.control / "config.yml").write_text(
            yaml.safe_dump(self.attack, sort_keys=False)
        )
        (self.control / "training-validation.json").write_text('{"state":"passed"}\n')
        records = [{"step": step, "batch_sha256": str(step)} for step in range(1, 1001)]
        self.observed = self.control / "models/attack/observed-batches.jsonl"
        self.observed.write_text("\n".join(map(json.dumps, records)) + "\n")
        order = hashlib.sha256(
            json.dumps([row["batch_sha256"] for row in records]).encode()
        ).hexdigest()
        self.probes = {
            "fixed_source_indices": list(range(10)),
            "heldout_source_indices": list(range(1000, 1032)),
            "forget_revision": "frozen-wmdp-bio-forget-corpus",
            "heldout_sha256": [f"text-{step}" for step in range(32)],
            "heldout_token_sha256": [f"tokens-{step}" for step in range(32)],
            "expected_nonbootstrap_order_sha256": order,
        }
        (self.compute / "probes.json").write_text(json.dumps(self.probes))
        (self.baseline / "source/misc/reference_get_ll.py").write_text(
            "# original helper\n"
        )
        (self.compute / "source/lm_eval_tasks/frozen.yaml").write_text(
            "group: frozen\n"
        )
        for cache in (
            self.original / "cache/axolotl/defense",
            self.compute / "cache/attack",
        ):
            (cache / "prepared.arrow").write_bytes(b"immutable prepared rows")
            (cache / "prepared.lock").write_text("not copied")

    def prepare(self, matrix=None):
        with patch.object(
            MODULE.subprocess, "check_output", return_value="frozen-commit\n"
        ):
            MODULE.prepare_hparam_study(
                self.run, EXAMPLE, self.original, self.baseline, self.compute, matrix
            )
        return json.loads((self.run / "study.json").read_text())

    def test_only_requested_scientific_values_change_in_fresh_defense_attack_pairs(
        self,
    ):
        program = self.prepare()
        self.assertEqual(len(program["cases"]), 8)
        runtime = {
            "base_model",
            "output_dir",
            "dataset_prepared_path",
            "trainer_cls",
            "wandb_name",
            "wandb_project",
            "save_steps",
            "save_strategy",
            "save_only_model",
            "save_total_limit",
        }
        for name, case in program["cases"].items():
            with self.subTest(case=name):
                folder = self.run / "cases" / name
                actual = yaml.safe_load((folder / "config.yml").read_text())
                reference = self.defense if case["role"] == "defense" else self.attack
                self.assertEqual(
                    yaml.safe_load((folder / "config.reference.yml").read_text()),
                    reference,
                )
                scientific = copy.deepcopy(actual)
                if case["role"] == "defense":
                    values = MODULE.HPARAM_MATRIX[case["condition"]]
                    self.assertEqual(
                        actual["learning_rate"], values["outer_learning_rate"]
                    )
                    self.assertEqual(
                        actual["custom_trainer_args"]["retain_coefficient"],
                        values["retain_coefficient"],
                    )
                    self.assertEqual(
                        actual["custom_trainer_args"]["attack_learning_rate"],
                        values["inner_learning_rate"],
                    )
                    scientific["learning_rate"] = reference["learning_rate"]
                    for field in ("retain_coefficient", "attack_learning_rate"):
                        scientific["custom_trainer_args"][field] = reference[
                            "custom_trainer_args"
                        ][field]
                    self.assertEqual(case["max_steps"], 125)
                    self.assertEqual(
                        case["save_steps"], [1, 2, 4, 8, 16, 32, 64, 96, 125]
                    )
                    self.assertEqual(case["state_steps"], [32, 64, 125])
                    self.assertEqual(case["inner_steps_per_update"], 16)
                    self.assertEqual(
                        actual["trainer_cls"],
                        "tar_dynamics_training.RecordedTARMETrainer",
                    )
                    self.assertNotIn(
                        "attack_weight_decay", actual["custom_trainer_args"]
                    )
                    self.assertEqual(actual["base_model"], self.defense["base_model"])
                else:
                    self.assertEqual(case["max_steps"], 1000)
                    self.assertEqual(case["save_steps"], MODULE.ATTACK_STEPS)
                    self.assertEqual(case["state_steps"], [128, 512, 1000])
                    self.assertEqual(
                        actual["trainer_cls"],
                        "tar_dynamics_training.RecordedRelearnTrainer",
                    )
                    self.assertEqual(actual["base_model"], case["defense_reference"])
                    self.assertEqual(
                        (folder / "models/defense").resolve(),
                        Path(case["defense_reference"]),
                    )
                self.assertEqual(
                    {
                        key: value
                        for key, value in scientific.items()
                        if key not in runtime
                    },
                    {
                        key: value
                        for key, value in reference.items()
                        if key not in runtime
                    },
                )
                self.assertFalse(actual["save_only_model"])
                runtime_text = (folder / "runtime.sh").read_text()
                self.assertIn("DYNAMICS_SAVE_STEPS=", runtime_text)
                self.assertIn("DYNAMICS_STATE_STEPS=", runtime_text)
                self.assertEqual(
                    "DYNAMICS_SOURCE_MANIFEST=" in runtime_text,
                    case["role"] == "attack",
                )
                self.assertFalse((folder / "status/train.json").exists())
        self.assertEqual(yaml.safe_load(self.defense_path.read_text()), self.defense)

    def test_reuses_only_external_B_and_keeps_real_defense_history_and_shared_probes(
        self,
    ):
        program = self.prepare()
        self.assertEqual(set(program["comparison_controls"]), {"reproduced-1000"})
        self.assertEqual(
            program["comparison_controls"]["reproduced-1000"]["folder"],
            str(self.control),
        )
        self.assertEqual(
            program["comparison_controls"]["reproduced-1000"]["schedule_steps"], 1000
        )
        self.assertFalse((self.run / "cases/baseline-lr1e-6").exists())
        self.assertEqual(program["profile_reference_case"], "hparam-r-lr1e-6")
        self.assertEqual(
            program["cases"][program["profile_reference_case"]]["role"], "attack"
        )
        self.assertEqual(len(program["bundles"]["defenses"]), 4)
        self.assertEqual(len(program["bundles"]["attacks"]), 4)
        self.assertEqual(program["evaluation_cases"], program["bundles"]["attacks"])
        self.assertEqual(program["jobs"], {})
        self.assertIn("local metadata sampler", program["checkpoint_contrast_note"])
        self.assertEqual(
            (self.run / "probes.json").read_bytes(),
            (self.compute / "probes.json").read_bytes(),
        )
        for name in program["bundles"]["attacks"]:
            case = program["cases"][name]
            settings = json.loads(
                (self.run / "cases" / name / "study.json").read_text()
            )
            self.assertEqual(
                settings["defense_checkpoint_root"], case["defense_reference"]
            )
            self.assertEqual(settings["defense_steps"], MODULE.DEFENSE_STEPS)
            self.assertEqual(settings["defense_frames"], [0, 8, 16, 32, 64, 125])
            self.assertEqual(settings["broad_defense_frames"], [0, 8, 32, 125])
            self.assertEqual(
                settings["fixed_source_indices"], self.probes["fixed_source_indices"]
            )
            self.assertEqual(
                settings["heldout_source_indices"],
                self.probes["heldout_source_indices"],
            )
            self.assertEqual(
                settings["shared_random_basis"],
                str(self.baseline / "geometry/random-basis.npy"),
            )
            self.assertEqual(
                settings["reference_bases"],
                {"reference": str(self.baseline / "geometry/attack-basis.npy")},
            )
            self.assertEqual(settings["local_plane"], "reference")
            self.assertEqual(settings["local_radius"], 0.15713339880469646)
            self.assertEqual(settings["attack_max_steps"], 1000)

    def test_immutable_cache_links_source_snapshots_and_matrix_provenance(self):
        program = self.prepare()
        for role, source in (
            ("defense", self.original / "cache/axolotl/defense"),
            ("attack", self.compute / "cache/attack"),
        ):
            target = self.run / "cache" / role
            self.assertEqual(
                (source / "prepared.arrow").stat().st_ino,
                (target / "prepared.arrow").stat().st_ino,
            )
            self.assertFalse((target / "prepared.lock").exists())
            self.assertEqual(
                (source / "prepared.arrow").read_bytes(), b"immutable prepared rows"
            )
        for relative, signature in program["source_sha256"].items():
            self.assertEqual(
                hashlib.sha256((self.run / relative).read_bytes()).hexdigest(),
                signature,
            )
        for filename in (
            "tar_compute.py",
            "tar_dynamics.py",
            "tar_dynamics_training.py",
            "tar_compute_compare.py",
            "tar_compute_evaluation.py",
            "run_tar_compute.sh",
            "run_tar_compute_landscapes.sh",
            "run_tar_compute_evaluation.sh",
            "reference_get_ll.py",
        ):
            self.assertIn(f"source/misc/{filename}", program["source_sha256"])
        self.assertIn("source/lm_eval_tasks/frozen.yaml", program["source_sha256"])
        self.assertEqual(
            json.loads((self.run / "matrix.json").read_text()), MODULE.HPARAM_MATRIX
        )
        self.assertEqual(
            program["matrix_sha256"],
            hashlib.sha256((self.run / "matrix.json").read_bytes()).hexdigest(),
        )
        frozen = (self.run / "study.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.run / "study.json").read_bytes(), frozen)

    def test_custom_matrix_can_add_a_pairwise_condition(self):
        matrix = {
            "RI": {
                "retain_coefficient": 4,
                "inner_learning_rate": 1e-5,
                "outer_learning_rate": 5e-6,
            }
        }
        program = self.prepare(matrix)
        self.assertEqual(
            set(program["cases"]), {"hparam-ri-defense", "hparam-ri-lr1e-6"}
        )
        self.assertEqual(program["hparam_matrix"], matrix)
        self.assertEqual(program["profile_reference_case"], "hparam-ri-lr1e-6")

    def test_invalid_matrix_rejected_before_any_directory_is_created(self):
        valid = copy.deepcopy(MODULE.HPARAM_MATRIX["R"])
        invalid = [
            {},
            [],
            {"../escape": valid},
            {"R": valid, "r": valid},
            {"R": {}},
            {"R": {**valid, "max_steps": 500}},
            {"R": {**valid, "retain_coefficient": True}},
            {"R": {**valid, "inner_learning_rate": float("nan")}},
            {"R": {**valid, "outer_learning_rate": -1e-5}},
            {"R": {**valid, "retain_coefficient": 2.5}},
            {"R": valid, "duplicate": valid},
        ]
        for matrix in invalid:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                self.prepare(matrix)
            self.assertFalse(self.run.exists())

    def test_frozen_config_drift_rejected_before_preparation(self):
        self.defense["custom_trainer_args"]["attack_epochs"] = 4
        self.defense_path.write_text(yaml.safe_dump(self.defense))
        with self.assertRaisesRegex(ValueError, "inner attack"):
            self.prepare()
        self.assertFalse(self.run.exists())

    def test_baseline_order_mismatch_rejected_before_preparation(self):
        (self.compute / "probes.json").write_text(
            json.dumps({**self.probes, "expected_nonbootstrap_order_sha256": "wrong"})
        )
        with self.assertRaisesRegex(ValueError, "attack order"):
            self.prepare()
        self.assertFalse(self.run.exists())

    def test_cli_loads_matrix_from_a_saved_json_file(self):
        matrix_path = self.root / "reviewed-matrix.json"
        matrix = {"I": MODULE.HPARAM_MATRIX["I"]}
        matrix_path.write_text(json.dumps(matrix))
        argv = ["tar_compute.py", "prepare-hparam-study", str(self.run)]
        for name, path in (
            ("example", EXAMPLE),
            ("original", self.original),
            ("baseline", self.baseline),
            ("compute", self.compute),
            ("matrix", matrix_path),
        ):
            argv.extend((f"--{name}", str(path)))
        with (
            patch("sys.argv", argv),
            patch.object(MODULE, "prepare_hparam_study") as prepare,
        ):
            MODULE.main()
        prepare.assert_called_once_with(
            self.run, EXAMPLE, self.original, self.baseline, self.compute, matrix
        )
        self.assertFalse(self.run.exists())


class HparamInputOrderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    @staticmethod
    def write_records(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(map(json.dumps, records)) + "\n")

    def validation_fixture(self, name, role, kind="fixed-sampler-hparam-study"):
        run = self.root / name
        folder = run / "cases" / role
        maximum = 125 if role == "defense" else 1000
        model = folder / "models" / role
        checkpoint = model / f"checkpoint-{maximum}"
        checkpoint.mkdir(parents=True)
        case = {
            "name": name,
            "role": role,
            "max_steps": maximum,
            "save_steps": [maximum],
            "state_steps": [maximum],
            "inner_steps_per_update": 16 if role == "defense" else None,
            "gaussian_sigma": None,
        }
        (folder / "case.json").write_text(json.dumps(case))
        for filename in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / filename).touch()
        state = {
            "global_step": maximum,
            "log_history": (
                [{"inner_step_total": step} for step in range(maximum * 16)]
                if role == "defense"
                else []
            ),
        }
        (checkpoint / "trainer_state.json").write_text(json.dumps(state))
        program = {
            "kind": kind,
            "baseline_dynamics": str(run / "frozen-baseline"),
            "compute_study": str(run / "frozen-compute"),
        }
        (run / "study.json").write_text(json.dumps(program))
        reference = (
            Path(program["baseline_dynamics"]) / "models/defense/observed-batches.jsonl"
            if role == "defense"
            else Path(program["compute_study"])
            / "cases/baseline-lr1e-6/models/attack/observed-batches.jsonl"
        )
        records = []
        for step in range(1, maximum + 1):
            rows = 24 if role == "defense" else 1
            batch = {
                "input_ids": [[step, row, 9] for row in range(rows)],
                "attention_mask": [[1, 1, 0] for _ in range(rows)],
                "labels": [[step, row, -100] for row in range(rows)],
            }
            if role == "defense":
                batch["metadata_ids"] = [0] * 4 + [1] * 4 + [2] * 16
            records.append(
                {
                    "step": step,
                    "batch": batch,
                    "batch_sha256": "same-unverified-stored-hash",
                    "learning_rate_used": 1e-6,
                    "loss_before_update": 1.0,
                    "loss_after_update": 0.5,
                    "source_rows": [step],
                    "elapsed_seconds": float(step),
                }
            )
        self.write_records(reference, records)
        actual = copy.deepcopy(records)
        for row in actual:
            row.update(
                learning_rate_used=1e-5,
                loss_before_update=5.0,
                loss_after_update=3.0,
                elapsed_seconds=row["step"] * 2.0,
            )
            row["batch"] = dict(reversed(list(row["batch"].items())))
        observed = model / "observed-batches.jsonl"
        self.write_records(observed, actual)
        return folder, observed, reference, actual

    def test_exact_batches_pass_for_both_roles_despite_changed_training_metrics(self):
        for role in ("defense", "attack"):
            with self.subTest(role=role):
                folder, _, reference, _ = self.validation_fixture(role, role)
                MODULE.validate_case(folder)
                validation = json.loads(
                    (folder / "training-validation.json").read_text()
                )
                self.assertEqual(validation["state"], "passed")
                proof = validation["input_order_validation"]
                self.assertEqual(proof["reference_observations"], str(reference))
                self.assertEqual(proof["batches"], 125 if role == "defense" else 1000)
                self.assertEqual(proof["state"], "passed")
                self.assertEqual(len(proof["canonical_step_batch_sha256"]), 64)

    def test_changed_tensors_reject_even_when_stored_hashes_are_identical(self):
        for role in ("defense", "attack"):
            for field in ("input_ids", "attention_mask", "labels", "metadata_ids"):
                if role == "attack" and field == "metadata_ids":
                    continue
                with self.subTest(role=role, field=field):
                    folder, observed, _, actual = self.validation_fixture(
                        f"{role}-{field}", role
                    )
                    if field == "metadata_ids":
                        actual[0]["batch"][field][0] = 2
                    else:
                        actual[0]["batch"][field][0][0] += 1
                    self.write_records(observed, actual)
                    with self.assertRaisesRegex(ValueError, "batch/order differs"):
                        MODULE.validate_case(folder)
                    self.assertFalse((folder / "training-validation.json").exists())

    def test_reordered_batches_reject_with_sequential_step_numbers(self):
        for role in ("defense", "attack"):
            with self.subTest(role=role):
                folder, observed, _, actual = self.validation_fixture(role, role)
                actual[0]["batch"], actual[1]["batch"] = (
                    actual[1]["batch"],
                    actual[0]["batch"],
                )
                self.write_records(observed, actual)
                with self.assertRaisesRegex(ValueError, "batch/order differs"):
                    MODULE.validate_case(folder)
                self.assertFalse((folder / "training-validation.json").exists())

    def test_incomplete_frozen_controls_reject_before_publishing_pass(self):
        for role in ("defense", "attack"):
            with self.subTest(role=role):
                folder, _, reference, actual = self.validation_fixture(role, role)
                self.write_records(reference, actual[:-1])
                with self.assertRaisesRegex(ValueError, "complete matched horizons"):
                    MODULE.validate_case(folder)
                self.assertFalse((folder / "training-validation.json").exists())

    def test_nonhparam_and_standalone_legacy_validation_do_not_require_batches(self):
        for kind in ("supplied-checkpoint-comparison", None):
            with self.subTest(kind=kind):
                folder, observed, reference, actual = self.validation_fixture(
                    str(kind), "defense", kind
                )
                reference.unlink()
                for row in actual:
                    del row["batch"]
                self.write_records(observed, actual)
                if kind is None:
                    (folder.parents[1] / "study.json").unlink()
                MODULE.validate_case(folder)
                validation = json.loads(
                    (folder / "training-validation.json").read_text()
                )
                self.assertEqual(validation["state"], "passed")
                self.assertNotIn("input_order_validation", validation)


if __name__ == "__main__":
    unittest.main()
