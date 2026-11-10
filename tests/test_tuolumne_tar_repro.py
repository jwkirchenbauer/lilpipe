"""Check that the reproduction report rejects incomplete or conflicting evidence."""

import importlib.util
import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tuolumne_tar_repro.py"
)
SPEC = importlib.util.spec_from_file_location("tuolumne_tar_repro", SCRIPT)
REPRO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRO)


class ReproductionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_results(self, directory, name="results.json", score=0.5, limit=None):
        directory.mkdir(parents=True, exist_ok=True)
        results = {task: {"acc,none": score} for task in REPRO.TASKS}
        document = {"results": results, "config": {"limit": limit}}
        (directory / name).write_text(json.dumps(document))

    def test_complete_scores_convert_to_percent(self):
        self.write_results(self.root)
        self.assertEqual(REPRO.scores(self.root), dict.fromkeys(REPRO.TASKS, 50))

    def test_limited_evaluation_cannot_count_as_complete(self):
        self.write_results(self.root, limit=10)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            REPRO.scores(self.root)

    def test_conflicting_results_cannot_be_silently_combined(self):
        self.write_results(self.root, "results_a.json", score=0.5)
        self.write_results(self.root, "results_b.json", score=0.6)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            REPRO.scores(self.root)

    def test_base_mismatch_stops_before_training(self):
        self.write_results(self.root / "reference/z7b-base", score=0.5)
        self.write_results(self.root / "evals/base", score=0.52)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "before training"):
                REPRO.compare(self.root, self.root, base_only=True)

    def test_generated_attacks_preserve_collaborator_configs(self):
        import yaml

        example = SCRIPT.parents[2]
        (self.root / "configs").mkdir()
        inputs = {
            "model": "/cached/zephyr",
            "datasets": {
                name: {"revision": "test-revision"}
                for name in ("cais/wmdp-bio-forget-corpus", "Salesforce/wikitext")
            },
        }
        (self.root / "inputs.json").write_text(json.dumps(inputs))
        REPRO.materialize(self.root, example)
        operational_fields = (
            "base_model",
            "output_dir",
            "dataset_prepared_path",
            "wandb_project",
            "wandb_name",
        )
        for rate in REPRO.ATTACK_RATES:
            source = (
                example
                / "configs/stages-wmdp/2_relearn"
                / (REPRO.DEFENSE + f"--relearn-fft-lr{rate}-ms128")
                / "selected.yml"
            )
            expected = yaml.safe_load(source.read_text())
            actual = yaml.safe_load(
                (self.root / "configs" / f"attack-lr{rate}.yml").read_text()
            )
            self.assertEqual(actual["base_model"], str(self.root / "models/defense"))
            for key in operational_fields:
                actual.pop(key)
                expected.pop(key)
            for dataset in actual["datasets"]:
                self.assertEqual(dataset.pop("revision"), "test-revision")
            self.assertEqual(actual, expected)


class DeviceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        import torch

        path = SCRIPT.parents[2] / "src/trainers/tr_tar.py"
        function = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "log_device_assignment"
        )
        scope = {"torch": torch, "os": os, "json": json, "Counter": Counter}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            scope,
        )
        self.log_devices = scope["log_device_assignment"]
        self.model = torch.nn.Linear(2, 3)

    def test_reports_actual_parameter_and_batch_devices(self):
        import torch

        self.model.hf_device_map = {"": "cpu"}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.log_devices(self.model, "test", inputs={"input_ids": torch.ones(1, 2)})
        report = json.loads(output.getvalue().split("TAR device check: ")[1])
        self.assertEqual(report["parameter_counts_by_device"], {"cpu": 9})
        self.assertEqual(report["batch"]["input_ids"]["device"], "cpu")

    def test_rejects_a_layer_on_the_wrong_device(self):
        self.model.hf_device_map = {"": 0}
        with self.assertRaisesRegex(RuntimeError, "expected cuda:0"):
            self.log_devices(self.model, "test")


if __name__ == "__main__":
    unittest.main()
