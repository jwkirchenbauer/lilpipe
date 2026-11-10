"""Check observation hooks and the geometry used for trajectory projections."""

import ast
import contextlib
import json
import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MISC = ROOT / "examples/tamper-resistance/scripts/misc"


class ObservationTests(unittest.TestCase):
    def test_post_update_measurement_preserves_weights_and_training_mode(self):
        path = MISC / "tar_dynamics_training.py"
        node = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "DynamicsCallback"
        )
        scope = {
            "TrainerCallback": object,
            "torch": torch,
            "math": math,
            "json": json,
            "Path": Path,
        }
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope
        )

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, input_ids):
                return SimpleNamespace(loss=(self.weight * input_ids).square().mean())

        model = Model().train()
        original = model.weight.detach().clone()
        trainer = SimpleNamespace(
            model=model,
            _measure_after=True,
            _dynamics_record={"step": 1, "loss_before_update": 5.0},
            _dynamics_batch={"input_ids": torch.ones(1, 2)},
            _prepare_inputs=lambda batch: batch,
            accelerator=SimpleNamespace(autocast=contextlib.nullcontext),
        )
        with tempfile.TemporaryDirectory() as directory:
            callback = scope["DynamicsCallback"](trainer, [1, 4])
            control = SimpleNamespace(should_save=False)
            callback.on_step_end(
                SimpleNamespace(output_dir=directory),
                SimpleNamespace(global_step=1),
                control,
            )
            record = json.loads(
                (Path(directory) / "observed-batches.jsonl").read_text()
            )
        self.assertEqual(record["loss_after_update"], 4.0)
        self.assertTrue(control.should_save)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.weight, original))
        self.assertIsNone(model.weight.grad)
        self.assertIsNone(trainer._dynamics_record)


class ProjectionTests(unittest.TestCase):
    def test_orthogonality_and_projection_residuals(self):
        spec = importlib.util.spec_from_file_location(
            "tar_dynamics", MISC / "tar_dynamics.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(12)
        displacements = rng.normal(size=(3, 8))
        random = rng.normal(size=(2, 8))
        vectors = np.concatenate([displacements, random])
        gram = vectors @ vectors.T
        coefficients, metadata = module.plane_coefficients(gram, 3)
        for plane in coefficients.values():
            basis = vectors.T @ plane
            np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
            coordinates = displacements @ basis
            residual = displacements - coordinates @ basis.T
            np.testing.assert_allclose(residual @ basis, 0, atol=1e-10)
            np.testing.assert_allclose(
                np.sum(displacements**2, axis=1),
                np.sum(coordinates**2, axis=1) + np.sum(residual**2, axis=1),
            )
        transverse = vectors.T @ coefficients["transverse"]
        np.testing.assert_allclose(displacements @ transverse, 0, atol=1e-10)
        self.assertEqual(metadata["attack_rank"], 3)
        self.assertGreater(metadata["attack_plane_energy_fraction"], 0.5)


class RecoveryTests(unittest.TestCase):
    def test_partial_surface_cache_preserves_coordinates_and_missing_fields(self):
        import pandas as pd

        path = MISC / "tar_dynamics.py"
        surfaces = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "surfaces"
        )
        helpers = [
            node
            for node in surfaces.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("cached", "save_rows")
        ]
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "surfaces").mkdir()
            scope = {"study": study, "pd": pd}
            exec(
                compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec"),
                scope,
            )
            rows = [
                {
                    "phase": "defense",
                    "step": 64,
                    "c0": 0.7856669940234822,
                    "probe_loss": 11.2,
                },
                {
                    "phase": "attack",
                    "step": 16,
                    "c0": -0.0982083742529353,
                    "probe_loss": 10.6,
                    "minibatch_loss": 10.5,
                },
            ]
            scope["save_rows"]("local-grids.csv", rows)
            recovered = scope["cached"]("local-grids.csv")
            self.assertEqual(recovered, rows)
            self.assertFalse(list((study / "surfaces").glob("*.tmp")))
            (study / "surfaces/empty.csv").write_text("\n")
            self.assertEqual(scope["cached"]("empty.csv"), [])


if __name__ == "__main__":
    unittest.main()
