"""Check observation hooks and the geometry used for trajectory projections."""

import ast
import contextlib
import json
import importlib.util
import math
import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
            "os": os,
            "time": time,
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
            _dynamics_started=time.perf_counter(),
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


class BasisTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "tar_dynamics", MISC / "tar_dynamics.py"
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_exact_fp32_bits_and_row_order_across_bounded_chunks(self):
        bits = np.array(
            [
                0x00000000,
                0x80000000,
                0x00000001,
                0x007FFFFF,
                0x00800000,
                0x3F800000,
                0x7F7FFFFF,
                0xBF800000,
                0x7F800000,
                0xFF800000,
                0x7FC01234,
                0xFFC05678,
                0x3F800001,
                0xC0000000,
            ],
            dtype=np.uint32,
        ).reshape(2, 7)
        source = bits.view(np.float32)
        with mock.patch.object(self.module.np, "array", wraps=np.array) as copy:
            basis = self.module._basis_to_device(source, "cpu", chunk_elements=3)
        self.assertEqual(basis.dtype, torch.float32)
        self.assertEqual(tuple(basis.shape), (2, 7))
        self.assertTrue(basis.is_contiguous())
        np.testing.assert_array_equal(basis.numpy().view(np.uint32), bits)
        self.assertEqual(
            [call.args[0].size for call in copy.call_args_list], [3, 3, 1, 3, 3, 1]
        )
        self.assertTrue(
            all(call.kwargs == {"copy": True} for call in copy.call_args_list)
        )
        expected = bits.copy()
        source.fill(0)
        np.testing.assert_array_equal(basis.numpy().view(np.uint32), expected)

    def test_read_only_mmap_remains_unchanged_and_result_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basis.npy"
            expected = np.arange(22, dtype=np.float32).reshape(2, 11)
            np.save(path, expected)
            original = path.read_bytes()
            mapped = np.load(path, mmap_mode="r")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                basis = self.module._basis_to_device(mapped, "cpu", chunk_elements=4)
            np.testing.assert_array_equal(basis.numpy(), expected)
            self.assertFalse(mapped.flags.writeable)
            self.assertFalse(np.shares_memory(mapped, basis.numpy()))
            basis.fill_(-1)
            np.testing.assert_array_equal(mapped, expected)
            self.assertEqual(path.read_bytes(), original)

    def test_rejects_unsupported_dtype_shape_and_chunk_size(self):
        for source in (
            np.zeros((2, 7), dtype=np.float64),
            np.zeros((2, 7), dtype=np.int32),
            np.zeros(7, dtype=np.float32),
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((2, 1, 7), dtype=np.float32),
        ):
            with self.subTest(dtype=source.dtype, shape=source.shape):
                with self.assertRaisesRegex(ValueError, "FP32.*shape"):
                    self.module._basis_to_device(source, "cpu")
        for chunk_elements in (0, -1, 1.5, True):
            with self.subTest(chunk_elements=chunk_elements):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.module._basis_to_device(
                        np.zeros((2, 7), dtype=np.float32),
                        "cpu",
                        chunk_elements=chunk_elements,
                    )


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
