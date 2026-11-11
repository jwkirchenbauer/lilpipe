"""Small CPU tests for native-coordinate and streamed-trajectory geometry."""

import importlib.util
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import torch
    from safetensors.torch import save_file
except ImportError as error:
    raise unittest.SkipTest("requires torch and safetensors") from error

SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_cleanroom_geometry.py"
)
SPEC = importlib.util.spec_from_file_location("tar_cleanroom_geometry", SCRIPT)
GEOMETRY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GEOMETRY
SPEC.loader.exec_module(GEOMETRY)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_shared_gaussians_are_repeatable_and_globally_orthonormal(self):
        specs = (("a", (4, 3)), ("b", (7,)))
        first = GEOMETRY.random_basis(specs, 42)
        second = GEOMETRY.random_basis(specs, 42)
        for i, axis in enumerate(first):
            for name, value in axis.items():
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, second[i][name], rtol=0, atol=0)
            self.assertAlmostEqual(GEOMETRY.dot(axis, axis), 1.0, places=6)
        self.assertAlmostEqual(GEOMETRY.dot(first[0], first[1]), 0.0, places=6)

    def test_first_plane_contains_actual_first_update_with_equal_axis_norms(self):
        delta = {"a": torch.tensor([0.3, -0.7, 0.2], dtype=torch.float32)}
        original = delta["a"].clone()
        gaussian = GEOMETRY.random_basis((("a", (3,)),), 42)[0]
        axes, length = GEOMETRY.first_update_basis(delta, gaussian)
        torch.testing.assert_close(axes[0]["a"], original, rtol=0, atol=0)
        self.assertAlmostEqual(GEOMETRY.dot(*axes), 0.0, places=7)
        self.assertAlmostEqual(GEOMETRY.dot(axes[1], axes[1]), length**2, places=6)

    def test_streamed_shards_match_direct_svd_and_projection_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor_values = {"a": torch.tensor([0.5, -0.2]), "b": torch.tensor([0.7])}
            displacements = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.25]]
            )
            specs = (("a", (2,)), ("b", (1,)))

            def write(folder, values):
                folder.mkdir()
                save_file(
                    {"a": values["a"]}, folder / "model-00001-of-00002.safetensors"
                )
                save_file(
                    {"b": values["b"]}, folder / "model-00002-of-00002.safetensors"
                )
                (folder / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "a": "model-00001-of-00002.safetensors",
                                "b": "model-00002-of-00002.safetensors",
                            }
                        }
                    )
                )

            write(root / "anchor", anchor_values)
            for i, delta in enumerate(displacements):
                write(
                    root / str(i),
                    {
                        "a": anchor_values["a"] + delta[:2],
                        "b": anchor_values["b"] + delta[2:],
                    },
                )
            anchor = GEOMETRY.CheckpointReader(root / "anchor")
            readers = [GEOMETRY.CheckpointReader(root / str(i)) for i in range(4)]
            try:
                gram = GEOMETRY.trajectory_gram(readers, anchor, specs)
                actual_deltas = torch.stack(
                    [
                        torch.cat(
                            [reader.get(name) - anchor.get(name) for name, _ in specs]
                        )
                        for reader in readers
                    ]
                ).double()
                torch.testing.assert_close(
                    gram, actual_deltas @ actual_deltas.T, rtol=1e-12, atol=1e-12
                )
                axes, metadata = GEOMETRY.fitted_basis(readers, anchor, specs, gram)
                coordinates, captures, basis_gram = GEOMETRY.projection_coordinates(
                    readers, anchor, specs, axes
                )
                _, singular, direct_basis = torch.linalg.svd(
                    actual_deltas, full_matrices=False
                )
                flat_axes = torch.stack(
                    [torch.cat([axis[name] for name, _ in specs]) for axis in axes]
                ).double()
                projector = flat_axes.T @ torch.linalg.solve(
                    flat_axes @ flat_axes.T, flat_axes
                )
                expected_projector = direct_basis[:2].T @ direct_basis[:2]
                torch.testing.assert_close(
                    projector, expected_projector, rtol=1e-6, atol=1e-7
                )
                torch.testing.assert_close(
                    basis_gram,
                    torch.eye(2, dtype=torch.float64) * metadata["scale_l2"] ** 2,
                    rtol=3e-7,
                    atol=1e-7,
                )
                torch.testing.assert_close(
                    captures[:3],
                    torch.ones(3, dtype=torch.float64),
                    rtol=1e-6,
                    atol=1e-7,
                )
                self.assertLess(float(captures[3]), 1e-10)
                self.assertLessEqual(float(coordinates.abs().max()), 1.0 + 1e-7)
                self.assertAlmostEqual(
                    metadata["top_two_displacement_energy_fraction"],
                    float(singular[:2].square().sum() / singular.square().sum()),
                    places=10,
                )
            finally:
                anchor.close()
                for reader in readers:
                    reader.close()

    def test_native_point_resets_anchor_and_does_not_accumulate_roundoff(self):
        model = torch.nn.Linear(2, 1, bias=False).float()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = torch.tensor([[0.2, -0.6]], dtype=torch.float32)
            save_file({"weight": base}, folder / "model.safetensors")
            axes = (
                {"weight": torch.tensor([[0.3, 0.7]])},
                {"weight": torch.tensor([[0.8, -0.1]])},
            )
            with GEOMETRY.CheckpointReader(folder) as anchor:
                GEOMETRY.set_point(model, anchor, axes, (0.25, -0.5))
                expected = base + 0.25 * axes[0]["weight"] - 0.5 * axes[1]["weight"]
                torch.testing.assert_close(model.weight, expected, rtol=1e-7, atol=1e-8)
                original = model.weight.detach().clone()
                GEOMETRY.set_point(model, anchor, axes, (125.0, -125.0))
                GEOMETRY.set_point(model, anchor, axes, (0.25, -0.5))
                torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
                self.assertEqual(model.weight.dtype, torch.float32)

    def test_reader_rejects_half_precision_and_fit_rejects_rank_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file(
                {"p": torch.ones(3, dtype=torch.bfloat16)}, folder / "model.safetensors"
            )
            with GEOMETRY.CheckpointReader(folder) as reader:
                with self.assertRaisesRegex(ValueError, "Native FP32"):
                    reader.get("p")
            with GEOMETRY.CheckpointReader(folder, allow_anchor_cast=True) as reader:
                actual = reader.get("p")
                self.assertEqual(actual.dtype, torch.float32)
                torch.testing.assert_close(actual, torch.ones(3), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "rank"):
            GEOMETRY.fitted_coefficients(torch.ones(4, 4, dtype=torch.float64))

    def test_three_plane_pipeline_writes_native_probe_and_projection_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "source").mkdir()
            (run / "source/tar_cleanroom.py").write_text("# frozen test source\n")
            anchor_folder = run / "anchor"
            anchor_folder.mkdir()
            base = torch.tensor([[0.5, -0.2, 0.7]], dtype=torch.float32)
            save_file({"weight": base}, anchor_folder / "model.safetensors")
            study = {"anchor": str(anchor_folder), "cases": {}}
            (run / "study.json").write_text(json.dumps(study))
            for step in (1, 4, 16, 64):
                folder = run / "cases/control/attacks/full-same" / f"checkpoint-{step}"
                folder.mkdir(parents=True)
                t = step / 64
                delta = torch.tensor([[0.1 * t, 0.03 * t * t, -0.02 * t]])
                save_file({"weight": base + delta}, folder / "model.safetensors")
            data = {
                group: [
                    {
                        "source_row": start + i,
                        "source_text_sha256": str(start + i),
                        "input_ids": [i, i + 1],
                    }
                    for i in range(count)
                ]
                for group, start, count in (
                    ("forget_train", 0, 8),
                    ("forget_heldout", 100, 32),
                )
            }
            data["content_sha256"] = "test-data-hash"
            model = torch.nn.Linear(3, 1, bias=False).float()

            def dump(path, value):
                Path(path).write_text(json.dumps(value, allow_nan=False))

            clean = SimpleNamespace(
                load_study=lambda _: study,
                prepare_data=lambda *_: data,
                runtime=lambda *_: (model, None),
                evaluate=lambda m, _: {"ce": float(m.weight.detach().square().sum())},
                digest=lambda value: str(len(json.dumps(value))),
                dump=dump,
            )
            args = SimpleNamespace(
                run=run,
                case="control",
                planes=["random", "first", "fit"],
                trajectory_steps=[1, 4, 16, 64],
                grid_size=3,
                seed=42,
                ce_limits=[0, 40000],
            )
            with (
                mock.patch.object(GEOMETRY, "load_cleanroom", return_value=clean),
                mock.patch.object(
                    GEOMETRY,
                    "plot",
                    side_effect=ModuleNotFoundError("No matplotlib", name="matplotlib"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                GEOMETRY.run_geometry(args)
            folder = run / "geometry/control"
            result = json.loads((folder / "geometry.json").read_text())
            self.assertEqual(list(result["planes"]), args.planes)
            self.assertEqual(result["weight_dtype"], "float32")
            self.assertEqual(len(result["probes"]["forget_heldout_subset"]), 8)
            self.assertIn("Numerical grids complete", result["figure_state"])
            for name, plane in result["planes"].items():
                self.assertEqual(len(plane["grid"]), 9)
                self.assertEqual(len(plane["projections"]), 4)
                if name == "first":
                    first = plane["projections"][0]
                    self.assertAlmostEqual(
                        first["displacement_energy_capture"], 1.0, places=6
                    )
                    self.assertAlmostEqual(
                        first["ce_error_projected_minus_actual"]["forget_train"],
                        0.0,
                        places=6,
                    )
            torch.testing.assert_close(model.weight, base, rtol=0, atol=0)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "requires existing plotting environment"
    )
    def test_plot_only_uses_common_explicit_scale_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            folder = run / "geometry/control"
            folder.mkdir(parents=True)
            grid = [
                {
                    "x": x,
                    "y": y,
                    "forget_train": {"ce": x * x + y * y},
                    "forget_heldout_subset": {"ce": 1 + x * x + y * y},
                }
                for y in (-1, 0, 1)
                for x in (-1, 0, 1)
            ]
            result = {
                "case": "control",
                "planes": {
                    "first": {
                        "coordinates": [-1, 0, 1],
                        "axis_labels": ["x", "y"],
                        "grid": grid,
                    }
                },
            }
            path = folder / "geometry.json"
            path.write_text(json.dumps(result))
            clean = SimpleNamespace(
                dump=lambda target, value: Path(target).write_text(json.dumps(value))
            )
            args = SimpleNamespace(run=run, case="control", ce_limits=[0, 10])
            with (
                mock.patch.object(GEOMETRY, "load_cleanroom", return_value=clean),
                mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(run / "plot-cache")}),
            ):
                GEOMETRY.plot_saved(args)
            result = json.loads(path.read_text())
            self.assertEqual(result["figure_state"], "complete")
            self.assertEqual(result["figure_ce_scale"], {"ce_min": 0, "ce_max": 10})
            self.assertTrue((folder / "slices.png").exists())
            self.assertTrue((folder / "slices.pdf").exists())


if __name__ == "__main__":
    unittest.main()
