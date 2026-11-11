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

    def test_matched_radius_random_uses_the_same_gaussian_pair_without_sign_changes(
        self,
    ):
        specs = (("a", (4, 3)), ("b", (7,)))
        pair = GEOMETRY.random_basis(specs, 42)
        original = tuple(
            {name: value.clone() for name, value in axis.items()} for axis in pair
        )
        radius = 0.753
        scaled, first_gaussian = GEOMETRY.matched_radius_random_basis(pair, radius)
        for index in range(2):
            self.assertAlmostEqual(
                math.sqrt(GEOMETRY.dot(scaled[index], scaled[index])), radius, places=6
            )
            for name, value in scaled[index].items():
                torch.testing.assert_close(
                    value, original[index][name] * radius, rtol=0, atol=0
                )
                self.assertTrue(torch.equal(value.sign(), original[index][name].sign()))
        for name, value in first_gaussian.items():
            torch.testing.assert_close(value, original[0][name], rtol=0, atol=0)
        self.assertAlmostEqual(GEOMETRY.dot(*scaled), 0, places=7)

    def test_helper_loader_and_provenance_use_the_executing_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            original = run / "source/tar_cleanroom.py"
            original.parent.mkdir()
            original.write_text("VERSION = 'original'\n")
            profile = run / "profile"
            profile.mkdir()
            actual = profile / "tar_cleanroom.py"
            actual.write_text(
                "VERSION = 'executing'\nEXTENSIONS_FILE = 'case-extensions.json'\n"
            )
            (run / "case-extensions.json").write_text('{"selected":"fresh-retain"}')
            with mock.patch.object(
                GEOMETRY, "__file__", str(profile / "tar_cleanroom_geometry.py")
            ):
                helper = GEOMETRY.load_cleanroom(run)
            self.assertEqual(helper.VERSION, "executing")
            provenance = GEOMETRY.helper_source_provenance(run, helper)
            self.assertEqual(
                provenance["executing_cleanroom_source"], str(actual.resolve())
            )
            self.assertNotEqual(
                provenance["cleanroom_source_sha256"],
                provenance["original_cleanroom_source_sha256"],
            )
            self.assertIn("case_extensions_sha256", provenance)

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

    def test_device_cache_matches_streaming_bits_and_ce_without_repeated_reads(self):
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(3, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1)
            )
            .float()
            .eval()
        )
        specs = tuple(
            (name, tuple(value.shape)) for name, value in model.named_parameters()
        )
        axes = GEOMETRY.random_basis(specs, 42)
        coordinates = [(0.0, 0.0), (1.0, 0.0), (-125.0, 125.0), (-0.375, 0.625)]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file(model.state_dict(), folder / "model.safetensors")
            with GEOMETRY.CheckpointReader(folder) as anchor:
                cache = GEOMETRY.DevicePointCache(model, anchor, axes)
                initial_anchor = {
                    name: value.clone() for name, value in cache.anchor.items()
                }
                for xy in coordinates:
                    GEOMETRY.set_point(model, anchor, axes, xy)
                    expected = {
                        name: value.detach().clone()
                        for name, value in model.named_parameters()
                    }
                    with mock.patch.object(anchor, "get", wraps=anchor.get) as reads:
                        cache.set(xy)
                        self.assertEqual(reads.call_count, 0)
                    for name, value in model.named_parameters():
                        torch.testing.assert_close(
                            value, expected[name], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            cache.anchor[name], initial_anchor[name], rtol=0, atol=0
                        )
                inputs = torch.tensor([[0.3, -0.2, 0.7]], dtype=torch.float32)

                def evaluate():
                    return {"probe": {"ce": float(model(inputs).square().mean())}}

                records = GEOMETRY.validate_point_cache(
                    cache, anchor, axes, coordinates, evaluate
                )
                for record in records:
                    self.assertEqual(record["parameter_max_abs_difference"], 0)
                    self.assertEqual(record["perturbation_relative_l2_difference"], 0)
                    self.assertEqual(record["ce_absolute_gaps"], {"probe": 0})
                self.assertEqual(
                    cache.resident_bytes,
                    3
                    * sum(
                        value.numel() * value.element_size()
                        for value in model.parameters()
                    ),
                )
                cache.close()
                self.assertEqual(cache.anchor, {})
                self.assertTrue(all(not axis for axis in cache.axes))

    def test_device_cache_corruption_fails_before_a_grid(self):
        model = torch.nn.Linear(3, 1, bias=False).float()
        axes = GEOMETRY.random_basis((("weight", (1, 3)),), 42)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file(model.state_dict(), folder / "model.safetensors")
            with GEOMETRY.CheckpointReader(folder) as anchor:
                cache = GEOMETRY.DevicePointCache(model, anchor, axes)
                cache.axes[0]["weight"][0, 0] += 0.01
                with self.assertRaisesRegex(RuntimeError, "exact FP32 parity"):
                    GEOMETRY.validate_point_cache(cache, anchor, axes, [(1.0, 0.0)])
                cache.close()

    @unittest.skipUnless(
        os.environ.get("TAR_CLEANROOM_TEST_POINT_CACHE_GPU") == "1"
        and torch.cuda.device_count() >= 2,
        "requires explicit TAR_CLEANROOM_TEST_POINT_CACHE_GPU=1 and two GPUs through launcher",
    )
    def test_device_cache_exact_parity_on_two_gpus(self):
        model = torch.nn.Module()
        model.register_parameter(
            "a", torch.nn.Parameter(torch.tensor([0.2, -0.6], device="cuda:0"))
        )
        model.register_parameter(
            "b", torch.nn.Parameter(torch.tensor([0.7, 0.3, -0.1], device="cuda:1"))
        )
        specs = tuple(
            (name, tuple(value.shape)) for name, value in model.named_parameters()
        )
        axes = GEOMETRY.random_basis(specs, 42)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_file(
                {
                    name: value.detach().cpu()
                    for name, value in model.named_parameters()
                },
                folder / "model.safetensors",
            )
            with GEOMETRY.CheckpointReader(folder) as anchor:
                cache = GEOMETRY.DevicePointCache(model, anchor, axes)
                records = GEOMETRY.validate_point_cache(
                    cache, anchor, axes, [(0, 0), (1, 0), (-125, 125)]
                )
                self.assertTrue(
                    all(
                        record["parameter_max_abs_difference"] == 0
                        and record["perturbation_relative_l2_difference"] == 0
                        for record in records
                    )
                )
                print(json.dumps({"two_gpu_cache_validation": records}))
                cache.close()

    def test_three_plane_pipeline_writes_native_probe_and_projection_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "source").mkdir()
            (run / "source/tar_cleanroom.py").write_text("# frozen test source\n")
            profile_helper = run / "profile/tar_cleanroom.py"
            profile_helper.parent.mkdir()
            profile_helper.write_text("# actual executing helper\n")
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
                __file__=str(profile_helper),
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
                self.assertEqual(plane["point_construction"]["mode"], "device cache")
                self.assertEqual(len(plane["point_construction"]["validation"]), 3)
                self.assertTrue(
                    all(
                        check["parameter_max_abs_difference"] == 0
                        for check in plane["point_construction"]["validation"]
                    )
                )
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

            # Adding the fixed reference to the control needs no model load or
            # forwards: it is literally the previously evaluated fitted plane.
            original_bytes = (folder / "geometry.json").read_bytes()
            original_planes = result["planes"]
            args.add_reference = True
            with (
                mock.patch.object(GEOMETRY, "load_cleanroom", return_value=clean),
                mock.patch.object(
                    clean,
                    "runtime",
                    side_effect=AssertionError("Control reuse must not load a model"),
                ),
            ):
                GEOMETRY.run_geometry(args)
            control = json.loads((folder / "geometry.json").read_text())
            self.assertEqual(
                list(control["planes"]),
                [name for name in GEOMETRY.PLANE_ORDER if name != "random-step"],
            )
            for name, plane in original_planes.items():
                self.assertEqual(control["planes"][name], plane)
            reference = control["planes"]["reference"]
            self.assertEqual(reference["grid"], control["planes"]["fit"]["grid"])
            self.assertEqual(
                reference["projections"], control["planes"]["fit"]["projections"]
            )
            archive = Path(control["augmentations"][-1]["previous_json"])
            self.assertEqual(archive.read_bytes(), original_bytes)
            # Adding metadata cannot change the fixed-axis identity.
            self.assertEqual(
                reference["reference_definition"]["identity_sha256"],
                GEOMETRY.reference_definition(control, folder / "geometry.json")[
                    "identity_sha256"
                ],
            )

            # Generic augmentation skips the existing reference and computes
            # only the missing matched-radius Gaussian grid.
            previous_control = control
            args.add_reference, args.add_planes = False, ["reference", "random-step"]
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
            control = json.loads((folder / "geometry.json").read_text())
            for name, plane in previous_control["planes"].items():
                self.assertEqual(control["planes"][name], plane)
            self.assertEqual(
                control["augmentations"][-1]["added_planes"], ["random-step"]
            )
            self.assertEqual(
                control["planes"]["random-step"]["scale_l2"],
                control["planes"]["first"]["scale_l2"],
            )
            self.assertEqual(
                control["executing_cleanroom_source"], str(profile_helper.resolve())
            )
            self.assertNotEqual(
                control["cleanroom_source_sha256"],
                control["original_cleanroom_source_sha256"],
            )

            # A fresh five-plane control run shares its reference/fit numbers.
            saved_control = (folder / "geometry.json").read_bytes()
            (folder / "geometry.json").unlink()
            args.add_reference, args.planes = False, list(GEOMETRY.PLANE_ORDER)
            args.add_planes = None
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
            fresh = json.loads((folder / "geometry.json").read_text())
            self.assertEqual(
                fresh["planes"]["reference"]["grid"], fresh["planes"]["fit"]["grid"]
            )
            self.assertEqual(
                fresh["planes"]["reference"]["reference_definition"]["identity_sha256"],
                reference["reference_definition"]["identity_sha256"],
            )
            self.assertEqual(
                fresh["planes"]["random-step"]["scale_l2"],
                fresh["planes"]["first"]["scale_l2"],
            )
            (folder / "geometry.json").write_bytes(saved_control)

            # A different defender gets the same physical reference axes/scale,
            # while its own first/fitted planes remain numerically untouched.
            study["cases"]["defender"] = {}
            center_folder = run / "cases/defender/defense-125"
            center_folder.mkdir(parents=True)
            center = base + torch.tensor([[0.2, -0.1, 0.4]])
            save_file({"weight": center}, center_folder / "model.safetensors")
            for step in GEOMETRY.REFERENCE_STEPS:
                checkpoint = (
                    run / "cases/defender/attacks/full-same" / f"checkpoint-{step}"
                )
                checkpoint.mkdir(parents=True)
                t = step / 64
                delta = torch.tensor([[-0.03 * t, 0.01 * t, 0.05 * t * t]])
                save_file({"weight": center + delta}, checkpoint / "model.safetensors")
            args.case, args.planes, args.add_reference = (
                "defender",
                ["first", "fit"],
                False,
            )
            args.point_cache = "streaming"
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
            case_path = run / "geometry/defender/geometry.json"
            previous = json.loads(case_path.read_text())
            self.assertTrue(
                all(
                    plane["point_construction"]["mode"] == "streaming"
                    for plane in previous["planes"].values()
                )
            )
            previous_bytes = case_path.read_bytes()
            args.add_reference = False
            args.add_planes = ["reference", "random-step"]
            args.point_cache = "device"
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
            updated = json.loads(case_path.read_text())
            for name, plane in previous["planes"].items():
                self.assertEqual(updated["planes"][name], plane)
            self.assertEqual(
                Path(updated["augmentations"][-1]["previous_json"]).read_bytes(),
                previous_bytes,
            )
            case_reference = updated["planes"]["reference"]
            self.assertEqual(
                updated["planes"]["random-step"]["scale_l2"],
                updated["planes"]["first"]["scale_l2"],
            )
            self.assertEqual(
                case_reference["reference_definition"]["identity_sha256"],
                reference["reference_definition"]["identity_sha256"],
            )
            self.assertEqual(case_reference["basis_gram"], reference["basis_gram"])
            self.assertNotEqual(case_reference["grid"], reference["grid"])
            self.assertLess(
                case_reference["projections"][-1]["displacement_energy_capture"], 0.999
            )
            torch.testing.assert_close(model.weight, center, rtol=0, atol=0)
            changed = (
                run / "cases/control/attacks/full-same/checkpoint-1/model.safetensors"
            )
            save_file({"weight": base}, changed)
            with self.assertRaisesRegex(ValueError, "reference checkpoint changed"):
                GEOMETRY.control_reference_axes(
                    run, reference["reference_definition"], (("weight", (1, 3)),)
                )

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
            self.assertTrue((folder / "slices-3d.png").exists())
            self.assertTrue((folder / "slices-3d.pdf").exists())

    def test_full_new_routes_native_checkpoints_three_probes_and_augmentation_separately(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary).resolve()
            (run / "source").mkdir()
            helper_path = run / "source/tar_cleanroom.py"
            helper_path.write_text("# executing fixture helper\n")
            (run / "case-extensions.json").write_text('{"fixture":"extension"}')
            case = "defender"
            center = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float32)
            base_folder = run / "cases" / case / "defense-125"
            base_folder.mkdir(parents=True)
            save_file({"weight": center}, base_folder / "model.safetensors")
            study = {
                "anchor": str(base_folder),
                "cases": {case: {}},
                "attack_variants": list(GEOMETRY.ATTACK_VARIANTS),
            }
            (run / "study.json").write_text(json.dumps(study))
            for variant in ("full-same", "full-new"):
                for step in GEOMETRY.REFERENCE_STEPS:
                    t = step / 64
                    delta = (
                        torch.tensor([[0.1 * t, 0.02 * t * t, 0]])
                        if variant == "full-same"
                        else torch.tensor([[0, 0.3 * t, -0.1 * t * t]])
                    )
                    path = (
                        run
                        / "cases"
                        / case
                        / "attacks"
                        / variant
                        / f"checkpoint-{step}"
                    )
                    path.mkdir(parents=True)
                    save_file({"weight": center + delta}, path / "model.safetensors")
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
                    ("forget_new_attack", 200, 8),
                )
            }
            data["content_sha256"] = "variant-fixture"
            model = torch.nn.Linear(3, 1, bias=False).float()
            clean = SimpleNamespace(
                __file__=str(helper_path),
                load_study=lambda _: study,
                prepare_data=lambda *_: data,
                runtime=mock.Mock(return_value=(model, None)),
                digest=lambda value: str(len(json.dumps(value))),
                evaluate=lambda m, rows: {
                    "ce": float(m.weight.detach().square().sum())
                    + rows[0]["source_row"] / 100,
                    "entropy": 0.5,
                },
                dump=lambda path, value: Path(path).write_text(
                    json.dumps(value, allow_nan=False)
                ),
            )
            canonical = GEOMETRY.geometry_folder(run, case)
            canonical.mkdir(parents=True)
            sentinels = {
                "geometry.json": b'{"canonical":"unchanged"}',
                "geometry-progress.json": b'{"progress":"unchanged"}',
            }
            for name, content in sentinels.items():
                (canonical / name).write_bytes(content)
            args = SimpleNamespace(
                run=run,
                case=case,
                attack_variant="full-new",
                planes=["first", "fit"],
                trajectory_steps=list(GEOMETRY.REFERENCE_STEPS),
                grid_size=3,
                seed=42,
                ce_limits=None,
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
            folder = canonical / "attack-variants/full-new"
            result = json.loads((folder / "geometry.json").read_text())
            self.assertEqual(result["attack_variant"], "full-new")
            self.assertEqual(
                list(result["probes"]),
                ["forget_train", "forget_heldout_subset", "forget_new_attack"],
            )
            self.assertEqual(len(result["probes"]["forget_new_attack"]), 8)
            self.assertEqual(
                result["probes"]["forget_new_attack"][0]["source_row"], 200
            )
            selected = run / "cases" / case / "attacks/full-new/checkpoint-1"
            self.assertEqual(
                result["checkpoint_files"]["1"], GEOMETRY.checkpoint_identity(selected)
            )
            with GEOMETRY.CheckpointReader(selected) as reader:
                expected_radius = float(
                    torch.linalg.vector_norm((reader.get("weight") - center).double())
                )
            self.assertAlmostEqual(
                result["planes"]["first"]["scale_l2"], expected_radius, places=12
            )
            self.assertIn("forget_new_attack", result["planes"]["fit"]["grid"][0])
            clean.runtime.assert_called_with(study, base_folder)
            plotted = []
            with (
                mock.patch.object(GEOMETRY, "load_cleanroom", return_value=clean),
                mock.patch.object(
                    GEOMETRY,
                    "plot",
                    side_effect=lambda value, target, limits: plotted.append(
                        (value["attack_variant"], target)
                    )
                    or {"ce_min": 0, "ce_max": 10},
                ),
            ):
                GEOMETRY.plot_saved(args)
            self.assertEqual(plotted, [("full-new", folder)])
            previous = (folder / "geometry.json").read_bytes()
            args.add_planes = ["random-step"]
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
            updated = json.loads((folder / "geometry.json").read_text())
            self.assertEqual(
                Path(updated["augmentations"][-1]["previous_json"]).read_bytes(),
                previous,
            )
            self.assertEqual(
                updated["augmentations"][-1]["original_cleanroom_source"],
                str(helper_path),
            )
            self.assertIn("case_extensions_sha256", updated["augmentations"][-1])
            for name, content in sentinels.items():
                self.assertEqual((canonical / name).read_bytes(), content)
            args.case, args.add_planes, args.planes = "control", None, ["reference"]
            clean.runtime.reset_mock()
            with (
                mock.patch.object(GEOMETRY, "load_cleanroom", return_value=clean),
                self.assertRaisesRegex(ValueError, "control reference is full-same"),
            ):
                GEOMETRY.run_geometry(args)
            clean.runtime.assert_not_called()

    @unittest.skipUnless(
        os.environ.get("TAR_CLEANROOM_TEST_PLOTS") == "1"
        and importlib.util.find_spec("matplotlib"),
        "requires explicit TAR_CLEANROOM_TEST_PLOTS=1 and existing plotting environment",
    )
    def test_five_plane_surfaces_share_heatmap_limits_and_sections_use_physical_radius(
        self,
    ):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            result = {"case": "fixture", "planes": {}}
            for column, name in enumerate(GEOMETRY.PLANE_ORDER):
                extent = 125 if name == "random" else 1
                coordinates = [-extent, 0, extent]
                result["planes"][name] = {
                    "coordinates": coordinates,
                    "axis_labels": ["Axis 1", "Axis 2"],
                    "scale_l2": 0.75,
                    "grid": [
                        {
                            "x": x,
                            "y": y,
                            "forget_train": {
                                "ce": column * 0.4
                                + (x / extent) ** 2
                                + 2 * (y / extent) ** 2
                            },
                            "forget_heldout_subset": {
                                "ce": column * 0.4
                                + (x / extent) ** 2
                                + 2 * (y / extent) ** 2
                                + 0.1
                            },
                        }
                        for y in coordinates
                        for x in coordinates
                    ],
                }
            figures = {}
            original_metadata = json.dumps(result, sort_keys=True)
            original_save = Figure.savefig

            def save(figure, filename, **kwargs):
                figures[Path(filename).name] = figure
                if str(filename).endswith(".png"):
                    kwargs["dpi"] = 50
                return original_save(figure, filename, **kwargs)

            with (
                mock.patch.object(Figure, "savefig", new=save),
                mock.patch.dict(os.environ, {"MPLCONFIGDIR": str(folder / "cache")}),
            ):
                returned = GEOMETRY.plot(result, folder, ce_limits=[0, 10])
            self.assertEqual(returned, {"ce_min": 0, "ce_max": 10})
            self.assertEqual(json.dumps(result, sort_keys=True), original_metadata)
            for stem in ("slices", "slices-3d", "matched-radius-sections"):
                for suffix in ("png", "pdf"):
                    self.assertTrue((folder / f"{stem}.{suffix}").is_file())
            heatmap = figures["slices.png"]
            surfaces = figures["slices-3d.png"]
            self.assertEqual(
                len(surfaces.axes), 11
            )  # Ten panels and one shared colorbar.
            for index, axis in enumerate(surfaces.axes[:10]):
                human_planes = (
                    "Broad random",
                    "Random at first-update radius",
                    "Fixed control attack plane",
                    "First-update plane",
                    "Fitted attack plane",
                )
                human_cohorts = (
                    "Forget training examples",
                    "Held-out forget examples (8)",
                )
                self.assertIn(human_planes[index // 2], axis.get_title())
                self.assertIn(human_cohorts[index % 2], axis.get_title())
                self.assertEqual(axis.get_zlim(), (0, 10))
                self.assertEqual((axis.elev, axis.azim), (30, -135))
                self.assertFalse(axis.computed_zorder)
                self.assertEqual(axis.collections[0].get_zorder(), 0)
                self.assertGreater(
                    axis.collections[2].get_zorder(), axis.collections[0].get_zorder()
                )
                self.assertTrue(
                    (axis.collections[2].get_facecolors()[0, :3] == [1, 1, 1]).all()
                )
                self.assertTrue(
                    (axis.collections[2].get_edgecolors()[0, :3] == [0, 0, 0]).all()
                )
                name = GEOMETRY.PLANE_ORDER[index // 2]
                expected_labels = (
                    ("Direction 1 (L2)", "Direction 2 (L2)")
                    if name == "random"
                    else (
                        ("First update\n(multiples)", "Orthogonal random\n(multiples)")
                        if name == "first"
                        else ("Direction 1 (radii)", "Direction 2 (radii)")
                    )
                )
                self.assertEqual(
                    (axis.get_xlabel(), axis.get_ylabel()), expected_labels
                )
                self.assertEqual(
                    (
                        heatmap.axes[index].get_xlabel(),
                        heatmap.axes[index].get_ylabel(),
                    ),
                    ("Axis 1", "Axis 2"),
                )
                self.assertEqual(
                    (axis.collections[0].norm.vmin, axis.collections[0].norm.vmax),
                    (0, 10),
                )
                self.assertEqual(
                    (
                        heatmap.axes[index].collections[0].norm.vmin,
                        heatmap.axes[index].collections[0].norm.vmax,
                    ),
                    (0, 10),
                )
                # Surface, measured nodes, anchor, and first marker only on first panels.
                self.assertEqual(
                    len(axis.collections),
                    4 if index // 2 == list(GEOMETRY.PLANE_ORDER).index("first") else 3,
                )
                self.assertFalse(axis.lines)
            canvas = FigureCanvasAgg(surfaces)
            canvas.draw()
            renderer = canvas.get_renderer()
            self.assertGreaterEqual(surfaces.get_figheight(), 14)
            for column in range(5):
                top, bottom = surfaces.axes[2 * column : 2 * column + 2]
                top_label_minimum = min(
                    label.get_window_extent(renderer).y0
                    for label in (top.xaxis.label, top.yaxis.label)
                )
                self.assertGreater(
                    top_label_minimum, bottom.title.get_window_extent(renderer).y1
                )
            for axis in surfaces.axes[:10]:
                for label in (axis.xaxis.label, axis.yaxis.label):
                    bounds = label.get_window_extent(renderer)
                    self.assertGreaterEqual(bounds.y0, surfaces.bbox.y0)
                    self.assertLessEqual(bounds.y1, surfaces.bbox.y1)
                    self.assertGreaterEqual(bounds.x0, surfaces.bbox.x0)
                    self.assertLessEqual(bounds.x1, surfaces.bbox.x1)
            for axis in figures["matched-radius-sections.png"].axes:
                self.assertEqual(axis.get_ylim(), (0, 10))
                self.assertEqual(len(axis.lines), 2)
                for line in axis.lines:
                    self.assertEqual(list(line.get_xdata()), [-0.75, 0, 0.75])
                    self.assertEqual(len(line.get_ydata()), 3)
            # The automatic scale also comes from all measured planes/groups.
            with mock.patch.object(
                Figure,
                "savefig",
                new=lambda figure, filename, **kwargs: figures.__setitem__(
                    Path(filename).name, figure
                ),
            ):
                automatic = GEOMETRY.plot(result, folder)
            expected_min = min(
                point[group]["ce"]
                for plane in result["planes"].values()
                for point in plane["grid"]
                for group in ("forget_train", "forget_heldout_subset")
            )
            expected_max = max(
                point[group]["ce"]
                for plane in result["planes"].values()
                for point in plane["grid"]
                for group in ("forget_train", "forget_heldout_subset")
            )
            self.assertEqual(
                automatic, {"ce_min": expected_min, "ce_max": expected_max}
            )
            self.assertTrue(
                all(
                    axis.get_zlim() == (expected_min, expected_max)
                    for axis in figures["slices-3d.png"].axes[:10]
                )
            )
            # The stronger new-data attacker gets a third row and contributes
            # to the same global CE range, without changing legacy fixtures.
            alternate = json.loads(json.dumps(result))
            alternate["attack_variant"] = "full-new"
            alternate["probes"] = {
                "forget_train": [],
                "forget_heldout_subset": [],
                "forget_new_attack": [],
            }
            alternate["planes"] = {
                name: alternate["planes"][name] for name in ("first", "fit")
            }
            for plane in alternate["planes"].values():
                for point in plane["grid"]:
                    point["forget_new_attack"] = {
                        "ce": point["forget_train"]["ce"] + 20
                    }
            with mock.patch.object(
                Figure,
                "savefig",
                new=lambda figure, filename, **kwargs: figures.__setitem__(
                    Path(filename).name, figure
                ),
            ):
                new_limits = GEOMETRY.plot(alternate, folder)
            self.assertGreater(new_limits["ce_max"], 20)
            self.assertEqual(len(figures["slices.png"].axes), 7)
            alternate_surface = figures["slices-3d.png"]
            self.assertEqual(len(alternate_surface.axes), 7)
            self.assertIn("full-new", alternate_surface._suptitle.get_text())
            self.assertIn(
                "Attacker forget examples (8; excluded from continuation)",
                alternate_surface.axes[2].get_title(),
            )
            self.assertIn(
                "Attacker forget examples (8; excluded from continuation)",
                alternate_surface.axes[5].get_title(),
            )
            self.assertTrue(
                all(
                    axis.get_zlim() == (new_limits["ce_min"], new_limits["ce_max"])
                    for axis in alternate_surface.axes[:6]
                )
            )
            alternate_canvas = FigureCanvasAgg(alternate_surface)
            alternate_canvas.draw()
            alternate_renderer = alternate_canvas.get_renderer()
            for column in range(2):
                panels = alternate_surface.axes[column * 3 : column * 3 + 3]
                for upper, lower in zip(panels, panels[1:]):
                    self.assertGreater(
                        min(
                            label.get_window_extent(alternate_renderer).y0
                            for label in (upper.xaxis.label, upper.yaxis.label)
                        ),
                        lower.title.get_window_extent(alternate_renderer).y1,
                    )
            for axis in alternate_surface.axes[:6]:
                for text in (axis.title, axis.xaxis.label, axis.yaxis.label):
                    bounds = text.get_window_extent(alternate_renderer)
                    self.assertGreaterEqual(bounds.x0, alternate_surface.bbox.x0)
                    self.assertLessEqual(bounds.x1, alternate_surface.bbox.x1)
                    self.assertGreaterEqual(bounds.y0, alternate_surface.bbox.y0)
                    self.assertLessEqual(bounds.y1, alternate_surface.bbox.y1)


if __name__ == "__main__":
    unittest.main()
