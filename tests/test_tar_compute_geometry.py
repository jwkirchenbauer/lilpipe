"""Check shared and condition-fitted planes without loading a language model."""

import contextlib
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
SPEC = importlib.util.spec_from_file_location("tar_compute_geometry", SCRIPT)
DYNAMICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DYNAMICS)


class PlaneCoefficientTests(unittest.TestCase):
    def test_default_planes_preserve_orthogonality_and_projection_residuals(self):
        rng = np.random.default_rng(21)
        attack = rng.normal(size=(3, 12))
        vectors = np.concatenate([attack, rng.normal(size=(2, 12))])
        coefficients, details = DYNAMICS.plane_coefficients(vectors @ vectors.T, 3)
        self.assertEqual(list(coefficients), ["random", "attack", "transverse"])
        self.assertEqual(details["attack_rank"], 3)
        for coefficients_for_plane in coefficients.values():
            basis = vectors.T @ coefficients_for_plane
            np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
            coordinates = attack @ basis
            residual = attack - coordinates @ basis.T
            np.testing.assert_allclose(residual @ basis, 0, atol=1e-10)
            np.testing.assert_allclose(
                np.sum(attack**2, axis=1),
                np.sum(coordinates**2, axis=1) + np.sum(residual**2, axis=1),
            )
        transverse = vectors.T @ coefficients["transverse"]
        np.testing.assert_allclose(attack @ transverse, 0, atol=1e-10)

    def test_extra_reference_columns_and_shared_random_remain_exact(self):
        rng = np.random.default_rng(33)
        attack = rng.normal(size=(3, 20))
        directions = np.linalg.qr(rng.normal(size=(20, 6)))[0].T.astype(np.float32)
        # Stored FP32 directions can be close to orthonormal without being exact.
        directions[0] *= np.float32(1 + 1e-6)
        directions[3] *= np.float32(1 - 1e-6)
        vectors = np.concatenate([attack, directions])
        coefficients, _ = DYNAMICS.plane_coefficients(
            vectors @ vectors.T,
            3,
            ["reference", "second"],
            planes=["second", "random", "attack", "reference"],
            preserve_shared=True,
        )
        self.assertEqual(
            list(coefficients), ["second", "random", "attack", "reference"]
        )
        for name, start in (("random", 0), ("reference", 2), ("second", 4)):
            basis = vectors.T @ coefficients[name]
            np.testing.assert_array_equal(basis, directions[start : start + 2].T)
            small = basis.T @ basis
            np.testing.assert_allclose(small, np.eye(2), atol=1e-5)
            coordinates = (attack @ basis) @ np.linalg.inv(small)
            residual = attack - coordinates @ basis.T
            np.testing.assert_allclose(residual @ basis, 0, atol=1e-10)
            captured = np.einsum("ni,ij,nj->n", coordinates, small, coordinates)
            np.testing.assert_allclose(
                np.sum(attack**2, axis=1), captured + np.sum(residual**2, axis=1)
            )

    def test_rank_one_attack_keeps_its_direction_and_completes_a_plane(self):
        rng = np.random.default_rng(4)
        first = np.arange(1, 9, dtype=float)
        first /= np.linalg.norm(first)
        attack = np.outer([1.0, -2.0, 3.0], first)
        vectors = np.concatenate([attack, rng.normal(size=(2, 8))])
        gram = vectors @ vectors.T
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            DYNAMICS.plane_coefficients(gram, 3)
        coefficients, details = DYNAMICS.plane_coefficients(
            gram, 3, allow_degenerate=True
        )
        basis = vectors.T @ coefficients["attack"]
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
        self.assertAlmostEqual(first @ basis[:, 0], 1.0)
        np.testing.assert_allclose(attack - attack @ basis @ basis.T, 0, atol=1e-10)
        self.assertEqual(details["attack_rank"], 1)
        self.assertAlmostEqual(details["attack_plane_energy_fraction"], 1.0)

    def test_rank_zero_attack_has_a_deterministic_random_completion(self):
        rng = np.random.default_rng(9)
        vectors = np.concatenate([np.zeros((3, 8)), rng.normal(size=(2, 8))])
        gram = vectors @ vectors.T
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            DYNAMICS.plane_coefficients(gram, 3)
        coefficients, details = DYNAMICS.plane_coefficients(
            gram, 3, allow_degenerate=True
        )
        repeated, _ = DYNAMICS.plane_coefficients(gram, 3, allow_degenerate=True)
        self.assertEqual(details["attack_rank"], 0)
        self.assertEqual(details["attack_plane_energy_fraction"], 0.0)
        for name, plane in coefficients.items():
            basis = vectors.T @ plane
            np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-10)
            np.testing.assert_array_equal(plane, repeated[name])
        np.testing.assert_array_equal(coefficients["attack"], coefficients["random"])

    def test_shared_planes_require_orthonormality_and_matching_columns(self):
        gram = np.eye(7)
        gram[5, 5] = 2.0
        with self.assertRaises(AssertionError):
            DYNAMICS.plane_coefficients(gram, 3, ["reference"], planes=["reference"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            DYNAMICS.plane_coefficients(np.eye(5), 3, ["reference"])


class GeometryStreamingTests(unittest.TestCase):
    def run_geometry(
        self, study, settings, weights, random=None, reads=None, fail_at=None
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4))
                self.other = torch.nn.Parameter(torch.zeros(4))

        opened, counts = [], {}
        reads = [] if reads is None else reads
        for folder in weights:
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "model.safetensors"
            if not path.exists():
                path.write_bytes(b"mock weights")

        @contextlib.contextmanager
        def safe_open(path, **kwargs):
            path = Path(path)
            opened.append(path.parent)

            def get_tensor(name):
                key = (path.parent, name)
                reads.append(key)
                counts[key] = counts.get(key, 0) + 1
                if fail_at == (*key, counts[key]):
                    raise RuntimeError("simulated geometry interruption")
                return weights[path.parent][name]

            yield SimpleNamespace(get_tensor=get_tensor)

        source = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Model())
        helper = {"get_directions": lambda *args: random}
        with (
            patch.dict(
                "sys.modules",
                {
                    "safetensors": SimpleNamespace(safe_open=safe_open),
                    "transformers": SimpleNamespace(AutoModelForCausalLM=source),
                },
            ),
            patch.object(torch, "device", return_value=torch.device("cpu")),
            patch.object(torch.cuda, "empty_cache"),
            patch.object(DYNAMICS.runpy, "run_path", return_value=helper) as run_path,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            DYNAMICS.geometry(study, settings)
        return (
            json.loads((study / "geometry/geometry.json").read_text()),
            opened,
            run_path.call_count,
        )

    @staticmethod
    def parameters(vector):
        return {
            "weight": torch.tensor(vector[:4], dtype=torch.bfloat16),
            "other": torch.tensor(vector[4:], dtype=torch.bfloat16),
        }

    def cached_case(self, root, name, original):
        study = root / name
        (study / "geometry").mkdir(parents=True)
        original.mkdir(exist_ok=True)
        for filename, directions in (
            ("random-basis.npy", np.eye(8)[2:4]),
            ("attack-basis.npy", np.eye(8)[4:6]),
        ):
            path = original / filename
            if not path.exists():
                np.save(path, directions.astype(np.float32))
        for plane, filename in (
            ("random", "random-basis.npy"),
            ("reference", "attack-basis.npy"),
        ):
            (study / "geometry" / f"{plane}-basis.npy").symlink_to(original / filename)
        anchor = study / "models/defense"
        weights = {
            anchor: self.parameters(np.zeros(8)),
            study
            / "models/attack/checkpoint-16": self.parameters(
                np.eye(8)[0] + np.eye(8)[4]
            ),
            study
            / "models/attack/checkpoint-1000": self.parameters(
                np.eye(8)[1] + np.eye(8)[5]
            ),
            original / "old/models/defense": self.parameters(np.ones(8)),
            original
            / "old/models/attack-lr1e-6/checkpoint-16": self.parameters(np.zeros(8)),
        }
        settings = {
            "attack_steps": [16, 1000],
            "original_run": str(original / "old"),
            "shared_random_basis": str(original / "random-basis.npy"),
            "reference_bases": {"reference": str(original / "attack-basis.npy")},
            "planes": ["random", "reference", "attack"],
        }
        return study, settings, weights, anchor

    def assert_matches_fresh(self, root, original, result, study):
        fresh, settings, weights, _ = self.cached_case(root, "fresh", original)
        settings["geometry_progress_cache"] = False
        expected, _, _ = self.run_geometry(fresh, settings, weights)
        self.assertEqual(result, expected)
        np.testing.assert_array_equal(
            np.load(study / "geometry/attack-basis.npy"),
            np.load(fresh / "geometry/attack-basis.npy"),
        )

    def test_gram_resume_skips_completed_parameter_reads_and_preserves_comparisons(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, settings, weights, anchor = self.cached_case(
                root, "resumed", root / "shared"
            )
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                self.run_geometry(
                    study, settings, weights, fail_at=(anchor, "other", 1)
                )
            state = json.loads((study / "geometry/gram-progress.json").read_text())
            self.assertEqual(state["completed_parameters"], 1)
            self.assertEqual(state["squared_differences"]["defense"], 4)
            self.assertFalse((study / "geometry/geometry.json").exists())
            reads = []
            result, _, _ = self.run_geometry(study, settings, weights, reads=reads)
            self.assertEqual(reads.count((anchor, "weight")), 1)
            self.assertEqual(reads.count((anchor, "other")), 2)
            self.assertAlmostEqual(
                result["rerun_distance_from_original"]["defense"], np.sqrt(8)
            )
            self.assert_matches_fresh(root, root / "shared", result, study)

    def test_basis_resume_replays_uncommitted_suffix_and_never_writes_shared_sources(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "shared"
            study, settings, weights, anchor = self.cached_case(
                root, "resumed", original
            )
            source_state = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in original.glob("*.npy")
            }
            save = DYNAMICS._geometry_progress_save

            def interrupt(path, value):
                if (
                    path.name == "basis-progress.json"
                    and value["completed_parameters"] == 2
                ):
                    # The second parameter was written/flushed, but its commit
                    # did not publish. Its whole suffix must be replayed.
                    raise RuntimeError("simulated basis interruption")
                save(path, value)

            with patch.object(
                DYNAMICS, "_geometry_progress_save", side_effect=interrupt
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated basis"):
                    self.run_geometry(study, settings, weights)
            state = json.loads((study / "geometry/basis-progress.json").read_text())
            self.assertEqual(state["completed_parameters"], 1)
            self.assertFalse((study / "geometry/geometry.json").exists())
            reads = []
            result, _, _ = self.run_geometry(study, settings, weights, reads=reads)
            self.assertNotIn((anchor, "weight"), reads)
            self.assertEqual(reads.count((anchor, "other")), 1)
            self.assert_matches_fresh(root, original, result, study)
            for path, expected in source_state.items():
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
            for plane in ("random", "reference"):
                self.assertTrue((study / f"geometry/{plane}-basis.npy").is_symlink())

    def test_changed_inputs_and_settings_invalidate_both_passes(self):
        for change in ("weights", "settings"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                study, settings, weights, anchor = self.cached_case(
                    root, "cached", root / "shared"
                )
                self.run_geometry(study, settings, weights)
                old_identity = json.loads(
                    (study / "geometry/gram-progress.json").read_text()
                )["identity"]
                if change == "weights":
                    # Same mocked numerical weights, different underlying input
                    # identity: old progress still must never be reused.
                    (anchor / "model.safetensors").write_bytes(
                        b"replacement mock weights"
                    )
                else:
                    settings["minimum_radius"] = 0.01
                reads = []
                self.run_geometry(study, settings, weights, reads=reads)
                self.assertEqual(reads.count((anchor, "weight")), 2)
                self.assertEqual(reads.count((anchor, "other")), 2)
                new_identity = json.loads(
                    (study / "geometry/gram-progress.json").read_text()
                )["identity"]
                self.assertNotEqual(old_identity, new_identity)

    def test_invalid_basis_coefficients_or_shape_restart_only_the_second_pass(self):
        for change in ("coefficients", "shape"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                study, settings, weights, anchor = self.cached_case(
                    root, "cached", root / "shared"
                )
                expected, _, _ = self.run_geometry(study, settings, weights)
                if change == "coefficients":
                    path = study / "geometry/basis-progress.json"
                    state = json.loads(path.read_text())
                    state["coefficient_identity"] = "stale coefficients"
                    path.write_text(json.dumps(state))
                else:
                    np.save(
                        study / "geometry/attack-basis.npy",
                        np.zeros((2, 7), dtype=np.float32),
                    )
                reads = []
                result, _, _ = self.run_geometry(study, settings, weights, reads=reads)
                self.assertEqual(result, expected)
                self.assertEqual(reads.count((anchor, "weight")), 1)
                self.assertEqual(reads.count((anchor, "other")), 1)

    def test_shared_planes_symlink_exact_arrays_and_project_from_attack_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study, original = root / "condition", root / "original"
            (study / "geometry").mkdir(parents=True)
            original.mkdir()
            rng = np.random.default_rng(12)
            directions = np.linalg.qr(rng.normal(size=(8, 4)))[0].T.astype(np.float32)
            directions[0] *= np.float32(1 + 1e-6)
            np.save(original / "random-basis.npy", directions[:2])
            np.save(original / "attack-basis.npy", directions[2:])
            (original / "geometry.json").write_text(
                json.dumps(
                    {
                        "names": ["weight", "other"],
                        "shapes": {"weight": [4], "other": [4]},
                        "offsets": {"weight": 0, "other": 4},
                    }
                )
            )
            anchor = study / "models/attack/checkpoint-0"
            baseline = np.arange(8, dtype=float)
            displacements = np.array(
                [[1, 0, 0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 1, 0]],
                dtype=float,
            )
            checkpoints = [
                study / f"models/attack/checkpoint-{step}" for step in (1, 1000)
            ]
            weights = {anchor: self.parameters(baseline)}
            weights.update(
                {
                    path: self.parameters(baseline + delta)
                    for path, delta in zip(checkpoints, displacements)
                }
            )
            settings = {
                "attack_steps": [1, 1000],
                "attack_anchor": str(anchor),
                "compare_original_weights": False,
                "shared_random_basis": str(original / "random-basis.npy"),
                "reference_bases": {"reference": str(original / "attack-basis.npy")},
                "planes": ["random", "reference", "attack"],
            }
            result, opened, helper_calls = self.run_geometry(study, settings, weights)
            self.assertEqual(opened, [anchor, *checkpoints])
            self.assertEqual(helper_calls, 0)
            self.assertEqual(result["rerun_distance_from_original"], {})
            self.assertIsNone(result["random_transverse_overlap"])
            self.assertEqual(list(result["planes"]), settings["planes"])
            self.assertEqual(np.shape(result["basis_actual_gram"]), (6, 6))
            self.assertAlmostEqual(result["radius"], 1.25 * np.sqrt(5))
            self.assertFalse((study / "geometry/transverse-basis.npy").exists())
            for plane, expected in (
                ("random", directions[:2]),
                ("reference", directions[2:]),
            ):
                target = study / f"geometry/{plane}-basis.npy"
                self.assertTrue(target.is_symlink())
                np.testing.assert_array_equal(np.load(target), expected)
                basis = expected.astype(np.float64)
                small = basis @ basis.T
                coordinates = (displacements @ basis.T) @ np.linalg.inv(small)
                residual = displacements - coordinates @ basis
                rows = result["planes"][plane]
                np.testing.assert_allclose(
                    [[row["c0"], row["c1"]] for row in rows], coordinates, atol=1e-12
                )
                np.testing.assert_allclose(
                    [row["off_plane_distance"] for row in rows],
                    np.linalg.norm(residual, axis=1),
                    atol=1e-12,
                )

    def test_rank_zero_geometry_uses_radius_floor_and_only_requested_basis(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "geometry").mkdir()
            source = study / "shared-basis.npy"
            saved = np.full((2, 8), 7, dtype=np.float32)
            np.save(source, saved)
            (study / "geometry/attack-basis.npy").symlink_to(source)
            anchor = study / "models/defense"
            checkpoint = study / "models/attack/checkpoint-1000"
            weights = {
                path: self.parameters(np.zeros(8)) for path in (anchor, checkpoint)
            }
            directions = [self.parameters(np.eye(8)[index]) for index in (0, 1)]
            settings = {
                "attack_steps": [1000],
                "compare_original_weights": False,
                "direction_seed": 17,
                "allow_degenerate": True,
                "planes": ["attack"],
            }
            result, opened, helper_calls = self.run_geometry(
                study, settings, weights, random=directions
            )
            self.assertEqual(opened, [anchor, checkpoint])
            self.assertEqual(helper_calls, 1)
            self.assertEqual(result["attack_rank"], 0)
            self.assertEqual(result["attack_plane_energy_fraction"], 0.0)
            self.assertEqual(result["radius"], 1e-3)
            self.assertEqual(np.shape(result["basis_actual_gram"]), (2, 2))
            self.assertEqual(
                list((study / "geometry").glob("*-basis.npy")),
                [study / "geometry/attack-basis.npy"],
            )
            row = result["planes"]["attack"][0]
            self.assertEqual(row["distance"], 0.0)
            self.assertEqual(row["off_plane_distance"], 0.0)
            self.assertEqual(row["captured_fraction"], 1.0)
            self.assertFalse((study / "geometry/attack-basis.npy").is_symlink())
            np.testing.assert_array_equal(np.load(source), saved)

    def test_default_geometry_compares_original_weights_and_keeps_three_planes(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / "geometry").mkdir()
            original = study / "original"
            anchor = study / "models/defense"
            weights = {
                path: self.parameters(np.zeros(8))
                for path in (anchor, original / "models/defense")
            }
            for step, index in ((16, 0), (32, 1)):
                for parent in (
                    study / "models/attack",
                    original / "models/attack-lr1e-6",
                ):
                    weights[parent / f"checkpoint-{step}"] = self.parameters(
                        np.eye(8)[index]
                    )
            settings = {
                "attack_steps": [16, 32],
                "direction_seed": 17,
                "original_run": str(original),
            }
            result, opened, helper_calls = self.run_geometry(
                study,
                settings,
                weights,
                random=[self.parameters(np.eye(8)[index]) for index in (2, 3)],
            )
            self.assertEqual(helper_calls, 1)
            self.assertEqual(set(opened), set(weights))
            self.assertEqual(
                result["rerun_distance_from_original"],
                {"defense": 0.0, "attack-16": 0.0, "attack-32": 0.0},
            )
            self.assertEqual(list(result["planes"]), ["random", "attack", "transverse"])
            self.assertEqual(result["random_transverse_overlap"], 1.0)
            self.assertEqual(result["radius"], 1.25)
            self.assertEqual(np.shape(result["basis_actual_gram"]), (6, 6))


if __name__ == "__main__":
    unittest.main()
