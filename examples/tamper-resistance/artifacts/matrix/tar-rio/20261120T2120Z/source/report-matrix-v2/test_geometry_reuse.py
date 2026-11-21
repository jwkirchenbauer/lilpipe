import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = HERE.parents[1]
COMPARISON = TARGET / "source/misc/tar_compute_compare.py"
ADAPTER = HERE / "geometry_reuse.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = _load_module("frozen_tar_compute_compare", COMPARISON)
geometry_reuse = _load_module("geometry_reuse", ADAPTER)


def _protocol_identity(protocol):
    return hashlib.sha256(
        json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def _fixture(root):
    folder = root / "case"
    geometry_dir = folder / "geometry"
    status_dir = folder / "status"
    geometry_dir.mkdir(parents=True)
    status_dir.mkdir()
    baseline = root / "baseline"
    baseline_geometry = baseline / "geometry"
    baseline_geometry.mkdir(parents=True)

    names = ["weight"]
    shapes = {"weight": [1]}
    offsets = {"weight": 0}
    layout = {"names": names, "shapes": shapes, "offsets": offsets}
    (baseline_geometry / "geometry.json").write_text(json.dumps(layout))
    for name in ("random-basis.npy", "attack-basis.npy"):
        (baseline_geometry / name).write_bytes(b"small fixture basis marker")
    (geometry_dir / "random-basis.npy").symlink_to(
        baseline_geometry / "random-basis.npy"
    )
    (geometry_dir / "reference-basis.npy").symlink_to(
        baseline_geometry / "attack-basis.npy"
    )

    input_path = root / "geometry-input.bin"
    input_path.write_bytes(b"fixture geometry input")
    input_stat = input_path.stat()
    frames = [0, 1]
    planes = {
        plane: [{"step": 1, "distance": 0.0}]
        for plane in ("random", "reference", "attack")
    }
    gram = [[1.0]]
    basis_actual_gram = [[1.0] * 6 for _ in range(6)]
    geom = {
        **layout,
        "attack_rank": 1,
        "attack_plane_energy_fraction": 1.0,
        "radius": 1.0,
        "steps": [1],
        "planes": planes,
        "gram": gram,
        "basis_actual_gram": basis_actual_gram,
    }
    (geometry_dir / "geometry.json").write_text(json.dumps(geom))

    fixed_indices = [0, 1]
    heldout_hashes = [f"heldout-{index}" for index in range(32)]
    settings = {
        "attack_max_steps": 1,
        "actual_attack_steps": [1],
        "attack_steps": [1],
        "attack_frames": frames,
        "broad_attack_frames": frames,
        "local_frames": frames,
        "broad_radius": 125,
        "broad_points": 3,
        "local_radius": comparison.LOCAL_RADIUS,
        "local_surface_points": 3,
        "local_plane": "reference",
        "fixed_source_indices": fixed_indices,
        "forget_revision": "fixture-revision",
        "shared_random_basis": str(baseline_geometry / "random-basis.npy"),
    }
    current_settings = {
        **settings,
        "capability_scores_file": str(root / "matrix" / "evaluation-scores.csv"),
        "cpu_landscape_anchor": True,
    }
    saved_settings = {
        **settings,
        "capability_scores_file": str(root / "original" / "evaluation-scores.csv"),
    }
    protocol = {
        "settings": saved_settings,
        "inputs": [
            {
                "path": str(input_path),
                "resolved": str(input_path.resolve()),
                "size": input_stat.st_size,
                "mtime_ns": input_stat.st_mtime_ns,
            }
        ],
    }
    identity = _protocol_identity(protocol)
    (geometry_dir / "progress-protocol.json").write_text(
        json.dumps({"protocol": protocol, "identity": identity})
    )
    (geometry_dir / "gram-progress.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "completed_parameters": len(names),
                "gram": gram,
            }
        )
    )
    (geometry_dir / "basis-progress.json").write_text(
        json.dumps(
            {
                "identity": identity,
                "completed_parameters": len(names),
                "actual_gram": basis_actual_gram,
                "actual_cross": [[0.0] * 6],
                "coefficient_identity": "a" * 64,
                "owned_files": {},
            }
        )
    )

    probes = {
        "fixed_source_indices": fixed_indices,
        "forget_revision": "fixture-revision",
        "heldout_sha256": heldout_hashes,
    }
    observed = {
        "fixed_documents": 10,
        "fixed_source_indices": fixed_indices,
        "heldout_sha256": heldout_hashes,
    }
    actual = [
        {"phase": "attack", "step": step, "probe_loss": 1.0, "heldout_loss": 1.1}
        for step in frames
    ]
    scores = [
        {"attack_step": step, "task": task, "jwk_percent": 50.0}
        for step in frames
        for task in ("wmdp_bio_robust", "mmlu_no_bio")
    ]
    projection = [
        {
            "plane": plane,
            "step": step,
            "distance": 0.0,
            "captured_fraction": 1.0,
            "off_plane_distance": 0.0,
            "projected_loss": 1.0,
            "probe_loss": 1.0,
            "loss_gap": 0.0,
        }
        for plane in planes
        for step in frames
    ]
    local = []
    broad = []
    for step in frames:
        for i, y in enumerate([-1.0, 0.0, 1.0]):
            for j, x in enumerate([-1.0, 0.0, 1.0]):
                local.append(
                    {
                        "phase": "attack",
                        "step": step,
                        "c0": x * comparison.LOCAL_RADIUS,
                        "c1": y * comparison.LOCAL_RADIUS,
                        "probe_loss": 1.0,
                    }
                )
                broad.append(
                    {
                        "phase": "attack",
                        "step": step,
                        "c0": x * 125,
                        "c1": y * 125,
                        "probe_loss": 1.0,
                    }
                )

    for stage in ("train", "evaluation", "analysis"):
        (status_dir / f"{stage}.json").write_text(json.dumps({"state": "complete"}))

    record = {
        "folder": folder,
        "settings": current_settings,
        "geometry": geom,
        "declarations": {},
        "probes": observed,
        "schedule_steps": 1,
        "case": {"max_steps": 1},
        "save_steps": [1],
        "actual": actual,
        "scores": scores,
        "projection": projection,
        "local": local,
        "broad": broad,
        "local_frames": frames,
        "broad_frames": frames,
    }
    program = {"baseline_dynamics": str(baseline)}
    return record, program, probes, input_path


class GeometryReuseTests(unittest.TestCase):
    def validate(self, record, program, probes):
        wrapped = geometry_reuse.wrap_checkpoint_validation(
            comparison.checkpoint_validation
        )
        return wrapped(record, program, probes)

    def test_actual_validator_accepts_two_operational_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, program, probes, _ = _fixture(Path(temporary))
            before = json.loads(json.dumps(record, default=str))
            result = self.validate(record, program, probes)
            self.assertEqual(result["issues"], [])
            self.assertEqual(
                result["geometry_reuse"]["changed_fields"],
                ["capability_scores_file", "cpu_landscape_anchor"],
            )
            self.assertFalse(result["geometry_reuse"]["geometry_recomputed"])
            self.assertEqual(json.loads(json.dumps(record, default=str)), before)

    def test_actual_validator_rejects_geometric_and_shared_input_settings(self):
        changes = {
            "local_radius": 0.25,
            "direction_seed": 99,
            "attack_steps": [1, 2],
            "shared_random_basis": "/different/shared/random-basis.npy",
        }
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                record, program, probes, _ = _fixture(Path(temporary))
                record["settings"][key] = value
                result = self.validate(record, program, probes)
                self.assertIn(
                    "Geometry progress settings or identity is stale", result["issues"]
                )
                self.assertNotIn("geometry_reuse", result)

    def test_actual_validator_rejects_corrupt_protocol_self_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, program, probes, _ = _fixture(Path(temporary))
            path = Path(record["folder"]) / "geometry/progress-protocol.json"
            progress = json.loads(path.read_text())
            progress["protocol"]["unexpected"] = "corrupt"
            path.write_text(json.dumps(progress))
            result = self.validate(record, program, probes)
            self.assertIn(
                "Geometry progress settings or identity is stale", result["issues"]
            )

    def test_actual_validator_rejects_stage_identity_and_accumulator_corruption(self):
        for corruption in ("identity", "accumulator"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as temporary,
            ):
                record, program, probes, _ = _fixture(Path(temporary))
                path = Path(record["folder"]) / "geometry/gram-progress.json"
                state = json.loads(path.read_text())
                if corruption == "identity":
                    state["identity"] = "bad"
                    expected = "Geometry gram progress is incomplete or stale"
                else:
                    state["gram"] = [[2.0]]
                    expected = "Geometry gram accumulator differs from final geometry"
                path.write_text(json.dumps(state))
                result = self.validate(record, program, probes)
                self.assertIn(expected, result["issues"])

    def test_actual_validator_rejects_changed_input_stat(self):
        with tempfile.TemporaryDirectory() as temporary:
            record, program, probes, input_path = _fixture(Path(temporary))
            input_path.write_bytes(b"changed geometry input with another size")
            result = self.validate(record, program, probes)
            self.assertTrue(
                any(
                    issue.startswith("Geometry cache input identity changed")
                    for issue in result["issues"]
                )
            )


if __name__ == "__main__":
    unittest.main()
