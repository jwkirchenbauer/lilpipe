"""Tiny metadata fixtures only; never invoke sealing on the real studies."""

import copy
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import seal


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def observations(path, maximum, role, metric=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for step in range(1, maximum + 1):
        batch = {
            "input_ids": [[step, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[step, 2, 3]],
        }
        if role == "defense":
            batch["metadata_ids"] = [[0, 1, 2]]
        records.append(
            {
                "step": step,
                "batch": batch,
                "loss": metric,
                "learning_rate": metric,
                "batch_sha256": "ignored",
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return records


class TrainingTests(unittest.TestCase):
    def fixture(self, root, role):
        maximum = 125 if role == "defense" else 1000
        reference = root / "baseline.jsonl"
        observations(reference, maximum, role)
        folder = root / "case"
        observed = folder / "models" / role / "observed-batches.jsonl"
        observations(observed, maximum, role, metric=123)
        signature = seal.batch_signature(reference, maximum, role)
        case = {
            "role": role,
            "max_steps": maximum,
            "save_steps": [1, maximum],
            "state_steps": [maximum],
        }
        save(folder / "status/train.json", {"state": "complete"})
        save(
            folder / "training-validation.json",
            {
                "state": "passed",
                "updates": maximum,
                "model_checkpoints": 2,
                "optimizer_checkpoints": [maximum],
                "input_order_validation": {
                    "state": "passed",
                    "batches": maximum,
                    "reference_observations": str(reference),
                    "canonical_step_batch_sha256": signature,
                },
            },
        )
        return folder, case, reference, observed

    def test_both_roles_require_actual_tensors_but_ignore_metrics(self):
        for role in ("defense", "attack"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                folder, case, reference, _ = self.fixture(Path(directory), role)
                result = seal.training_gate(folder, case, reference, {})
                self.assertEqual(result["updates"], case["max_steps"])
                self.assertEqual(len(result["canonical_step_batch_sha256"]), 64)

    def test_changed_reordered_or_missing_batch_rejects_stale_passed_proof(self):
        for change in ("tokens", "order", "missing"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                folder, case, reference, observed = self.fixture(
                    Path(directory), "defense"
                )
                records = [
                    json.loads(line) for line in observed.read_text().splitlines()
                ]
                if change == "tokens":
                    records[1]["batch"]["input_ids"][0][0] += 1
                elif change == "order":
                    records[0]["batch"], records[1]["batch"] = (
                        records[1]["batch"],
                        records[0]["batch"],
                    )
                else:
                    records.pop()
                observed.write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                with self.assertRaises(seal.GateError):
                    seal.training_gate(folder, case, reference, {})

    def test_pending_status_failed_guard_wrong_reference_and_checkpoint_gaps_reject(
        self,
    ):
        for change in ("pending", "guard", "reference", "milestone"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                folder, case, reference, _ = self.fixture(Path(directory), "defense")
                proof_path = folder / "training-validation.json"
                proof = seal.read(proof_path)
                if change == "pending":
                    save(folder / "status/train.json", {"state": "running"})
                elif change == "guard":
                    proof["input_order_validation"]["state"] = "failed"
                elif change == "reference":
                    proof["input_order_validation"]["reference_observations"] = str(
                        Path(directory) / "other"
                    )
                else:
                    proof["optimizer_checkpoints"] = []
                save(proof_path, proof)
                with self.assertRaises(seal.GateError):
                    seal.training_gate(folder, case, reference, {})


class SourceTests(unittest.TestCase):
    def test_manifest_checks_source_hash_and_rejects_tensor_members_without_reading(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "worker.py"
            source.write_text("frozen\n")
            manifest = root / "manifest.sha256"
            manifest.write_text(f"{seal.digest(source)}  worker.py\n")
            self.assertEqual(
                len(seal.verify_small_manifest(manifest, 1000)["members"]), 1
            )
            source.write_text("changed\n")
            with self.assertRaisesRegex(seal.GateError, "mismatch"):
                seal.verify_small_manifest(manifest, 1000)
            tensor = root / "basis.npy"
            tensor.write_bytes(b"toy")
            manifest.write_text("0" * 64 + "  basis.npy\n")
            with patch.object(
                seal, "digest", side_effect=AssertionError("tensor read")
            ):
                with self.assertRaisesRegex(seal.GateError, "tensor"):
                    seal.verify_small_manifest(manifest, 1000)


class CurationTests(unittest.TestCase):
    def fixture(self, root):
        repository = (root / "lilpipe").resolve()
        example = repository / "examples/tamper-resistance"
        hr, fr = example / "artifacts/hr", example / "artifacts/fr"
        config = seal.read(Path(__file__).with_name("requirements.json"))
        for path in (
            hr / "report.md",
            hr / "figure-index.md",
            hr / "checkpoint-report.md",
            hr / "static/comparison/report.md",
            hr / "static/comparison/findings.md",
            fr / "static/full-factorial/report.md",
        ):
            write(path, "Completed fixture.\n")
        figure = hr / "cases/attack/curve.png"
        write(figure, "small image fixture")
        write(hr / "figure-index.md", "[curve](cases/attack/curve.png)\n")
        for run in (hr, fr):
            write(run / "study.json", "{}\n")
            write(run / "launches/completed/logs/flux-done.out", "completed worker\n")
        return repository, hr, fr, config, figure

    def test_excludes_weights_arrays_caches_large_and_live_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, hr, fr, config, figure = self.fixture(Path(directory))
            forbidden = [
                hr / "models/defense/model.safetensors",
                hr / "geometry/attack-basis.npy",
                hr / "cache/progress.json",
                hr / "arrays/scalars.npz",
                hr / "source/__pycache__/worker.pyc",
                hr / "launches/seal-final/logs/flux-live.out",
            ]
            for path in forbidden:
                write(path, "excluded")
            (hr / "geometry/broken-basis.npy").symlink_to(hr / "missing.npy")
            large = hr / "launches/completed/logs/huge.out"
            with large.open("wb") as stream:
                stream.truncate(config["maximum_log_bytes"] + 1)
            selected, excluded = seal.select_payload(
                hr, fr, repository, config, {figure}
            )
            self.assertIn(figure, selected)
            self.assertFalse(set(forbidden) & selected)
            self.assertNotIn(large, selected)
            self.assertTrue(
                any(row["reason"] == "large worker stdout/log" for row in excluded)
            )

    def test_required_broken_or_model_link_fails_before_publication(self):
        for target in ("missing.png", "models/defense/config.json"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                repository, hr, fr, config, _ = self.fixture(Path(directory))
                if target.startswith("models/"):
                    write(hr / target, "{}")
                write(hr / "report.md", f"[bad]({target})\n")
                with self.assertRaisesRegex(seal.GateError, "link"):
                    seal.select_payload(hr, fr, repository, config, set())
                self.assertFalse((hr / "artifact-manifest.sha256").exists())

    def test_repository_relative_bundle_hashes_both_roots_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, hr, fr, config, figure = self.fixture(Path(directory))
            selected, excluded = seal.select_payload(
                hr, fr, repository, config, {figure}
            )
            result = seal.publish_bundle(
                hr, fr, repository, config, {"fixture_only": True}, selected, excluded
            )
            self.assertEqual(result["state"], "verified")
            entries = seal.manifest_entries(hr / "artifact-manifest.sha256", repository)
            self.assertTrue(any(path.is_relative_to(fr) for path in entries))
            self.assertFalse(
                any(
                    line.startswith("/")
                    for line in (hr / "tracked-artifacts.txt").read_text().splitlines()
                )
            )
            self.assertNotIn(hr / "artifact-manifest.sha256", entries)
            with self.assertRaisesRegex(seal.GateError, "already exist"):
                seal.publish_bundle(hr, fr, repository, config, {}, selected, excluded)
            figure.write_text("changed")
            with self.assertRaisesRegex(seal.GateError, "mismatch"):
                seal.verify_bundle(hr, repository)

    def test_mutation_during_hashing_rejected_and_manifest_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proof.json"
            path.write_text("original")
            original_digest = seal.digest

            def mutate(candidate):
                value = original_digest(candidate)
                candidate.write_text("modified after hashing")
                return value

            with patch.object(seal, "digest", side_effect=mutate):
                with self.assertRaisesRegex(seal.GateError, "changed while hashing"):
                    seal.stable_hashes({path})
            manifest = Path(directory) / "manifest.sha256"
            manifest.write_text("0" * 64 + "  ../escape.json\n")
            with self.assertRaisesRegex(seal.GateError, "safe repository-relative"):
                seal.manifest_entries(manifest, Path(directory))

    def test_verify_refuses_symlinks_before_opening_a_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "array.npy"
            target.write_bytes(b"not read")
            alias = Path(directory) / "image.png"
            alias.symlink_to(target)
            with patch.object(
                seal, "digest", side_effect=AssertionError("target read")
            ):
                with self.assertRaisesRegex(seal.GateError, "symlink"):
                    seal.verify_hashes({alias: "0" * 64})

    def test_cli_requires_an_allocation_even_for_verify(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = dict(os.environ)
            environment.pop("FLUX_JOB_ID", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("seal.py")),
                    "verify",
                    directory,
                    directory,
                ],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allocated launcher job", result.stderr)


class AggregateTests(unittest.TestCase):
    def fixture(self, output):
        output.mkdir(parents=True)
        save(
            output / "aggregate-validation.json",
            {
                "state": "complete",
                "conditions": ["B", "R", "I", "O", "RIO"],
                "numeric_validation_rows": 5,
            },
        )

        def csv_rows(name, values):
            with (output / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)

        csv_rows(
            "numeric-validation-coverage.csv",
            [
                {"condition": cell, "issues": 0, "physical_comparable": True}
                for cell in ("B", "R", "I", "O", "RIO")
            ],
        )
        csv_rows(
            "fixed-plane-coverage.csv",
            [
                {
                    "condition": cell,
                    "plane": plane,
                    "measured_values": 289,
                    "grid_points": 17,
                }
                for cell in ("B", "R", "I", "O", "RIO")
                for plane in ("random", "reference", "attack")
            ],
        )
        figures = [
            {
                "figure": f"end-defense-fixed-{plane}",
                "status": "complete",
                "color_min": 0,
                "color_max": 15,
                "frames": "endpoint",
                "panels": 5,
            }
            for plane in ("random", "reference", "attack")
        ]
        for name, frames in (
            ("broad-random-r125", [0, 128, 1000]),
            ("local-reference-matched-attack", [0, 1, 8, 32, 128, 512, 1000]),
            ("defense-history-local-reference", [0, 8, 32, 125]),
        ):
            figures.append(
                {
                    "figure": name,
                    "status": "complete",
                    "color_min": 0,
                    "color_max": 15,
                    "frames": ";".join(map(str, frames)),
                    "panels": 5 * len(frames),
                }
            )
        csv_rows("figure-index.csv", figures)

    def test_complete_five_condition_coverage_requires_all_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aggregate"
            self.fixture(output)
            _, required = seal.aggregate_gate(output)
            self.assertEqual(len(required), 12)
            values = seal.rows(output / "figure-index.csv")
            next(
                row
                for row in values
                if row["figure"] == "local-reference-matched-attack"
            )["frames"] = "0;1000"
            with (output / "figure-index.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
            with self.assertRaisesRegex(seal.GateError, "frame coverage"):
                seal.aggregate_gate(output)


class LossMatrixTests(unittest.TestCase):
    def fixture(self, root):
        repository, hr, fr, config, figure = CurationTests.fixture(self, root)
        path = hr / "cases/hparam-r-lr1e-6/surfaces/minibatch-losses.npz"
        path.parent.mkdir(parents=True)
        steps = np.array(seal.LOSS_MATRIX_STEPS, dtype=np.int64)
        batches = np.array([1, 2, 128, 1000], dtype=np.int64)
        losses = (
            np.arange(len(steps) * len(batches), dtype=np.float64).reshape(
                len(steps), len(batches)
            )
            / 10
            + 1
        )
        save(
            path.parents[1] / "study.json",
            {
                "matrix_steps": steps.tolist(),
                "matrix_batch_steps": batches.tolist(),
                "attack_max_steps": 1000,
            },
        )
        arrays = {"losses": losses, "steps": steps, "batch_steps": batches}
        np.savez(path, **arrays)
        return repository, hr, fr, config, figure, path, arrays

    def test_exact_tiny_ce_matrix_is_curated_hashed_and_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, hr, fr, config, figure, path, _ = self.fixture(Path(directory))
            original = path.read_bytes()
            proof = seal.validate_loss_matrix(path, hr, config)
            self.assertEqual(proof["shape"], [7, 4])
            selected, excluded = seal.select_payload(
                hr, fr, repository, config, {figure, path}
            )
            self.assertIn(path, selected)
            result = seal.publish_bundle(
                hr, fr, repository, config, {"fixture_only": True}, selected, excluded
            )
            self.assertEqual(result["state"], "verified")
            self.assertIn(
                path, seal.manifest_entries(hr / "artifact-manifest.sha256", repository)
            )
            self.assertEqual(path.read_bytes(), original)

    def test_similar_filenames_elsewhere_and_other_npz_stay_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            repository, hr, fr, config, _, path, _ = self.fixture(Path(directory))
            other_paths = [
                hr / "models/attack/minibatch-losses.npz",
                hr / "geometry/minibatch-losses.npz",
                hr / "cases/hparam-ri-lr1e-6/surfaces/minibatch-losses.npz",
                path.with_name("other.npz"),
                fr / path.relative_to(hr),
            ]
            for other in other_paths:
                other.parent.mkdir(parents=True, exist_ok=True)
                other.write_bytes(path.read_bytes())
                self.assertIsNotNone(seal.exclusion(other, repository, config, hr))
                with patch.object(
                    seal,
                    "digest",
                    side_effect=AssertionError("non-allowlisted NPZ read"),
                ):
                    with self.assertRaises(seal.GateError):
                        seal.verify_hashes(
                            {other: "0" * 64}, loss_matrix_root=hr, config=config
                        )

    def test_large_allowlisted_archive_rejected_before_zip_read(self):
        with tempfile.TemporaryDirectory() as directory:
            _, hr, _, config, _, path, _ = self.fixture(Path(directory))
            path.write_bytes(b"x" * (seal.LOSS_MATRIX_MAXIMUM_BYTES + 1))
            with patch.object(
                seal.zipfile,
                "ZipFile",
                side_effect=AssertionError("large archive opened"),
            ):
                with self.assertRaisesRegex(seal.GateError, "large"):
                    seal.validate_loss_matrix(path, hr, config)

    def test_schema_shape_dtype_finite_and_axis_mismatches_rejected(self):
        for change in (
            "member",
            "shape",
            "dtype",
            "object",
            "nan",
            "negative",
            "steps",
            "batches",
        ):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                _, hr, _, config, _, path, arrays = self.fixture(Path(directory))
                if change == "member":
                    arrays["model_weights"] = np.zeros(1)
                elif change == "shape":
                    arrays["losses"] = np.zeros((8, 4))
                elif change == "dtype":
                    arrays["losses"] = arrays["losses"].astype(np.float32)
                elif change == "object":
                    arrays["losses"] = arrays["losses"].astype(object)
                elif change == "nan":
                    arrays["losses"][0, 0] = np.nan
                elif change == "negative":
                    arrays["losses"][0, 0] = -1
                elif change == "steps":
                    arrays["steps"][1] = 2
                else:
                    arrays["batch_steps"][-1] = 999
                np.savez(path, **arrays)
                with self.assertRaises(seal.GateError):
                    seal.validate_loss_matrix(path, hr, config)

    def test_declared_unpacked_size_and_fake_huge_shape_rejected_before_array_view(
        self,
    ):
        for change in ("unpacked", "header"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                _, hr, _, config, _, path, _ = self.fixture(Path(directory))
                if change == "unpacked":
                    losses = b"x" * (seal.LOSS_MATRIX_MAXIMUM_BYTES + 1)
                else:
                    stream = io.BytesIO()
                    np.lib.format.write_array_header_1_0(
                        stream,
                        {"descr": "<f8", "fortran_order": False, "shape": (7, 10**9)},
                    )
                    losses = stream.getvalue()
                    self.assertTrue(losses.startswith(b"\x93NUMPY"))
                with zipfile.ZipFile(
                    path, "w", compression=zipfile.ZIP_DEFLATED
                ) as archive:
                    archive.writestr("losses.npy", losses)
                    archive.writestr("steps.npy", b"not reached")
                    archive.writestr("batch_steps.npy", b"not reached")
                with patch.object(
                    np,
                    "frombuffer",
                    side_effect=AssertionError("unbounded array constructed"),
                ):
                    with self.assertRaises(seal.GateError):
                        seal.validate_loss_matrix(path, hr, config)

    def test_symlink_at_allowlisted_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, hr, _, config, _, path, _ = self.fixture(Path(directory))
            target = path.with_name("model.npz")
            path.rename(target)
            path.symlink_to(target)
            with self.assertRaisesRegex(seal.GateError, "symlink"):
                seal.validate_loss_matrix(path, hr, config)


if __name__ == "__main__":
    unittest.main()
