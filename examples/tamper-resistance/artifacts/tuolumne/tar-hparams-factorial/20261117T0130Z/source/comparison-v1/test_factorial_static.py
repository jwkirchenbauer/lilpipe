"""Synthetic CPU tests; no model loading or scientific rendering."""

import copy
import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from factorial_static_report import (
    CELLS,
    METRICS,
    RADII,
    InvalidInput,
    bits,
    factorial_contrasts,
    grid_metrics,
    load_standard,
    parameters,
    validate_records,
)
from static_recipe import file_identity, sha


def response(coefficients):
    result = {}
    for cell in CELLS:
        r, i, o = bits(cell)
        result[cell] = sum(
            coefficients.get(term, 0) * value
            for term, value in (
                ("B", 1),
                ("R", r),
                ("I", i),
                ("O", o),
                ("RI", r * i),
                ("RO", r * o),
                ("IO", i * o),
                ("RIO", r * i * o),
            )
        )
    return result


def rows(center=10):
    return [
        {
            "radius": radius,
            "c0": -radius + i * radius / 4,
            "c1": -radius + j * radius / 4,
            "loss": float(center),
        }
        for radius in RADII
        for i, j in itertools.product(range(9), repeat=2)
    ]


def records():
    metadata = {
        "get_ll_sha256": "original",
        "torch": "torch",
        "hip": "hip",
        "seed": 42,
        "document_sha256": [str(i) for i in range(10)],
        "dataset_revision": "pinned",
        "radii": RADII,
        "n_pts": 9,
        "batch_size": 2,
        "max_seq_len": 512,
    }
    return [
        {
            "cell": cell,
            "parameters": parameters(cell),
            "common": copy.deepcopy(metadata),
            "rows": rows(),
            "batch_hashes": [str(i) for i in range(5)],
            "order_sha256": "same",
        }
        for cell in CELLS
    ]


class FactorialTests(unittest.TestCase):
    def anchored(self, values):
        return {
            row["factors"]: row["value"]
            for row in factorial_contrasts(values)
            if row["kind"] == "anchored_coefficient"
        }

    def test_additive_main_effects_and_zero_interactions(self):
        actual = self.anchored(response({"B": 11, "R": 2, "I": -3, "O": 0.5}))
        self.assertEqual(
            actual, {"R": 2, "I": -3, "O": 0.5, "RI": 0, "RO": 0, "IO": 0, "RIO": 0}
        )

    def test_pair_effect_and_conditional_difference_recovery(self):
        contrasts = factorial_contrasts(response({"B": 4, "R": 2, "RI": 5}))
        pair = [
            row
            for row in contrasts
            if row["kind"] == "conditional_pair" and row["factors"] == "RI"
        ]
        self.assertEqual([row["value"] for row in pair], [5, 5])
        marginal = next(
            row
            for row in contrasts
            if row["kind"] == "marginal_main" and row["factors"] == "R"
        )
        self.assertEqual((marginal["value"], marginal["normalization"]), (4.5, 4))
        self.assertEqual(self.anchored(response({"B": 4, "R": 2, "RI": 5}))["RIO"], 0)

    def test_three_way_is_change_in_conditional_pair_and_raw_coefficient(self):
        contrasts = factorial_contrasts(response({"B": 4, "RIO": 7}))
        pair = [
            row
            for row in contrasts
            if row["kind"] == "conditional_pair" and row["factors"] == "RI"
        ]
        self.assertEqual([row["value"] for row in pair], [0, 7])
        self.assertEqual(self.anchored(response({"B": 4, "RIO": 7}))["RIO"], 7)
        marginal = next(
            row
            for row in contrasts
            if row["kind"] == "marginal_pair" and row["factors"] == "RI"
        )
        self.assertEqual((marginal["value"], marginal["normalization"]), (3.5, 2))

    def test_all_coefficients_recovered_together(self):
        expected = {"R": 2, "I": -3, "O": 0.5, "RI": 5, "RO": -2, "IO": 0.75, "RIO": 7}
        self.assertEqual(self.anchored(response({"B": 4, **expected})), expected)

    def test_missing_factorial_cell_rejects(self):
        values = response({"B": 11})
        del values["IO"]
        with self.assertRaisesRegex(InvalidInput, "All 8"):
            factorial_contrasts(values)


class GridDiagnosticTests(unittest.TestCase):
    def test_exact_neighbor_and_perimeter_definitions_keep_negative_signs(self):
        values = rows()
        for row in values:
            if row["radius"] != 100:
                continue
            if max(abs(row["c0"]), abs(row["c1"])) == 100:
                row["loss"] = 12
            elif (
                row["c0"] in (-25, 0, 25)
                and row["c1"] in (-25, 0, 25)
                and (row["c0"] or row["c1"])
            ):
                row["loss"] = 9
        self.assertEqual(
            grid_metrics(values, 100),
            {"center_ce": 10, "rim_rise": -1, "center_elevation": -2},
        )

    def test_rim_uses_maximum_of_eight_neighbors(self):
        values = rows()
        target = next(
            row
            for row in values
            if row["radius"] == 125 and row["c0"] == row["c1"] == 31.25
        )
        target["loss"] = 14
        self.assertEqual(grid_metrics(values, 125)["rim_rise"], 4)


class ComparabilityTests(unittest.TestCase):
    def test_complete_eight_records_pass_with_exact_common_inputs(self):
        proof = validate_records(records())
        self.assertEqual(proof["defense_order_sha256"], "same")
        self.assertEqual(len(proof["evaluated_batch_sha256"]), 5)

    def test_missing_duplicate_or_wrong_factor_case_rejected(self):
        for change in (
            lambda values: values.pop(),
            lambda values: values.__setitem__(1, copy.deepcopy(values[0])),
            lambda values: values[1]["parameters"].__setitem__(
                "retain_coefficient", 2.5
            ),
        ):
            values = records()
            change(values)
            with self.assertRaises(InvalidInput):
                validate_records(values)

    def test_source_versions_seed_texts_tokens_and_order_must_match(self):
        for field in (
            "get_ll_sha256",
            "torch",
            "hip",
            "seed",
            "document_sha256",
            "dataset_revision",
            "radii",
        ):
            values = records()
            values[1]["common"][field] = "changed"
            with self.assertRaises(InvalidInput):
                validate_records(values)
        for field in ("batch_hashes", "order_sha256"):
            values = records()
            values[1][field] = (
                ["changed"] * 5 if field == "batch_hashes" else "different"
            )
            with self.assertRaises(InvalidInput):
                validate_records(values)

    def test_missing_or_shifted_grid_axes_rejected(self):
        for change in (
            lambda rows: rows.pop(),
            lambda rows: rows[0].__setitem__("c0", -999.999),
        ):
            values = records()
            change(values[1]["rows"])
            with self.assertRaisesRegex(InvalidInput, "axes"):
                validate_records(values)


class SavedCacheTests(unittest.TestCase):
    def fixture(self, root):
        folder, static = root / "case", root / "static"
        model = folder / "models/defense"
        for path in (folder / "status", static, model):
            path.mkdir(parents=True)

        def save(path, value):
            path.write_text(json.dumps(value))

        for name in ("train", "static"):
            save(folder / "status" / f"{name}.json", {"state": "complete"})
        save(
            folder / "case.json",
            {
                "role": "defense",
                "max_steps": 125,
                "defense_hyperparameters": parameters("RI"),
            },
        )
        save(model / "config.json", {"fixture": True})
        (model / "model.safetensors").write_bytes(b"tiny identity fixture")
        (folder / "config.yml").write_text("fixture: true\n")
        reference = root / "baseline-observations.jsonl"
        reference.write_text("frozen reference\n")
        save(
            folder / "training-validation.json",
            {
                "state": "passed",
                "updates": 125,
                "input_order_validation": {
                    "state": "passed",
                    "batches": 125,
                    "canonical_step_batch_sha256": "a" * 64,
                    "reference_observations": str(reference),
                },
            },
        )
        source = root / "get_ll.py"
        source.write_text("# frozen fixture source\n")
        protocol = {
            "training_validation_sha256": sha(folder / "training-validation.json"),
            "defense_config_sha256": sha(folder / "config.yml"),
            "baseline_tokenizer_and_collation_match": True,
            "default_dataset_fixed10_matches_pinned": True,
            "checkpoint": str(model),
            "checkpoint_files": file_identity(model),
            "source_sha256": {"get_ll.py": sha(source)},
            "evaluated_batch_sha256": ["b" * 64] * 5,
            "torch": "torch",
            "hip": "hip",
            "seed": 42,
            "document_sha256": [str(i) for i in range(10)],
            "dataset_revision": "pinned",
            "radii": RADII,
            "n_pts": 9,
            "batch_size": 2,
            "max_seq_len": 512,
        }
        save(static / "protocol.json", protocol)
        with (static / "grid.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("radius", "c0", "c1", "loss"))
            writer.writeheader()
            writer.writerows(rows())
        save(
            static / "validation.json",
            {
                "state": "passed",
                "points": 648,
                "grid_sha256": sha(static / "grid.csv"),
                "protocol_sha256": sha(static / "protocol.json"),
            },
        )
        return {
            "cell": "RI",
            "case_folder": str(folder),
            "static_folder": str(static),
            "baseline_observations": str(reference),
            "get_ll_source": str(source),
        }

    def test_guarded_complete_numeric_fixture_loads_without_model_read(self):
        with tempfile.TemporaryDirectory() as directory:
            record = load_standard(self.fixture(Path(directory)))
            self.assertEqual((record["cell"], len(record["rows"])), ("RI", 648))
            self.assertEqual(record["order_sha256"], "a" * 64)

    def test_changed_model_config_grid_or_source_rejected(self):
        for target in (
            "case/models/defense/model.safetensors",
            "case/config.yml",
            "static/grid.csv",
            "get_ll.py",
        ):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                entry = self.fixture(root)
                path = root / target
                path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaises(InvalidInput):
                    load_standard(entry)

    def test_pending_status_or_failed_exact_batch_guard_rejected(self):
        for target, value in (
            ("status/train.json", {"state": "running"}),
            ("status/static.json", {"state": "failed"}),
            (
                "training-validation.json",
                {
                    "state": "passed",
                    "updates": 125,
                    "input_order_validation": {"state": "failed", "batches": 125},
                },
            ),
        ):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                entry = self.fixture(root)
                (root / "case" / target).write_text(json.dumps(value))
                with self.assertRaises(InvalidInput):
                    load_standard(entry)


if __name__ == "__main__":
    unittest.main()
