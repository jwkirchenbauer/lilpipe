"""Check comparison math, partial endpoints, and evidence for shared slices."""

import copy
import builtins
import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_compute_compare.py"
)
SPEC = importlib.util.spec_from_file_location("tar_compute_compare", SCRIPT)
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_identity(metadata):
    metadata["identity"] = hashlib.sha256(
        json.dumps(
            metadata["protocol"], sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def profile_fixture(root):
    folder = root / "cases/baseline-lr1e-5"
    anchor = folder / "models/defense"
    shared = root / "baseline/geometry/random-basis.npy"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"the report must never load this basis array")
    write_json(shared.parent / "geometry.json", {})
    checkpoints = {
        str(step): (
            anchor if step == 0 else folder / "models/attack" / f"checkpoint-{step}"
        )
        for step in range(9)
    }
    for checkpoint in checkpoints.values():
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"only stat these weights")
    write_json(anchor / "config.json", {})
    write_json(anchor / "tokenizer.json", {})
    write_json(anchor / "tokenizer_config.json", {})
    source = root / "source/jobs/profiles/tar_dynamics.py"
    source.parent.mkdir(parents=True)
    source.write_text("# frozen profile source\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    probes = {
        "fixed_source_indices": list(range(10)),
        "forget_revision": "frozen-revision",
    }
    settings = {
        **probes,
        "attack_anchor": str(anchor),
        "shared_random_basis": str(shared),
    }
    write_json(folder / "study.json", settings)
    write_json(folder / "status/step-profiles.json", {"state": "complete"})
    case = {
        "role": "attack",
        "max_steps": 1000,
        "save_steps": [1, 2, 3, 4, 5, 6, 7, 8, 1000],
        "gaussian_sigma": None,
    }
    steps, rows, observed = {}, [], []
    for step in range(1, 9):
        before = {"probe_loss": 5 - 0.1 * (step - 1), "minibatch_loss": 4 + 0.1 * step}
        after = {
            "probe_loss": 5 - 0.1 * step,
            "minibatch_loss": before["minibatch_loss"] - 0.3,
        }
        norm = 0.5 + 0.01 * (step - 1)
        selected = []
        for fraction in COMPARE.PROFILE_FRACTIONS:
            selected.append(
                {
                    "update_step": step,
                    "fraction": float(fraction),
                    "signed_displacement": float(fraction) * norm,
                    **{
                        field: before[field]
                        + float(fraction) * (after[field] - before[field])
                        + 0.8 * float(fraction) * (1 - float(fraction))
                        for field in before
                    },
                }
            )
        rows.extend(selected)
        batch = {"input_ids": [[step]], "labels": [[step]]}
        observed.append(
            {
                "step": step,
                "batch": batch,
                "batch_sha256": hashlib.sha256(
                    json.dumps(batch, sort_keys=True).encode()
                ).hexdigest(),
                "loss_before_update": before["minibatch_loss"] - 0.2,
                "loss_after_update": after["minibatch_loss"] + 0.1,
            }
        )
        steps[str(step)] = {
            "update_l2": norm,
            "raw_update_l2": norm,
            "evaluated_update_l2": norm / 2,
            "changed_fraction": 0.02,
            "changed_scalars": 2,
            "raw_changed_scalars": 2,
            "raw_changed_fraction": 0.02,
            "evaluated_changed_scalars": 1,
            "evaluated_changed_fraction": 0.01,
            "saved_dtype_scalars": {
                "before": {"torch.bfloat16": 80, "torch.float32": 20},
                "after": {"torch.bfloat16": 80, "torch.float32": 20},
            },
            "arithmetic_t0_evaluated_cast_mismatches": 0,
            "arithmetic_t1_evaluated_cast_mismatches": 2,
            "t0_weights_exact": True,
            "t1_weights_exact": True,
            "actual_before": before,
            "actual_after": after,
            "endpoint_errors": {
                f"t{fraction}_{field}": abs(
                    next(row[field] for row in selected if row["fraction"] == fraction)
                    - endpoint[field]
                )
                for fraction, endpoint in ((0, before), (1, after))
                for field in before
            },
            "actual_losses_csv_errors": {},
            "sampled_barriers": {
                field: COMPARE.sampled_segment_barrier(selected, field)
                for field in before
            },
            "recorded_train_before_ce_error": 0.2,
            "recorded_train_after_ce_error": -0.1,
        }
    trace = folder / "models/attack/observed-batches.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in observed))
    inputs = [
        shared,
        shared.parent / "geometry.json",
        trace,
        *sorted(anchor.glob("*.json")),
        *(path / "model.safetensors" for path in checkpoints.values()),
    ]
    protocol = {
        "version": 2,
        "source_sha256": source_sha,
        "settings": settings,
        "inputs": [
            {
                "path": str(path.absolute()),
                "resolved": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in inputs
        ],
        "batch_hashes": {str(row["step"]): row["batch_sha256"] for row in observed},
        "checkpoint_paths": {step: str(path) for step, path in checkpoints.items()},
        "fractions": COMPARE.PROFILE_FRACTIONS.tolist(),
        "grid_c0": COMPARE.PROFILE_C0.tolist(),
        "grid_c1": COMPARE.PROFILE_C1.tolist(),
        **probes,
        "tokenizer_source": str(anchor),
        "torch_version": "fixture",
        "precision": "BF16",
        "coordinate_convention": "Raw saved-state displacement",
        "endpoint_convention": "Exact saved endpoints copied to model evaluation dtypes",
        "interpretation": "sampled segments",
    }
    metadata = {
        "state": "complete",
        "source_sha256": source_sha,
        "protocol": protocol,
        **probes,
        "fixed_text_sha256": [
            hashlib.sha256(f"fixed {index}".encode()).hexdigest() for index in range(10)
        ],
        "steps": steps,
        "endpoint_ce_tolerance": 1e-4,
        "first_update_plane": {
            "status": "fitted",
            "first_update_l2": 0.5,
            "q0_norm": 1,
            "q1_norm": 1,
            "q0_dot_q1": 0,
            "q0_radius": 1,
            "q1_radius": 125,
            "shared_random_row": 0,
            "anchor": str(anchor),
            "after_checkpoint": str(checkpoints["1"]),
            "endpoint_coordinates": [0.5, 0],
            "endpoint_in_window": True,
            "plane_endpoint_bf16_mismatches": 0,
            "plane_endpoint_evaluated_cast_mismatches": 0,
            "plane_endpoint_losses": copy.deepcopy(steps["1"]["actual_after"]),
            "plane_endpoint_ce_errors": {"probe_loss": 0, "minibatch_loss": 0},
            "endpoint_comparison_convention": "Raw normalized axes cast to model evaluation dtypes",
            "center_errors": {"probe_loss": 0, "minibatch_loss": 0},
        },
    }
    profile_identity(metadata)
    grid = [
        {
            "c0": float(x),
            "c1": float(y),
            "probe_loss": 5 - 0.2 * float(x) + 0.001 * float(y),
            "minibatch_loss": 4.1 - 0.6 * float(x) + 0.001 * float(y),
        }
        for x in COMPARE.PROFILE_C0
        for y in COMPARE.PROFILE_C1
    ]
    record = {
        "name": "baseline-lr1e-5",
        "folder": folder,
        "analysis": folder,
        "case": case,
        "settings": settings,
        "actual": [],
        "trace": COMPARE.observations(trace, 1000),
        "step_profiles": metadata,
        "step_profile_rows": rows,
        "first_update_grid": grid,
    }
    program = {
        "baseline_dynamics": str(shared.parent.parent),
        "source_sha256": {str(source.relative_to(root)): source_sha},
    }
    return record, program, probes


def write_profile_adapter_fixture(root, record, program, probes):
    program = {**program, "cases": {record["name"]: record["case"]}}
    write_json(root / "study.json", program)
    write_json(root / "probes.json", probes)
    write_json(
        record["folder"] / "surfaces/step-profiles.json", record["step_profiles"]
    )
    write_csv(
        record["folder"] / "surfaces/step-profiles.csv", record["step_profile_rows"]
    )
    write_csv(
        record["folder"] / "surfaces/first-update-grid.csv", record["first_update_grid"]
    )
    return program


class ComparisonMathTests(unittest.TestCase):
    def test_budget_maximum_includes_stronger_rates_and_earlier_checkpoints(self):
        records = {}
        for rate, score in (("1e-5", 60), ("1e-6", 35), ("5e-6", 58)):
            records[f"baseline-lr{rate}"] = {
                "case": {
                    "save_steps": [16, 128, 256, 512, 1000],
                    "attack_learning_rate": rate,
                },
                "scores": [
                    {"task": "wmdp_bio_robust", "attack_step": 0, "jwk_percent": 33},
                    {
                        "task": "wmdp_bio_robust",
                        "attack_step": 16,
                        "jwk_percent": score,
                    },
                    {
                        "task": "wmdp_bio_robust",
                        "attack_step": 128,
                        "jwk_percent": score - 1,
                    },
                    {"task": "mmlu_no_bio", "attack_step": 128, "jwk_percent": 99},
                ],
            }
        records["baseline-lr5e-6"]["scores"].append(
            {"task": "wmdp_bio_robust", "attack_step": 256, "jwk_percent": 61}
        )
        result = COMPARE.strongest_attacks(records)
        baseline = [row for row in result if row["defense_family"] == "baseline"]
        self.assertEqual(baseline[0]["peak_measured_wmdp"], 60)
        self.assertEqual(baseline[0]["peak_learning_rate"], "1e-5")
        self.assertEqual(baseline[0]["peak_attack_step"], 16)
        self.assertEqual(baseline[0]["measured_case_checkpoints"], 9)
        self.assertEqual(baseline[0]["expected_case_checkpoints"], 9)
        self.assertTrue(baseline[0]["complete_through_budget"])
        self.assertEqual(baseline[1]["peak_measured_wmdp"], 61)
        self.assertEqual(baseline[1]["peak_case"], "baseline-lr5e-6")
        self.assertEqual(baseline[1]["measured_case_checkpoints"], 10)
        self.assertEqual(baseline[1]["expected_case_checkpoints"], 12)
        self.assertFalse(baseline[1]["complete_through_budget"])
        long = next(row for row in result if row["defense_family"] == "long")
        self.assertIsNone(long["peak_measured_wmdp"])
        self.assertEqual(long["measured_learning_rates"], 0)
        self.assertFalse(long["complete_through_budget"])

    def test_overlap_uses_recorded_plane_order_and_marks_rank_zero(self):
        gram = np.eye(6)
        gram[:2, 4:6] = np.diag([0.8, 0.6])
        gram[4:6, :2] = gram[:2, 4:6].T
        geometry = {
            "attack_rank": 3,
            "planes": {"attack": [], "random": [], "reference": []},
            "basis_actual_gram": gram.tolist(),
        }
        value, note = COMPARE.plane_overlap(geometry)
        self.assertAlmostEqual(value, 0.5)
        self.assertEqual(note, "")
        geometry["attack_rank"] = 0
        value, note = COMPARE.plane_overlap(geometry)
        self.assertIsNone(value)
        self.assertIn("rank-zero", note)
        geometry["attack_rank"] = 1
        self.assertIn("random completion", COMPARE.plane_overlap(geometry)[1])

    def test_partial_surface_keeps_physical_coordinates_and_missing_points(self):
        radius = COMPARE.LOCAL_RADIUS
        rows = [
            {
                "phase": "attack",
                "step": 0,
                "c0": -radius,
                "c1": radius,
                "probe_loss": 9.0,
            },
            {"phase": "attack", "step": 0, "c0": 0, "c1": 0, "probe_loss": 10.0},
            {"phase": "attack", "step": 1000, "c0": 0, "c1": 0, "probe_loss": 4.0},
        ]
        frame = COMPARE.surface_grid(rows, 0, radius, 3)
        np.testing.assert_allclose(frame["coordinates"], [-radius, 0, radius])
        self.assertEqual(frame["measured"], 2)
        self.assertEqual(frame["expected"], 9)
        self.assertEqual(frame["values"][2, 0], 9.0)
        self.assertEqual(frame["values"][1, 1], 10.0)
        self.assertTrue(np.isnan(frame["values"][0, 0]))
        self.assertEqual(frame["issues"], [])
        rows.append({"phase": "attack", "step": 0, "c0": 1.0, "c1": 0, "probe_loss": 3})
        invalid = COMPARE.surface_grid(rows, 0, radius, 3)
        self.assertIn("physical lattice", invalid["issues"][0])

    def test_family_color_limits_do_not_hide_height_or_shape_differences(self):
        low, high = COMPARE.color_limits([3, 4, 12, 15])
        self.assertLess(low, 3)
        self.assertGreater(high, 15)
        self.assertEqual(
            COMPARE.color_limits([-0.1, 0.2, -3, 1], relative=True), (-3, 3)
        )
        low, high = COMPARE.color_limits([0, 0], relative=True)
        self.assertGreater(high, low)
        self.assertEqual(low, -high)


class AggregateFixtureTests(unittest.TestCase):
    def test_partial_endpoints_and_ambiguous_source_rows_remain_honest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            folder = run / "cases/noise-1e-4"
            analysis = folder
            case = {
                "role": "attack",
                "max_steps": 1000,
                "save_steps": [128, 1000],
                "bootstrap_seed": None,
            }
            write_json(
                folder / "source-dataset.json",
                {
                    "population_size": 3,
                    "draw_size": 3,
                    "unique_source_rows": 2,
                    "batch_sources": {"same": [0, 1], "different": [2]},
                    "drawn_source_rows": [0, 0, 1],
                },
            )
            write_json(
                folder / "runtime.json",
                {"wall_seconds": 3600, "allocated_gpu_seconds": 14400},
            )
            write_json(
                folder / "models/attack/noise.json",
                {"actual_l2_after_cast": 0.25, "changed_fraction": 0.01},
            )
            write_json(
                analysis / "summary.json",
                {
                    "attack_plane_energy_fraction": 0.8,
                    "planes": {
                        "reference": {
                            "minimum_capture": 0.2,
                            "maximum_projection_loss_gap": 3.5,
                        }
                    },
                },
            )
            write_json(
                analysis / "geometry/geometry.json",
                {
                    "steps": [128],
                    "attack_rank": 2,
                    "planes": {"attack": [{"step": 128, "distance": 0.4}]},
                },
            )
            write_csv(
                analysis / "surfaces/actual-losses.csv",
                [
                    {
                        "phase": "attack",
                        "step": 0,
                        "probe_loss": 11,
                        "heldout_loss": 12,
                    },
                    {
                        "phase": "attack",
                        "step": 128,
                        "probe_loss": 4,
                        "heldout_loss": 5,
                    },
                ],
            )
            write_csv(
                folder / "evaluation-scores.csv",
                [
                    {"attack_step": 0, "task": "mmlu_no_bio", "jwk_percent": 54},
                    {"attack_step": 0, "task": "wmdp_bio_robust", "jwk_percent": 33},
                    {"attack_step": 128, "task": "wmdp_bio_robust", "jwk_percent": 45},
                ],
            )
            trace = folder / "models/attack/observed-batches.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "step": step,
                            "batch_sha256": "same",
                            "source_rows": [0, 1],
                            "forward_backward_seconds": 2,
                        }
                    )
                    for step in (1, 2)
                )
            )
            record = COMPARE.collect_case(run, "noise-1e-4", case)
            metrics = record["metrics"]
            self.assertEqual(metrics["fixed_ce_start"], 11)
            self.assertIsNone(metrics["fixed_ce_final"])
            self.assertIsNone(metrics["heldout_ce_final"])
            self.assertIsNone(metrics["actual_displacement"])
            self.assertEqual(metrics["latest_actual_step"], 128)
            self.assertEqual(metrics["latest_geometry_step"], 128)
            self.assertEqual(metrics["peak_measured_wmdp"], 45)
            self.assertEqual(metrics["peak_measured_wmdp_step"], 128)
            self.assertIsNone(metrics["final_wmdp"])
            self.assertIsNone(metrics["final_drop_from_base_pp"])
            self.assertEqual(metrics["initial_mmlu"], 54)
            self.assertTrue(metrics["initial_within_5pp_base"])
            self.assertFalse(metrics["evaluation_complete"])
            self.assertEqual(metrics["observed_distinct_content_hashes"], 1)
            self.assertEqual(metrics["prepared_unique_source_rows"], 2)
            self.assertEqual(metrics["recorded_forward_backward_hours"], 4 / 3600)
            self.assertEqual(metrics["wall_hours"], 1)
            self.assertEqual(metrics["allocated_gpu_hours"], 4)
            self.assertEqual(metrics["maximum_functional_projection_gap"], 3.5)
            self.assertEqual(metrics["noise_actual_l2"], 0.25)

    def test_partial_trace_is_reported_without_inventing_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text('{"step":1,"batch_sha256":"a"}\n{"step":')
            result = COMPARE.observations(path, 2)
            self.assertEqual(result["observed_steps"], 1)
            self.assertFalse(result["observations_complete"])
            self.assertIsNone(result["forward_backward_seconds"])
            self.assertIn("Incomplete observation JSON", result["issues"][0])


class StepProfileTests(unittest.TestCase):
    def test_profile_values_reuses_complete_numbers_despite_failed_external_plot_status(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            write_json(
                record["folder"] / "status/step-profiles.json",
                {"state": "failed", "exit_code": 139},
            )
            write_profile_adapter_fixture(root, record, program, probes)
            adjacent = root / "source/jobs/profiles/tar_compute_compare.py"
            with (
                patch.object(COMPARE, "__file__", str(adjacent)),
                patch.object(
                    COMPARE.runpy,
                    "run_path",
                    side_effect=AssertionError(
                        "Complete numbers must not invoke model source"
                    ),
                ),
            ):
                result = COMPARE.profile_values(root, record["name"])
            self.assertTrue(result["reused"])
            self.assertEqual(
                json.loads(
                    (record["folder"] / "status/step-profiles.json").read_text()
                )["state"],
                "failed",
            )
            record["step_profile_state"] = "failed"
            self.assertFalse(
                COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )["step_profiles_complete"]
            )
            self.assertTrue(
                COMPARE.validate_step_profiles(
                    record,
                    program,
                    probes,
                    record["step_profiles"],
                    require_status=False,
                )["step_profiles_complete"]
            )

    def test_profile_values_suppresses_only_function_global_renderer_and_restores_it(
        self,
    ):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            original, returned_dict_renderer = Mock(), Mock()
            scope = {"_profile_plots": original, "calls": []}
            exec(
                "def step_profiles(folder, settings):\n    calls.append((folder, settings))\n    _profile_plots(folder, object())\n",
                scope,
            )
            namespace = {
                "step_profiles": scope["step_profiles"],
                "_profile_plots": returned_dict_renderer,
            }
            invalid = {"step_profiles_comparable": False, "issues": ["pending"]}
            valid = {"step_profiles_comparable": True, "issues": []}
            adjacent = root / "source/jobs/profiles/tar_compute_compare.py"
            with (
                patch.object(COMPARE, "__file__", str(adjacent)),
                patch.object(
                    COMPARE,
                    "_profile_adapter_record",
                    side_effect=[(record, program, invalid), (record, program, valid)],
                ),
                patch.object(COMPARE.runpy, "run_path", return_value=namespace),
            ):
                result = COMPARE.profile_values(root, record["name"])
            self.assertFalse(result["reused"])
            self.assertEqual(scope["calls"], [(record["folder"], record["settings"])])
            original.assert_not_called()
            returned_dict_renderer.assert_not_called()
            self.assertIs(
                scope["step_profiles"].__globals__["_profile_plots"], original
            )
            self.assertIs(namespace["_profile_plots"], returned_dict_renderer)
            exec(
                "def step_profiles(folder, settings):\n    raise RuntimeError('numeric failure')\n",
                scope,
            )
            with (
                patch.object(COMPARE, "__file__", str(adjacent)),
                patch.object(
                    COMPARE,
                    "_profile_adapter_record",
                    return_value=(record, program, invalid),
                ),
                patch.object(
                    COMPARE.runpy,
                    "run_path",
                    return_value={"step_profiles": scope["step_profiles"]},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "numeric failure"):
                    COMPARE.profile_values(root, record["name"])
            self.assertIs(
                scope["step_profiles"].__globals__["_profile_plots"], original
            )

    def test_profile_report_uses_cpu_helper_on_valid_cache_with_failed_external_status(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            source = root / "source/jobs/profiles/tar_dynamics.py"
            source.write_bytes(SCRIPT.with_name("tar_dynamics.py").read_bytes())
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            program["source_sha256"] = {str(source.relative_to(root)): digest}
            record["step_profiles"]["source_sha256"] = digest
            record["step_profiles"]["protocol"]["source_sha256"] = digest
            profile_identity(record["step_profiles"])
            write_profile_adapter_fixture(root, record, program, probes)
            write_json(
                record["folder"] / "status/step-profiles.json",
                {"state": "failed", "exit_code": 139},
            )
            before = {
                path.name: path.read_bytes()
                for path in (record["folder"] / "surfaces").iterdir()
            }
            real_import = builtins.__import__

            def cpu_import(name, *args, **kwargs):
                if name.split(".")[0] in {
                    "torch",
                    "transformers",
                    "datasets",
                    "safetensors",
                    "accelerate",
                }:
                    raise AssertionError(
                        f"CPU rendering imported model dependency {name}"
                    )
                return real_import(name, *args, **kwargs)

            with (
                patch.object(
                    COMPARE, "__file__", str(source.with_name("tar_compute_compare.py"))
                ),
                patch.object(builtins, "__import__", side_effect=cpu_import),
            ):
                result = COMPARE.profile_report(root, record["name"])
            self.assertEqual(result["state"], "complete")
            for stem in ("early-update-profiles", "first-update-landscape"):
                for extension in ("png", "pdf"):
                    self.assertGreater(
                        (record["folder"] / f"{stem}.{extension}").stat().st_size, 1000
                    )
            self.assertEqual(
                before,
                {
                    path.name: path.read_bytes()
                    for path in (record["folder"] / "surfaces").iterdir()
                },
            )
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
            self.assertEqual(
                json.loads(
                    (record["folder"] / "status/step-profiles.json").read_text()
                )["state"],
                "failed",
            )

    def test_profile_report_rejects_invalid_numbers_before_loading_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            record["step_profile_rows"].pop()
            write_profile_adapter_fixture(root, record, program, probes)
            with patch.object(
                COMPARE.runpy,
                "run_path",
                side_effect=AssertionError("Invalid cache must not load a renderer"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "Cannot render invalid numerical"
                ):
                    COMPARE.profile_report(root, record["name"])

    def test_complete_profiles_validate_and_metrics_use_only_measured_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            with patch.object(
                np, "load", side_effect=AssertionError("Never load giant arrays")
            ):
                valid = COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )
            self.assertEqual(valid["issues"], [])
            self.assertTrue(valid["step_profiles_complete"])
            self.assertTrue(valid["step_profiles_comparable"])
            self.assertTrue(valid["first_update_plane_comparable"])
            metrics = COMPARE.step_profile_metrics(record, valid)
            self.assertAlmostEqual(metrics["first_update_l2"], 0.5)
            self.assertAlmostEqual(metrics["first_update_evaluated_l2"], 0.25)
            self.assertAlmostEqual(
                metrics["first_update_evaluated_changed_fraction"], 0.01
            )
            self.assertAlmostEqual(metrics["first_update_fixed_ce_gain"], 0.1)
            self.assertAlmostEqual(metrics["first_update_batch_ce_gain"], 0.3)
            self.assertGreater(metrics["maximum_sampled_segment_barrier"], 0)
            rows = [
                {"fraction": value, "probe_loss": loss}
                for value, loss in ((-1, 100), (0, 3), (0.5, 5), (1, 4), (2, 200))
            ]
            self.assertEqual(COMPARE.sampled_segment_barrier(rows, "probe_loss"), 1)
            self.assertIsNone(COMPARE.sampled_segment_barrier(rows[:-2], "probe_loss"))

    def test_invalid_identity_endpoints_batches_and_lattices_are_withheld(self):
        with tempfile.TemporaryDirectory() as directory:
            original, program, probes = profile_fixture(Path(directory))
            mutations = (
                lambda r: r["step_profile_rows"].pop(),
                lambda r: r["step_profile_rows"][0].update(fraction=-2),
                lambda r: r["step_profile_rows"][0].update(signed_displacement=99),
                lambda r: r["step_profile_rows"][0].update(probe_loss=float("nan")),
                lambda r: r["step_profiles"]["steps"]["1"].update(
                    t1_weights_exact=False
                ),
                lambda r: r["step_profiles"]["steps"]["1"]["actual_after"].pop(
                    "probe_loss"
                ),
                lambda r: r["step_profiles"]["steps"]["1"]["sampled_barriers"].update(
                    probe_loss=100
                ),
                lambda r: r["step_profiles"]["steps"]["1"].update(
                    recorded_train_before_ce_error=99
                ),
                lambda r: r["step_profiles"]["first_update_plane"].update(q0_norm=0.9),
                lambda r: r["step_profiles"]["first_update_plane"].update(
                    q0_dot_q1=0.01
                ),
                lambda r: r["step_profiles"]["first_update_plane"].update(q1_radius=1),
                lambda r: r["step_profiles"]["first_update_plane"].update(
                    plane_endpoint_bf16_mismatches=1
                ),
                lambda r: r["step_profiles"]["first_update_plane"][
                    "center_errors"
                ].update(probe_loss=0.01),
                lambda r: r["first_update_grid"][0].update(c0=-125, c1=-1),
                lambda r: r["first_update_grid"].pop(),
                lambda r: r["step_profiles"]["protocol"]["batch_hashes"].update(
                    {"1": "0" * 64}
                ),
                lambda r: r["step_profiles"]["protocol"]["checkpoint_paths"].update(
                    {"1": "/wrong"}
                ),
                lambda r: r["step_profiles"]["protocol"]["inputs"][0].update(size=1),
                lambda r: r["step_profiles"]["protocol"].update(inputs=[]),
                lambda r: r["step_profiles"]["protocol"].update(
                    tokenizer_source="/wrong"
                ),
                lambda r: r["step_profiles"].update(endpoint_ce_tolerance=0.1),
            )
            for mutate in mutations:
                with self.subTest(mutation=mutate):
                    record = copy.deepcopy(original)
                    mutate(record)
                    profile_identity(record["step_profiles"])
                    valid = COMPARE.validate_step_profiles(
                        record, program, probes, original["step_profiles"]
                    )
                    self.assertFalse(valid["step_profiles_comparable"])
                    self.assertTrue(valid["issues"])
                    self.assertTrue(
                        all(
                            value is None
                            for value in COMPARE.step_profile_metrics(
                                record, valid
                            ).values()
                        )
                    )
            missing_reference = COMPARE.validate_step_profiles(
                original, program, probes, {}
            )
            self.assertTrue(missing_reference["step_profiles_complete"])
            self.assertFalse(missing_reference["step_profiles_comparable"])
            changed_reference = copy.deepcopy(original["step_profiles"])
            changed_reference["fixed_text_sha256"][0] = "0" * 64
            self.assertFalse(
                COMPARE.validate_step_profiles(
                    original, program, probes, changed_reference
                )["step_profiles_comparable"]
            )

    def test_later_actual_losses_rows_are_checked_without_saved_csv_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            self.assertEqual(
                record["step_profiles"]["steps"]["1"]["actual_losses_csv_errors"], {}
            )
            record["actual"] = [
                {"phase": "attack", "step": step, "probe_loss": 5 - 0.1 * step}
                for step in range(9)
            ]
            self.assertTrue(
                COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )["step_profiles_complete"]
            )
            record["actual"][1]["probe_loss"] += 0.01
            invalid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            self.assertFalse(invalid["step_profiles_complete"])
            self.assertTrue(
                any("current actual-losses.csv" in issue for issue in invalid["issues"])
            )

    def test_mixed_precision_plane_records_roundoff_but_requires_functional_endpoint_fidelity(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            proof = record["step_profiles"]["first_update_plane"]
            proof.update(
                plane_endpoint_bf16_mismatches=7,
                plane_endpoint_evaluated_cast_mismatches=7,
            )
            for field in ("probe_loss", "minibatch_loss"):
                proof["plane_endpoint_losses"][field] += 5e-5
                proof["plane_endpoint_ce_errors"][field] = 5e-5
            valid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            self.assertEqual(valid["issues"], [])
            metrics = COMPARE.step_profile_metrics(record, valid)
            self.assertEqual(metrics["first_update_plane_evaluated_cast_mismatches"], 7)
            self.assertAlmostEqual(metrics["first_update_plane_fixed_ce_error"], 5e-5)
            proof["plane_endpoint_losses"]["probe_loss"] += 0.01
            proof["plane_endpoint_ce_errors"]["probe_loss"] += 0.01
            self.assertFalse(
                COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )["step_profiles_complete"]
            )

    def test_malformed_profile_metadata_is_withheld_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            record["step_profiles"]["protocol"] = None
            invalid = COMPARE.validate_step_profiles(record, program, probes, {})
            self.assertFalse(invalid["step_profiles_complete"])
            self.assertEqual(invalid["first_update_plane_status"], "invalid")

    def test_completed_metadata_does_not_override_running_or_failed_profile_status(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            for state in ("running", "failed", "missing"):
                record["step_profile_state"] = state
                self.assertFalse(
                    COMPARE.validate_step_profiles(
                        record, program, probes, record["step_profiles"]
                    )["step_profiles_complete"]
                )

    def test_rectangular_grid_preserves_both_physical_axes_and_missing_values(self):
        x, y = np.linspace(-1, 1, 3), np.linspace(-125, 125, 5)
        rows = [
            {"c0": float(a), "c1": float(b), "probe_loss": float(10 * a + b)}
            for a in x
            for b in y
        ]
        frame = COMPARE.rectangular_grid(rows, x, y)
        self.assertEqual(frame["values"].shape, (5, 3))
        self.assertEqual(frame["values"][0, 2], -115)
        self.assertEqual(frame["values"][4, 0], 115)
        self.assertEqual(frame["issues"], [])
        rows.pop()
        frame = COMPARE.rectangular_grid(rows, x, y)
        self.assertTrue(np.isnan(frame["values"][4, 2]))
        rows.append({"c0": 125, "c1": 1, "probe_loss": 3})
        self.assertTrue(COMPARE.rectangular_grid(rows, x, y)["issues"])

    def test_explicit_degenerate_skip_keeps_segments_without_inventing_a_plane(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            record["first_update_grid"] = []
            record["step_profiles"]["first_update_plane"].update(
                status="skipped_parallel_shared_direction",
                orthogonal_completion_raw_norm=0,
            )
            valid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            self.assertTrue(valid["step_profiles_complete"])
            self.assertTrue(valid["step_profiles_comparable"])
            self.assertFalse(valid["first_update_plane_comparable"])
            record["step_profiles"]["first_update_plane"][
                "orthogonal_completion_raw_norm"
            ] = 0.01
            self.assertFalse(
                COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )["step_profiles_complete"]
            )

    def test_zero_update_is_an_explicit_complete_skip_with_constant_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            first = record["step_profiles"]["steps"]["1"]
            first.update(
                update_l2=0,
                raw_update_l2=0,
                evaluated_update_l2=0,
                changed_fraction=0,
                changed_scalars=0,
                raw_changed_fraction=0,
                raw_changed_scalars=0,
                evaluated_changed_fraction=0,
                evaluated_changed_scalars=0,
                actual_after=copy.deepcopy(first["actual_before"]),
                recorded_train_after_ce_error=0.2,
            )
            first["sampled_barriers"] = {"probe_loss": 0, "minibatch_loss": 0}
            for row in record["step_profile_rows"]:
                if row["update_step"] == 1:
                    row.update(signed_displacement=0, **first["actual_before"])
                else:
                    row["probe_loss"] += 0.1
            for step in range(2, 9):
                for side in ("actual_before", "actual_after"):
                    record["step_profiles"]["steps"][str(step)][side][
                        "probe_loss"
                    ] += 0.1
            proof = record["step_profiles"]["first_update_plane"]
            proof.update(
                status="skipped_zero_first_update",
                first_update_l2=0,
                endpoint_coordinates=[0, 0],
            )
            for key in (
                "q0_norm",
                "q1_norm",
                "q0_dot_q1",
                "plane_endpoint_bf16_mismatches",
                "center_errors",
            ):
                proof.pop(key)
            record["first_update_grid"] = []
            valid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            self.assertEqual(valid["issues"], [])
            self.assertTrue(valid["step_profiles_comparable"])
            self.assertFalse(valid["first_update_plane_comparable"])
            self.assertEqual(
                COMPARE.step_profile_metrics(record, valid)[
                    "first_update_fixed_ce_gain"
                ],
                0,
            )

    def test_raw_saved_norm_is_cross_checked_with_available_step_one_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = profile_fixture(Path(directory))
            record["geometry"] = {
                "planes": {"attack": [{"step": 1, "distance": 0.5000001}]}
            }
            self.assertTrue(
                COMPARE.validate_step_profiles(
                    record, program, probes, record["step_profiles"]
                )["step_profiles_complete"]
            )
            record["geometry"]["planes"]["attack"][0]["distance"] = 0.3
            invalid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            self.assertFalse(invalid["step_profiles_complete"])
            self.assertIn(
                "Raw saved first-update norm disagrees", ";".join(invalid["issues"])
            )

    def test_profile_coverage_is_required_only_when_declared_and_never_supplies_final_ce(
        self,
    ):
        record = {
            "case": {"role": "attack"},
            "metrics": {
                "train_state": "complete",
                "evaluation_state": "complete",
                "evaluation_complete": True,
                "analysis_state": "complete",
                "fixed_ce_final": None,
            },
        }
        self.assertTrue(
            COMPARE.scientific_case_complete(record, {"issues": []}, {}, False)
        )
        self.assertFalse(
            COMPARE.scientific_case_complete(record, {"issues": []}, {}, True)
        )
        self.assertTrue(
            COMPARE.scientific_case_complete(
                record, {"issues": []}, {"step_profiles_comparable": True}, True
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            profile, program, probes = profile_fixture(Path(directory))
            write_json(
                profile["folder"] / "surfaces/step-profiles.json",
                profile["step_profiles"],
            )
            write_csv(
                profile["folder"] / "surfaces/step-profiles.csv",
                profile["step_profile_rows"],
            )
            write_csv(
                profile["folder"] / "surfaces/first-update-grid.csv",
                profile["first_update_grid"],
            )
            write_csv(
                profile["folder"] / "surfaces/actual-losses.csv",
                [
                    {"phase": "attack", "step": step, "probe_loss": 5 - 0.1 * step}
                    for step in range(9)
                ],
            )
            measured = COMPARE.collect_case(
                Path(directory), profile["name"], profile["case"]
            )
            valid = COMPARE.validate_step_profiles(
                measured, program, probes, profile["step_profiles"]
            )
            self.assertTrue(valid["step_profiles_complete"])
            measured["metrics"].update(COMPARE.step_profile_metrics(measured, valid))
            self.assertIsNone(measured["metrics"]["fixed_ce_final"])
            self.assertIsNone(measured["metrics"]["final_wmdp"])
            self.assertEqual(measured["metrics"]["latest_actual_step"], 8)

    def test_family_profile_and_rectangular_landscape_figures_render(self):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            valid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            records, validations = {record["name"]: record}, {record["name"]: valid}
            names = [record["name"], "pending-case"]
            COMPARE.early_profile_figure(
                records, validations, "fixture", names, root / "profiles"
            )
            COMPARE.first_update_figure(
                records, validations, "fixture", names, root / "landscape"
            )
            for stem in ("profiles", "landscape"):
                for extension in ("png", "pdf"):
                    self.assertGreater(
                        (root / f"{stem}.{extension}").stat().st_size, 1000
                    )

    def test_first_update_surface_overlays_sampled_profile_without_changing_windows(
        self,
    ):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            valid = COMPARE.validate_step_profiles(
                record, program, probes, record["step_profiles"]
            )
            small, large = copy.deepcopy(record), copy.deepcopy(record)
            for current, norm in ((small, 0.01), (large, 1.5)):
                current["step_profiles"]["steps"]["1"]["update_l2"] = norm
                current["step_profiles"]["first_update_plane"]["endpoint_in_window"] = (
                    norm <= 1
                )
                for row in current["step_profile_rows"]:
                    if row["update_step"] == 1:
                        row["signed_displacement"] = row["fraction"] * norm
            for row in large["step_profile_rows"]:
                if row["update_step"] == 1 and row["fraction"] == 0.5:
                    row["probe_loss"] = 50
                if row["update_step"] == 1 and row["fraction"] == 2:
                    row["probe_loss"] = 100
            with patch.object(COMPARE, "save_figure") as save:
                COMPARE.first_update_figure(
                    {"small": small, "large": large},
                    {"small": valid, "large": valid},
                    "fixture",
                    ["small", "large"],
                    root / "overlay",
                )
            figure = save.call_args.args[0]
            first, second = figure.axes[0], figure.axes[2]
            for axis, current in ((first, small), (second, large)):
                line = axis.lines[0]
                x, y, z = line.get_data_3d()
                expected = sorted(
                    (
                        row
                        for row in current["step_profile_rows"]
                        if row["update_step"] == 1
                        and -1 <= row["signed_displacement"] <= 1
                    ),
                    key=lambda row: row["fraction"],
                )
                np.testing.assert_allclose(
                    x, [row["signed_displacement"] for row in expected]
                )
                np.testing.assert_allclose(y, 0)
                np.testing.assert_allclose(z, [row["probe_loss"] for row in expected])
                self.assertEqual(axis.get_xlim(), (-1, 1))
                self.assertEqual(axis.get_ylim(), (-125, 125))
                self.assertIn("Sampled interpolation/extrapolation", line.get_label())
                self.assertIn("Only t=0/1", line.get_label())
            self.assertEqual(len(first.lines[0].get_data_3d()[0]), 25)
            self.assertLess(len(second.lines[0].get_data_3d()[0]), 25)
            self.assertEqual(first.get_zlim(), second.get_zlim())
            self.assertGreater(first.get_zlim()[1], 50)
            self.assertLess(first.get_zlim()[1], 100)

    def test_report_keeps_shared_comparisons_and_legacy_coverage_while_profiles_pending(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record, program, probes = profile_fixture(root)
            name, folder = record["name"], record["folder"]
            program.update(cases={name: record["case"]}, step_profiles_required=True)
            write_json(root / "study.json", program)
            write_json(root / "probes.json", probes)
            for stage in ("train", "evaluation", "analysis"):
                write_json(folder / f"status/{stage}.json", {"state": "complete"})
            rows = [
                {"attack_step": step, "task": "wmdp_bio_robust", "jwk_percent": 40}
                for step in [0] + record["case"]["save_steps"]
            ]
            rows.extend(
                {"attack_step": step, "task": "mmlu_no_bio", "jwk_percent": 54}
                for step in (0, 1000)
            )
            write_csv(folder / "evaluation-scores.csv", rows)
            for stem in ("early-update-profiles", "first-update-landscape"):
                for extension in ("png", "pdf"):
                    (folder / f"{stem}.{extension}").write_text(
                        "existing producer output"
                    )
            shared = {
                "issues": [],
                "shared_probes_verified": True,
                "local_comparable": True,
                "broad_comparable": True,
            }
            with (
                patch.object(COMPARE, "FAMILIES", {"fixture": [name]}),
                patch.object(COMPARE, "validate_comparability", return_value=shared),
                patch.object(COMPARE, "trajectory_figure"),
                patch.object(COMPARE, "surface_figure"),
                patch.object(COMPARE, "early_profile_figure"),
                patch.object(COMPARE, "first_update_figure"),
            ):
                pending = COMPARE.report(root)
                self.assertEqual(pending["coverage"], "partial")
                self.assertTrue(pending["comparability"][name]["local_comparable"])
                self.assertFalse(
                    pending["step_profile_comparability"][name][
                        "step_profiles_complete"
                    ]
                )
                program.pop("step_profiles_required")
                write_json(root / "study.json", program)
                published = COMPARE.report(root)
                self.assertEqual(
                    published["coverage"],
                    "complete",
                    published.get("step_profile_comparability"),
                )
                program["step_profiles_required"] = True
                write_json(root / "study.json", program)
                write_json(
                    folder / "surfaces/step-profiles.json", record["step_profiles"]
                )
                write_csv(
                    folder / "surfaces/step-profiles.csv", record["step_profile_rows"]
                )
                write_csv(
                    folder / "surfaces/first-update-grid.csv",
                    record["first_update_grid"],
                )
                published = COMPARE.report(root)
                self.assertEqual(
                    published["coverage"],
                    "complete",
                    published.get("step_profile_comparability"),
                )
            report = (root / "report.md").read_text()
            for link in (
                "fixture-early-update-profiles.png",
                "fixture-first-update-landscape.pdf",
                "early-update-profiles.pdf",
                "first-update-landscape.png",
            ):
                self.assertIn(link, report)
            self.assertIn("schedule and exposure choices", report)
            self.assertIn("Final / peak measured WMDP", report)
            self.assertIn("planes differ between conditions", report)


class ComparabilityTests(unittest.TestCase):
    def fixture(self, root):
        baseline, analysis = root / "baseline", root / "case/analysis"
        (baseline / "geometry").mkdir(parents=True)
        (analysis / "geometry").mkdir(parents=True)
        for plane, filename in (
            ("random", "random-basis.npy"),
            ("reference", "attack-basis.npy"),
        ):
            source = baseline / "geometry" / filename
            source.write_bytes(b"directions are never loaded by the report")
            (analysis / "geometry" / f"{plane}-basis.npy").symlink_to(source)
        probes = {
            "fixed_source_indices": list(range(10)),
            "heldout_source_indices": list(range(100, 132)),
            "heldout_sha256": [str(index) for index in range(32)],
            "forget_revision": "frozen-revision",
            "expected_nonbootstrap_order_sha256": "clean-order",
        }
        record = {
            "name": "clean",
            "analysis": analysis,
            "case": {"bootstrap_seed": None},
            "trace": {"issues": []},
            "metrics": {
                "observations_complete": True,
                "observed_order_sha256": "clean-order",
            },
            "settings": {
                "shared_random_basis": str(baseline / "geometry/random-basis.npy"),
                "reference_bases": {
                    "reference": str(baseline / "geometry/attack-basis.npy")
                },
                "forget_revision": "frozen-revision",
                "local_plane": "reference",
                "local_radius": COMPARE.LOCAL_RADIUS,
                "plane_radii": {"random": 1, "reference": 1},
                "planes": ["random", "reference", "attack"],
                "broad_radius": 125,
            },
            "probes": {
                **copy.deepcopy(probes),
                "fixed_documents": 10,
                "heldout_excludes_observed_attack_documents": True,
            },
        }
        return record, {"baseline_dynamics": str(baseline)}, probes

    def test_matching_metadata_and_physical_sources_enable_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = self.fixture(Path(directory))
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertTrue(result["local_comparable"])
            self.assertTrue(result["broad_comparable"])
            self.assertTrue(result["shared_probes_verified"])
            self.assertEqual(result["issues"], [])
            record["probes"]["heldout_sha256"][0] = "different"
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertFalse(result["broad_comparable"])
            self.assertFalse(result["shared_probes_verified"])

    def test_scale_and_data_order_failures_are_not_silently_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            record, program, probes = self.fixture(Path(directory))
            record["settings"]["local_radius"] *= 2
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertTrue(result["broad_comparable"])
            record["settings"]["local_radius"] = COMPARE.LOCAL_RADIUS
            record["settings"]["broad_radius"] = 99
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertTrue(result["local_comparable"])
            self.assertFalse(result["broad_comparable"])
            record["metrics"]["observed_order_sha256"] = "different"
            result = COMPARE.validate_comparability(record, program, probes)
            self.assertFalse(result["local_comparable"])
            self.assertIn("different", record["metrics"]["observed_order_sha256"])
            self.assertTrue(
                any("data order differs" in issue for issue in result["issues"])
            )


if __name__ == "__main__":
    unittest.main()
