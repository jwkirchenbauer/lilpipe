import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_compute_compare.py"
)
SPEC = importlib.util.spec_from_file_location("tar_compute_compare", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rio_trajectory_peak_and_first_observed_recovery_thresholds():
    record = {
        "schedule_steps": 6,
        "scores": [
            {"task": "wmdp_bio_robust", "attack_step": "0", "jwk_percent": "40"},
            {"task": "wmdp_bio_robust", "attack_step": "2", "jwk_percent": "45"},
            {"task": "wmdp_bio_robust", "attack_step": "4", "jwk_percent": "50"},
            {"task": "wmdp_bio_robust", "attack_step": "6", "jwk_percent": "47"},
            {"task": "mmlu_no_bio", "attack_step": "4", "jwk_percent": "55"},
        ],
    }

    result = MODULE.rio_trajectory_metrics(record)

    assert result["peak_wmdp"] == 50
    assert result["peak_wmdp_step"] == 4
    assert result["measured_wmdp_points"] == 4
    assert result["first_at_or_above_45"] == 2
    assert result["first_at_or_above_50"] == 4
    assert result[f"first_at_or_above_{0.9 * MODULE.BASE_WMDP:g}"] == 4


def test_rio_trajectory_distinguishes_unreached_from_missing():
    record = {
        "schedule_steps": 4,
        "scores": [
            {"task": "wmdp_bio_robust", "attack_step": "0", "jwk_percent": "40"},
            {"task": "wmdp_bio_robust", "attack_step": "4", "jwk_percent": "44"},
        ],
    }

    result = MODULE.rio_trajectory_metrics(record)

    assert result["peak_wmdp"] == 44
    assert result["first_at_or_above_45"] is None
    assert result["first_at_or_above_50"] is None
    missing = MODULE.rio_trajectory_metrics({"schedule_steps": 4, "scores": []})
    assert missing["peak_wmdp"] is None
    assert missing["first_at_or_above_45"] is None


def test_rio_final_report_refuses_missing_measurements(tmp_path):
    stems = {
        "rio-k4",
        "rio-k64",
        "rio-k256",
        "rio-prefix-forward-legacy",
        "rio-prefix-reverse-legacy",
        "rio-fresh-matched",
        "rio-prefix-forward-matched",
        "rio-prefix-reverse-matched",
    }
    cases = {f"{stem}-lr1e-6": {"role": "attack", "max_steps": 1000} for stem in stems}
    (tmp_path / "study.json").write_text(
        json.dumps(
            {
                "kind": "rio-k-fidelity-study",
                "cases": cases,
                "comparison_controls": {
                    "rio-k16": {
                        "folder": str(tmp_path / "missing-control"),
                        "schedule_steps": 1000,
                    }
                },
                "baseline_dynamics": str(tmp_path / "missing-baseline"),
            }
        )
    )
    (tmp_path / "probes.json").write_text("{}")

    with pytest.raises(ValueError, match="pending measurements"):
        MODULE.rio_report(tmp_path)

    validation = json.loads((tmp_path / "rio-comparison-validation.json").read_text())
    assert validation["coverage"] == "pending"
    assert not (tmp_path / "rio-report.md").exists()


def test_parity_pilot_accepts_nested_actual_comparison_schema(tmp_path):
    folder = tmp_path / "pilot"
    (folder / "status").mkdir(parents=True)
    (folder / "models/attack/checkpoint-16").mkdir(parents=True)
    (folder / "status/train.json").write_text(json.dumps({"state": "complete"}))
    parameter_checks = [
        {
            "step": step,
            "simulated_vs_actual": {
                "relative_update_error": 0.0,
                "difference_l2": 0.0,
                "maximum_absolute_difference": 0.0,
                "exact_equal": True,
                "dtype_mismatches": 0,
            },
            "archived_vs_actual": {},
        }
        for step in (1, 2, 4, 8, 16)
    ]
    validation = folder / "parity.json"
    validation.write_text(
        json.dumps(
            {
                "state": "complete",
                "trajectory_parity": "outside_declared_tolerance",
                "actual_updates": 16,
                "actual_horizon": 1000,
                "actual_loss_errors": [0.0] * 16,
                "parameter_checks": parameter_checks,
            }
        )
    )

    result = MODULE._rio_parity_pilot(
        {
            "parity_pilot": {
                "required": True,
                "folder": str(folder),
                "validation": str(validation),
            }
        }
    )

    assert result["complete"]
    assert result["verdict"] == "outside_declared_tolerance"


def test_rio_defense_reference_normalizes_models_defense_path(tmp_path):
    run = tmp_path / "study"
    defense = run / "cases/rio-k4-defense"
    model_path = defense / "models/defense"

    assert (
        MODULE._rio_defense_folder(
            run, "rio-k4-lr1e-6", {"defense_reference": str(model_path)}
        )
        == defense.resolve()
    )
    assert (
        MODULE._rio_defense_folder(run, "rio-k16", {"source_ref_defense": str(defense)})
        == defense.resolve()
    )


def test_rio_replay_audit_uses_emitted_fidelity_schema(tmp_path):
    replay = tmp_path / "replay-inner.jsonl"
    rows = []
    for outer_step in range(1, 126):
        inner = [{"supervised_tokens": 3} for _ in range(16)]
        rows.append(
            {
                "outer_step": outer_step,
                "attack_max_steps": 16,
                "inner_step_count": 16,
                "inner_steps": inner,
                "supervised_tokens": 48,
                "attack_trace_sha256": "a" * 64,
                "attack_schedule_trace_sha256": "b" * 64,
                "replay_batch_sequence_sha256": "c" * 64,
            }
        )
    replay.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = MODULE._rio_replay_audit(
        replay, 16, require_trace=True, require_schedule=True
    )

    assert result["complete"]
    assert result["inner_steps"] == 2000
    assert result["supervised_tokens"] == 6000


def test_rio_k_case_does_not_require_replay_log(tmp_path):
    defense = tmp_path / "cases/rio-k4-defense"
    defense.mkdir(parents=True)

    result = MODULE._rio_case_replay_audit(
        tmp_path,
        "rio-k4-lr1e-6",
        {"family": "k", "simulation_steps": 4},
        defense,
    )

    assert result is None
    assert not (defense / "models/defense/replay-inner.jsonl").exists()
