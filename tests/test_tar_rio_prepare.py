"""Preparation contracts for the fixed-outer RIO K study."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import yaml


EXAMPLE = Path(__file__).parents[1] / "examples/tamper-resistance"
SPEC = importlib.util.spec_from_file_location(
    "tar_compute", EXAMPLE / "scripts/misc/tar_compute.py"
)
tar_compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tar_compute)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def write_config(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def observed(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_observations(count, *, defense):
    rows = []
    for step in range(1, count + 1):
        batch = {"input_ids": [[step]], "attention_mask": [[1]], "labels": [[step]]}
        if defense:
            batch["metadata_ids"] = [0, 1, 2]
        digest = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
        rows.append({"step": step, "batch": batch, "batch_sha256": digest})
    return rows


def make_fixture(root):
    example = EXAMPLE
    original = root / "original"
    baseline = root / "baseline"
    compute = root / "compute"
    rio = root / "rio"
    base_model = "/models/zephyr"
    defense_config = {
        "base_model": base_model,
        "max_steps": 125,
        "warmup_steps": 12,
        "learning_rate": 1e-5,
        "seed": 42,
        "data_seed": 42,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "bf16": True,
        "lr_scheduler": "linear",
        "optimizer": "adamw_torch",
        "weight_decay": 0.0,
        "attn_implementation": "eager",
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "remove_unused_columns": False,
        "train_on_inputs": True,
        "tf32": False,
        "gradient_checkpointing": False,
        "max_grad_norm": 1.0,
        "datasets": [
            {"path": "forget", "revision": "forget-rev"},
            {"path": "retain", "revision": "retain-rev"},
        ],
        "micro_batch_size": 24,
        "custom_trainer_args": {
            "forget_coefficient": 1.0,
            "retain_coefficient": 4.0,
            "defence_forget_batch_size": 4,
            "defence_retain_batch_size": 4,
            "attack_learning_rate": 1e-5,
            "attack_forget_batch_size": 16,
            "attack_forget_batch_size_per_iteration": 1,
            "attack_epochs": 1,
        },
    }
    original_config = {"base_model": base_model, "datasets": defense_config["datasets"]}
    write_config(original / "configs/defense.yml", original_config)
    write_config(rio / "cases/hparam-rio-defense/config.yml", defense_config)
    write_json(
        rio / "cases/hparam-rio-defense/case.json",
        {"config": str(rio / "cases/hparam-rio-defense/config.yml")},
    )
    write_json(
        rio / "cases/hparam-rio-defense/training-validation.json", {"state": "passed"}
    )
    (rio / "cases/hparam-rio-defense/models/defense/checkpoint-125").mkdir(parents=True)
    (
        rio / "cases/hparam-rio-defense/models/defense/checkpoint-125/model.safetensors"
    ).touch()
    observed(
        rio / "cases/hparam-rio-defense/models/defense/observed-batches.jsonl",
        make_observations(125, defense=True),
    )

    attack_config = {
        "base_model": base_model,
        "max_steps": 1000,
        "warmup_steps": 0,
        "learning_rate": 1e-6,
        "seed": 1,
        "data_seed": 42,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "bf16": True,
        "lr_scheduler": "linear",
        "weight_decay": 0.0,
        "attn_implementation": "eager",
        "optimizer": "adamw_torch",
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "remove_unused_columns": True,
        "train_on_inputs": True,
        "tf32": False,
        "gradient_checkpointing": False,
        "max_grad_norm": 1.0,
        "datasets": defense_config["datasets"],
    }
    write_config(compute / "cases/baseline-lr1e-6/config.yml", attack_config)
    write_json(
        compute / "cases/baseline-lr1e-6/training-validation.json", {"state": "passed"}
    )
    write_json(
        rio / "cases/hparam-rio-lr1e-6/training-validation.json", {"state": "passed"}
    )
    observed(
        compute / "cases/baseline-lr1e-6/models/attack/observed-batches.jsonl",
        make_observations(1000, defense=False),
    )
    control_attack = make_observations(1000, defense=False)
    for row in control_attack:
        row["learning_rate_used"] = 1e-6
    observed(
        compute / "cases/baseline-lr1e-6/models/attack/observed-batches.jsonl",
        control_attack,
    )
    rio_attack = make_observations(1000, defense=False)
    for row in rio_attack:
        row["learning_rate_used"] = 1e-6
    observed(
        rio / "cases/hparam-rio-lr1e-6/models/attack/observed-batches.jsonl",
        rio_attack,
    )
    write_json(
        compute / "probes.json",
        {
            "fixed_source_indices": list(range(10)),
            "heldout_source_indices": list(range(1000, 1032)),
            "forget_revision": "forget-rev",
        },
    )
    for path in (
        original / "cache/axolotl/defense",
        compute / "cache/attack",
        compute / "source/lm_eval_tasks",
        baseline / "source/misc",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (baseline / "source/misc/reference_get_ll.py").touch()
    return example, original, baseline, compute, rio


def test_prepare_rio_study_emits_k_and_fidelity_cases(tmp_path):
    example, original, baseline, compute, rio = make_fixture(tmp_path)
    run = tmp_path / "prepared"

    tar_compute.prepare_rio_study(run, example, original, baseline, compute, rio)

    study = json.loads((run / "study.json").read_text())
    assert study["kind"] == "rio-k-fidelity-study"
    assert study["primary_k_values"] == [4, 16, 64, 256]
    assert study["fixed_outer_updates"] == 125
    assert study["profile_reference_case"] == "rio-k4-lr1e-6"
    assert study["protocol_document"] == "source/TAR_RIO_K_AND_FIDELITY.md"
    assert (run / study["protocol_document"]).is_file()
    assert (run / "source/misc/tar_inner_recording.py").is_file()
    assert (run / "source/misc/tar_basis_structure.py").is_file()
    assert study["comparison_controls"]["rio-k16"]["schedule_steps"] == 1000
    assert study["comparison_controls"]["rio-k16"]["folder"] == str(
        rio / "cases/hparam-rio-lr1e-6"
    )
    assert study["comparison_controls"]["rio-k16"]["source_ref_defense"] == str(
        rio / "cases/hparam-rio-defense"
    )
    assert study["rio_attack_trace"]["rows"] == 1000
    assert len(study["rio_attack_trace"]["batch_hashes"]) == 1000
    assert len(study["rio_attack_trace"]["learning_rates"]) == 1000
    cases = study["cases"]
    assert len(cases) == 16
    for name, k, epochs in (
        ("rio-k4", 4, 1),
        ("rio-k64", 64, 4),
        ("rio-k256", 256, 16),
    ):
        config = yaml.safe_load(
            (run / "cases" / f"{name}-defense/config.yml").read_text()
        )
        assert config["max_steps"] == 125
        assert config["custom_trainer_args"]["attack_max_steps"] == k
        assert config["custom_trainer_args"]["attack_epochs"] == epochs
        assert config["micro_batch_size"] == 24
        case = json.loads((run / "cases" / f"{name}-defense/case.json").read_text())
        assert case["inner_steps_per_update"] == case["simulation_steps"] == k
        assert case["family"] == "k"
        assert case["simulation_data_policy"] == "fresh"
        attack_case = json.loads(
            (run / "cases" / f"{name}-lr1e-6/case.json").read_text()
        )
        assert attack_case["inner_steps_per_update"] == k
        assert attack_case["simulation_steps"] == attack_case["k"] == k
        assert f"K={k}" in attack_case["label"]
    for name, order, optimizer in (
        ("rio-prefix-forward-legacy", "forward", "legacy"),
        ("rio-prefix-reverse-legacy", "reverse", "legacy"),
        ("rio-fresh-matched", "fresh", "matched"),
        ("rio-prefix-forward-matched", "forward", "matched"),
        ("rio-prefix-reverse-matched", "reverse", "matched"),
    ):
        config = yaml.safe_load(
            (run / "cases" / f"{name}-defense/config.yml").read_text()
        )
        assert config["trainer_cls"].endswith("RecordedReplayTARMETrainer")
        assert config["custom_trainer_args"]["attack_trace_order"] == (
            "forward" if order == "fresh" else order
        )
        assert config["custom_trainer_args"]["attack_optimizer_mode"] == optimizer
        assert config["custom_trainer_args"]["attack_trace_path"] == (
            None if order == "fresh" else str(run / "rio-attack-observed-batches.jsonl")
        )
        assert config["custom_trainer_args"]["attack_schedule_trace_path"] == (
            str(run / "rio-attack-observed-batches.jsonl")
            if optimizer == "matched"
            else None
        )
        case = json.loads((run / "cases" / f"{name}-defense/case.json").read_text())
        assert case["family"] == "fidelity"
        assert "data" in case["label"]
        assert case["simulation_data_policy"] == order
        assert case["simulation_optimizer"] == optimizer
        attack_case = json.loads(
            (run / "cases" / f"{name}-lr1e-6/case.json").read_text()
        )
        assert attack_case["inner_steps_per_update"] == 16
        assert attack_case["simulation_steps"] == attack_case["k"] == 16
        assert "K=16" in attack_case["label"]


def test_rio_input_order_validation_checks_outer_sampler(tmp_path):
    run = tmp_path / "study"
    case_folder = run / "cases/rio-k4-defense"
    case_folder.mkdir(parents=True)
    reference = tmp_path / "rio-observations.jsonl"
    rows = make_observations(125, defense=True)
    observed(reference, rows)
    write_json(
        run / "study.json",
        {
            "kind": "rio-k-fidelity-study",
            "original_rio_defense_observations": str(reference),
            "compute_study": str(tmp_path),
        },
    )
    case = {"role": "defense", "max_steps": 125}

    result = tar_compute.validate_hparam_input_order(case_folder, case, rows)

    assert result["state"] == "passed"
    changed = list(rows)
    changed[-1] = dict(rows[-1])
    changed[-1]["batch"] = {"input_ids": [[99]]}
    with pytest.raises(ValueError, match="Missing canonical defense tensors"):
        tar_compute.validate_hparam_input_order(case_folder, case, changed)


@pytest.mark.parametrize(
    ("policy", "optimizer"),
    [
        ("forward", "legacy"),
        ("reverse", "legacy"),
        ("fresh", "matched"),
        ("forward", "matched"),
        ("reverse", "matched"),
    ],
)
def test_validate_rio_fidelity_inner_records(tmp_path, policy, optimizer):
    run = tmp_path / "study"
    folder = run / "cases/case-defense"
    model = folder / "models/defense"
    model.mkdir(parents=True)
    source = tmp_path / "rio-attack-observed-batches.jsonl"
    trace = []
    for step in range(1, 17):
        batch = {
            "input_ids": [[step, step + 1, step + 2]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[1, -100, step + 2]],
        }
        digest = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
        trace.append(
            {
                "step": step,
                "batch": batch,
                "batch_sha256": digest,
                "learning_rate_used": step / 1e6,
            }
        )
    observed(source, trace)
    trace_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    study = {
        "original_rio_attack_observations": str(source),
        "rio_attack_trace": {"sha256": trace_sha},
    }
    raw_logs = []
    outer_records = []
    ordered_trace = list(trace)
    if policy == "reverse":
        ordered_trace.reverse()
    for outer_step in range(1, 126):
        inner = []
        observed_outer = {
            "batch": {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "metadata_ids": [2] * 16,
            }
        }
        for index in range(16):
            row = ordered_trace[index]
            batch = row["batch"]
            if policy == "fresh":
                inner_batch = {
                    field: [batch[field][0]]
                    for field in ("input_ids", "attention_mask", "labels")
                }
                for field in ("input_ids", "attention_mask", "labels"):
                    observed_outer["batch"][field].append(batch[field][0])
            else:
                inner_batch = batch
            digest = hashlib.sha256(
                json.dumps(inner_batch, sort_keys=True).encode()
            ).hexdigest()
            inner.append(
                {
                    "batch_sha256": digest,
                    "supervised_tokens": 1,
                    "learning_rate_used": (
                        trace[index]["learning_rate_used"]
                        if optimizer == "matched"
                        else 1e-5
                    ),
                }
            )
        outer_records.append(observed_outer)
        raw_logs.append(
            {
                "outer_step": outer_step,
                "inner_step_count": 16,
                "inner_steps": inner,
                "attack_optimizer_mode": optimizer,
                "attack_schedule_trace_sha256": (
                    trace_sha if optimizer == "matched" else None
                ),
                "attack_trace_sha256": trace_sha if policy != "fresh" else None,
                "optimizer": {
                    "weight_decay": 0.0 if optimizer == "matched" else 0.01,
                    "max_grad_norm": 1.0 if optimizer == "matched" else None,
                },
            }
        )
    observed(model / "replay-inner.jsonl", raw_logs)
    case = {
        "simulation_data_policy": policy,
        "simulation_optimizer": optimizer,
    }

    result = tar_compute.validate_rio_replay(folder, case, study, outer_records)

    assert result["state"] == "passed"
    assert result["outer_records"] == 125
