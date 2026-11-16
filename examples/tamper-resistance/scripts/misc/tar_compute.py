"""Materialize and validate the controlled TAR compute/intervention study."""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml


DEFENSE = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
RATES = ("1e-5", "1e-6", "5e-6")
ATTACK_STEPS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    112,
    128,
    192,
    256,
    384,
    512,
    768,
    1000,
]
DEFENSE_STEPS = [1, 2, 4, 8, 16, 32, 64, 96, 125]
HPARAM_MATRIX = {
    "R": {
        "retain_coefficient": 4.0,
        "inner_learning_rate": 1e-6,
        "outer_learning_rate": 5e-6,
    },
    "I": {
        "retain_coefficient": 2.5,
        "inner_learning_rate": 1e-5,
        "outer_learning_rate": 5e-6,
    },
    "O": {
        "retain_coefficient": 2.5,
        "inner_learning_rate": 1e-6,
        "outer_learning_rate": 1e-5,
    },
    "RIO": {
        "retain_coefficient": 4.0,
        "inner_learning_rate": 1e-5,
        "outer_learning_rate": 1e-5,
    },
}


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def case_config(
    reference, *, base_model, output, prepared, role, steps, warmup=None, epochs=None
):
    config = copy.deepcopy(reference)
    config.update(
        base_model=str(base_model),
        output_dir=str(output),
        dataset_prepared_path=str(prepared),
        wandb_project="jwk-tar-landscape-compute",
        wandb_name=output.parents[1].name,
        max_steps=steps,
        save_steps=steps,
        save_strategy="steps",
        save_only_model=False,
        save_total_limit=len(ATTACK_STEPS) + 3,
    )
    config["trainer_cls"] = "tar_dynamics_training." + (
        "RecordedTARMETrainer"
        if role == "defense"
        else (
            "RecordedGaussianRelearnTrainer"
            if "gaussian_sigma" in config.get("custom_trainer_args", {})
            else "RecordedRelearnTrainer"
        )
    )
    if warmup is not None:
        config["warmup_steps"] = warmup
    if epochs is not None:
        config["custom_trainer_args"]["attack_epochs"] = epochs
    return config


def prepare(run, example, original, baseline):
    run.mkdir(parents=True, exist_ok=False)
    for name in ("source/misc", "cases", "logs", "status", "cache"):
        (run / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        example / "src",
        run / "source/src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for filename in (
        "tar_compute.py",
        "tar_dynamics.py",
        "tar_dynamics_training.py",
        "run_tar_compute.sh",
    ):
        shutil.copy2(
            example / "scripts/misc" / filename, run / "source/misc" / filename
        )
    shutil.copy2(
        baseline / "source/misc/reference_get_ll.py",
        run / "source/misc/reference_get_ll.py",
    )
    shutil.copytree(original / "source/lm_eval_tasks", run / "source/lm_eval_tasks")
    # Prepared Arrow caches are immutable; hard links preserve the baseline and
    # let new preparation fingerprints live under this study's separate directory.
    for mode, cache in (("defense", "defense"), ("attack", "attack-lr1e-6")):
        shutil.copytree(
            original / "cache/axolotl" / cache,
            run / "cache" / mode,
            copy_function=os.link,
            ignore=shutil.ignore_patterns("*.lock"),
        )
    defense_runtime = yaml.safe_load((original / "configs/defense.yml").read_text())
    revisions = {d["path"]: d["revision"] for d in defense_runtime["datasets"]}
    base = defense_runtime["base_model"]
    cases = {}

    def add(
        name,
        role,
        reference_path,
        defense_path,
        maximum,
        *,
        epochs=None,
        warmup=None,
        bootstrap_seed=None,
        smoke=False,
    ):
        folder = run / "cases" / name
        for child in ("models", "logs", "status"):
            (folder / child).mkdir(parents=True, exist_ok=True)
        reference = yaml.safe_load(reference_path.read_text())
        (folder / "config.reference.yml").write_text(reference_path.read_text())
        output = folder / "models" / ("defense" if role == "defense" else "attack")
        if role == "attack":
            (folder / "models/defense").symlink_to(
                defense_path, target_is_directory=True
            )
        config = case_config(
            reference,
            base_model=base if role == "defense" else defense_path,
            output=output,
            prepared=run / "cache" / role,
            role=role,
            steps=maximum,
            warmup=warmup,
            epochs=epochs,
        )
        for dataset in config["datasets"]:
            dataset["revision"] = revisions[dataset["path"]]
        config["wandb_name"] = name
        (folder / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        saves = (
            [1]
            if smoke
            else (
                DEFENSE_STEPS + [192, 250, 375, 500]
                if role == "defense" and maximum == 500
                else DEFENSE_STEPS if role == "defense" else ATTACK_STEPS
            )
        )
        state_saves = (
            [1]
            if smoke
            else (
                [125, 250, 375, 500]
                if role == "defense" and maximum == 500
                else [32, 64, 125] if role == "defense" else [128, 512, 1000]
            )
        )
        item = {
            "name": name,
            "role": role,
            "folder": str(folder),
            "config": str(folder / "config.yml"),
            "reference_config": str(reference_path),
            "max_steps": maximum,
            "save_steps": saves,
            "state_steps": state_saves,
            "defense_reference": str(defense_path),
            "bootstrap_seed": bootstrap_seed,
            "inner_steps_per_update": (
                16 * config.get("custom_trainer_args", {}).get("attack_epochs", 1)
                if role == "defense"
                else None
            ),
            "attack_learning_rate": (
                config["learning_rate"] if role == "attack" else None
            ),
            "gaussian_sigma": config.get("custom_trainer_args", {}).get(
                "gaussian_sigma"
            ),
            "base_model": base,
        }
        dump(folder / "case.json", item)
        exports = {
            "TAR_CASE_NAME": name,
            "TAR_CASE_ROLE": role,
            "TAR_CASE_CONFIG": str(folder / "config.yml"),
            "DYNAMICS_SAVE_STEPS": ",".join(map(str, saves)),
            "DYNAMICS_STATE_STEPS": ",".join(map(str, state_saves)),
        }
        if role == "attack":
            exports["DYNAMICS_SOURCE_MANIFEST"] = str(folder / "source-dataset.json")
        if bootstrap_seed is not None:
            exports["DYNAMICS_BOOTSTRAP_SEED"] = str(bootstrap_seed)
        (folder / "runtime.sh").write_text(
            "\n".join(
                f"export {key}={shlex.quote(value)}" for key, value in exports.items()
            )
            + "\n"
        )
        cases[name] = item

    defense_ref = example / "configs/stages-wmdp/0_unlearn" / DEFENSE / "selected.yml"
    existing_defense = original / "models/defense"
    add("smoke-k64", "defense", defense_ref, base, 1, epochs=4, warmup=0, smoke=True)
    add("defense-500x16", "defense", defense_ref, base, 500, epochs=1, warmup=48)
    add("defense-125x64", "defense", defense_ref, base, 125, epochs=4, warmup=12)
    bundles = {
        "smoke": ["smoke-k64"],
        "controls": [],
        "long-attacks": [],
        "deep-attacks": [],
        "noise-small": [],
        "noise-large": [],
        "data": [],
    }
    for family, defense_path, bundle in (
        ("baseline", existing_defense, "controls"),
        ("long", run / "cases/defense-500x16/models/defense", "long-attacks"),
        ("deep", run / "cases/defense-125x64/models/defense", "deep-attacks"),
    ):
        for rate in RATES:
            name = f"{family}-lr{rate}"
            reference = (
                example
                / "configs/stages-wmdp/2_relearn"
                / f"{DEFENSE}--relearn-fft-lr{rate}-ms1000/selected.yml"
            )
            add(name, "attack", reference, defense_path, 1000)
            bundles[bundle].append(name)
    for i, sigma in enumerate(("1e-4", "1e-3", "1e-2", "1e-1")):
        name = f"noise-{sigma}"
        reference = (
            example
            / "configs/stages-wmdp/2_relearn"
            / f"{DEFENSE}--relearn-perturb-and-fft-sigma{sigma}-lr1e-5-ms1000/selected.yml"
        )
        add(name, "attack", reference, existing_defense, 1000)
        bundles["noise-small" if i < 2 else "noise-large"].append(name)
    for seed in (101, 102, 103):
        name = f"bootstrap-{seed}"
        reference = (
            example
            / "configs/stages-wmdp/2_relearn"
            / f"{DEFENSE}--relearn-fft-lr1e-5-ms1000/selected.yml"
        )
        add(name, "attack", reference, existing_defense, 1000, bootstrap_seed=seed)
        bundles["data"].append(name)
    program = {
        "state": "prepared",
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=example, text=True
        ).strip(),
        "original_run": str(original),
        "baseline_dynamics": str(baseline),
        "base_model": base,
        "cases": cases,
        "bundles": bundles,
        "jobs": {},
        "defense_schedule": "Linear over each declared horizon; warmup 12/125 = 48/500. The two 8000-inner-update conditions align LR progress by inner-update budget.",
        "attack_schedule": "Named 1000-step linear schedule, no warmup. Prefixes differ from the prior 128-step schedule; compare interventions against the new clean controls.",
        "bootstrap": "Full-size sampling with replacement of the common prepared attack dataset; independent seeds 101,102,103; defense unchanged.",
        "analysis": "Fixed common probes/reference planes plus condition-fitted planes, actual/projection loss errors, actual minibatches, end-defense and attack dynamics.",
        "source_sha256": {
            str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (run / "source").rglob("*")
            if p.is_file()
        },
    }
    dump(run / "study.json", program)
    print(f"Prepared {len(cases)} cases at {run}")


def validate_hparam_input_order(folder, case, records):
    """Require the fixed-sampler study to replay its matching frozen batches."""
    study_path = Path(folder).resolve().parents[1] / "study.json"
    if not study_path.is_file():
        return None
    program = json.loads(study_path.read_text())
    if program.get("kind") != "fixed-sampler-hparam-study":
        return None
    defense = case["role"] == "defense"
    maximum = 125 if defense else 1000
    reference = (
        Path(program["baseline_dynamics"]) / "models/defense/observed-batches.jsonl"
        if defense
        else Path(program["compute_study"])
        / "cases/baseline-lr1e-6/models/attack/observed-batches.jsonl"
    )
    control = [json.loads(line) for line in reference.read_text().splitlines()]
    steps = list(range(1, maximum + 1))
    if (
        case["max_steps"] != maximum
        or [row["step"] for row in control] != steps
        or [row["step"] for row in records] != steps
    ):
        raise ValueError(
            "Hparam input-order validation requires complete matched horizons"
        )
    required = {"input_ids", "attention_mask", "labels"}
    if defense:
        required.add("metadata_ids")
    signature = hashlib.sha256()
    for actual, expected in zip(records, control):
        batches = [actual.get("batch"), expected.get("batch")]
        if any(
            not isinstance(batch, dict) or not required <= batch.keys()
            for batch in batches
        ):
            raise ValueError(
                f"Missing canonical batch tensors at {case['role']} step {actual['step']}"
            )
        canonical = [
            json.dumps(batch, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for batch in batches
        ]
        if canonical[0] != canonical[1]:
            raise ValueError(
                f"Hparam {case['role']} batch/order differs from frozen control at step {actual['step']}"
            )
        signature.update(
            json.dumps(
                [actual["step"], batches[0]],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
    return {
        "state": "passed",
        "reference_observations": str(reference),
        "batches": maximum,
        "canonical_step_batch_sha256": signature.hexdigest(),
    }


def validate_case(folder):
    case = json.loads((folder / "case.json").read_text())
    model = folder / "models" / ("defense" if case["role"] == "defense" else "attack")
    maximum = case["max_steps"]
    state = json.loads((model / f"checkpoint-{maximum}/trainer_state.json").read_text())
    assert state["global_step"] == maximum
    records = [
        json.loads(line)
        for line in (model / "observed-batches.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == list(range(1, maximum + 1))
    assert all(math.isfinite(r["loss_before_update"]) for r in records)
    for step in case["save_steps"]:
        assert (model / f"checkpoint-{step}/model.safetensors").exists()
    for step in case["state_steps"]:
        for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            assert (model / f"checkpoint-{step}" / filename).exists(), (step, filename)
    if case["role"] == "defense":
        inner = [
            int(r["inner_step_total"])
            for r in state["log_history"]
            if "inner_step_total" in r
        ]
        expected = maximum * case["inner_steps_per_update"]
        # The collaborator logs the cumulative counter before incrementing it.
        assert inner == list(range(expected)), (
            len(inner),
            expected,
            inner[-1:] if inner else [],
        )
    else:
        assert all(
            math.isfinite(r["loss_after_update"]) and r["source_rows"] for r in records
        )
        if case["gaussian_sigma"] is not None:
            assert (model / "checkpoint-0/model.safetensors").exists()
            noise = json.loads((model / "noise.json").read_text())
            assert math.isfinite(noise["actual_l2_after_cast"])
    input_order = validate_hparam_input_order(folder, case, records)
    summary = {
        "state": "passed",
        "case": case["name"],
        "updates": maximum,
        "model_checkpoints": len(case["save_steps"]),
        "optimizer_checkpoints": case["state_steps"],
        "training_elapsed_seconds": records[-1]["elapsed_seconds"],
    }
    if input_order is not None:
        summary["input_order_validation"] = input_order
    dump(folder / "training-validation.json", summary)
    print(json.dumps(summary))


def hparam_matrix(matrix=None):
    """Validate named subsets of the three declared two-level factors."""
    matrix = copy.deepcopy(HPARAM_MATRIX if matrix is None else matrix)
    levels = {
        "retain_coefficient": (2.5, 4.0),
        "inner_learning_rate": (1e-6, 1e-5),
        "outer_learning_rate": (5e-6, 1e-5),
    }
    if not isinstance(matrix, dict) or not matrix:
        raise ValueError("The hparam matrix must be a nonempty named JSON object")
    names, configurations = set(), set()
    for name, values in matrix.items():
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
            or name.lower() in names
        ):
            raise ValueError("Hparam names must be unique safe case identifiers")
        names.add(name.lower())
        if not isinstance(values, dict) or set(values) != set(levels):
            raise ValueError(f"{name}: specify exactly {', '.join(levels)}")
        for field, allowed in levels.items():
            value = values[field]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value not in allowed
            ):
                raise ValueError(f"{name}: {field} must be one of {allowed}")
        configuration = tuple(values[field] for field in levels)
        if configuration == (2.5, 1e-6, 5e-6):
            raise ValueError("Reuse the existing B control instead of rerunning it")
        if configuration in configurations:
            raise ValueError("Hparam matrix contains duplicate scientific conditions")
        configurations.add(configuration)
    return matrix


def prepare_hparam_study(run, example, original, baseline, compute, matrix=None):
    """Prepare fixed-sampler contrasts without running training or analysis."""
    matrix = hparam_matrix(matrix)
    run, example, original, baseline, compute = (
        Path(path).resolve() for path in (run, example, original, baseline, compute)
    )
    if run.exists():
        raise FileExistsError("Use a new directory for a new hparam study")
    defense_source = original / "configs/defense.yml"
    control = compute / "cases/baseline-lr1e-6"
    attack_source = control / "config.yml"
    defense = yaml.safe_load(defense_source.read_text())
    attack = yaml.safe_load(attack_source.read_text())
    expected_defense = {
        "max_steps": 125,
        "warmup_steps": 12,
        "learning_rate": 5e-6,
        "seed": 42,
        "data_seed": 42,
        "micro_batch_size": 24,
        "gradient_accumulation_steps": 1,
        "sequence_len": 512,
        "bf16": True,
        "lr_scheduler": "linear",
        "weight_decay": 0.0,
        "attn_implementation": "eager",
        "optimizer": "adamw_torch",
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "remove_unused_columns": False,
        "train_on_inputs": True,
        "tf32": False,
        "gradient_checkpointing": False,
        "max_grad_norm": 1.0,
    }
    expected_inner = {
        "forget_coefficient": 1.0,
        "retain_coefficient": 2.5,
        "defence_forget_batch_size": 4,
        "defence_retain_batch_size": 4,
        "attack_forget_batch_size": 16,
        "attack_forget_batch_size_per_iteration": 1,
        "attack_epochs": 1,
        "attack_learning_rate": 1e-6,
    }
    expected_attack = {
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
    }
    for label, actual, expected in (
        ("frozen defense", defense, expected_defense),
        ("frozen inner attack", defense.get("custom_trainer_args", {}), expected_inner),
        ("completed attack control", attack, expected_attack),
    ):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"The {label} differs from the original BF16 K16 protocol")
    if not Path(defense["base_model"]).is_absolute():
        raise ValueError("The frozen defense must pin an absolute base-model snapshot")
    revisions = {
        dataset["path"]: dataset.get("revision") for dataset in defense["datasets"]
    }
    if not all(revisions.values()) or any(
        dataset.get("revision") != revisions.get(dataset["path"])
        for dataset in defense["datasets"] + attack["datasets"]
    ):
        raise ValueError("Frozen defense and attack dataset revisions must agree")
    if (
        json.loads((control / "training-validation.json").read_text()).get("state")
        != "passed"
    ):
        raise ValueError("The reused B attack must have passed training validation")
    probes = json.loads((compute / "probes.json").read_text())
    if (
        len(probes.get("fixed_source_indices", [])) != 10
        or len(probes.get("heldout_source_indices", [])) != 32
        or probes.get("forget_revision") != revisions.get("cais/wmdp-bio-forget-corpus")
    ):
        raise ValueError(
            "The existing fixed10 and attack-held-out32 probes are required"
        )
    records = [
        json.loads(line)
        for line in (control / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    order = hashlib.sha256(
        json.dumps([row["batch_sha256"] for row in records]).encode()
    ).hexdigest()
    if [row["step"] for row in records] != list(range(1, 1001)) or order != probes.get(
        "expected_order_sha256_by_horizon", {}
    ).get("1000", probes.get("expected_nonbootstrap_order_sha256")):
        raise ValueError("The reused B attack order must match the frozen probes")

    run.mkdir(parents=True)
    for child in ("source/misc", "cases", "logs", "status", "cache"):
        (run / child).mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        example / "src",
        run / "source/src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for filename in (
        "tar_compute.py",
        "tar_dynamics.py",
        "tar_dynamics_training.py",
        "tar_compute_compare.py",
        "tar_compute_evaluation.py",
        "run_tar_compute.sh",
        "run_tar_compute_landscapes.sh",
        "run_tar_compute_evaluation.sh",
    ):
        shutil.copy2(
            example / "scripts/misc" / filename, run / "source/misc" / filename
        )
    shutil.copy2(
        baseline / "source/misc/reference_get_ll.py",
        run / "source/misc/reference_get_ll.py",
    )
    shutil.copytree(compute / "source/lm_eval_tasks", run / "source/lm_eval_tasks")
    cache_sources = {
        "defense": original / "cache/axolotl/defense",
        "attack": compute / "cache/attack",
    }
    for role, source in cache_sources.items():
        shutil.copytree(
            source,
            run / "cache" / role,
            copy_function=os.link,
            ignore=shutil.ignore_patterns("*.lock"),
        )
    shutil.copy2(compute / "probes.json", run / "probes.json")
    dump(run / "matrix.json", matrix)
    cases = {}
    program = {
        "state": "prepared",
        "kind": "fixed-sampler-hparam-study",
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=example, text=True
        ).strip(),
        "original_run": str(original),
        "baseline_dynamics": str(baseline),
        "compute_study": str(compute),
        "base_model": defense["base_model"],
        "hparam_matrix": matrix,
        "matrix_file": str(run / "matrix.json"),
        "matrix_sha256": hashlib.sha256((run / "matrix.json").read_bytes()).hexdigest(),
        "comparison_controls": {
            "reproduced-1000": {
                "folder": str(control),
                "schedule_steps": 1000,
                "label": "B: retain 2.5, inner 1e-6, outer 5e-6",
            }
        },
        "checkpoint_contrast_note": "Controlled hyperparameter contrasts in the original BF16 pathway with the local metadata sampler: retain coefficient 2.5 or 4, inner LR 1e-6 or 1e-5, and outer LR 5e-6 or 1e-5. Explicit case labels and matrix.json identify each condition. The existing B control is reused. All conditions retain K16, 125 defense updates, and matched LR1e-6 attacks over 1000 updates. Joint changes can interact; this is one defense/attacker seed. The uploaded checkpoint's legacy OPTML data pipeline is not a factorial condition and is not reproduced by changing these three values alone.",
        "checkpoint_report_title": "Controlled TAR hyperparameter comparison",
        "defense_history_note": "Each new defender has nine saved checkpoints across its 125 defense updates. The analysis uses this observed defense history and the subsequent independent attack. The original B control reuses its completed saved trajectory.",
        "defense_schedule": "125 linear updates, warmup 12, 16 simulated inner steps per update (2000 total); seed/data seed 42/42. Inner AdamW resets each outer update and retains its original default weight decay 0.01.",
        "attack_schedule": "Fresh independent LR1e-6 1000-update linear attacks, no warmup, seed/data seed 1/42, and the reused control's exact prepared data order.",
        "analysis": "Exact frozen fixed10 and attack-held-out32, original physical random/reference bases, and each new defense's real saved history.",
        "shared_probes": str(run / "probes.json"),
        "prepared_cache_sources": {
            role: str(path) for role, path in cache_sources.items()
        },
        "cases": cases,
        "bundles": {"defenses": [], "attacks": [], "all": []},
        "jobs": {},
    }
    for condition, values in matrix.items():
        stem = f"hparam-{condition.lower()}"
        defended = run / "cases" / f"{stem}-defense/models/defense"
        label = f"{condition}: retain {values['retain_coefficient']:g}, inner {values['inner_learning_rate']:g}, outer {values['outer_learning_rate']:g}"
        program["bundles"][stem] = []
        for role, reference, source, maximum, saves, state_saves in (
            ("defense", defense, defense_source, 125, DEFENSE_STEPS, [32, 64, 125]),
            ("attack", attack, attack_source, 1000, ATTACK_STEPS, [128, 512, 1000]),
        ):
            name = f"{stem}-defense" if role == "defense" else f"{stem}-lr1e-6"
            folder = run / "cases" / name
            for child in ("models", "logs", "status", "geometry", "surfaces", "cache"):
                (folder / child).mkdir(parents=True)
            (folder / "source").symlink_to(run / "source", target_is_directory=True)
            if role == "attack":
                (folder / "models/defense").symlink_to(
                    defended, target_is_directory=True
                )
            config = case_config(
                reference,
                base_model=defense["base_model"] if role == "defense" else defended,
                output=folder / "models" / role,
                prepared=run / "cache" / role,
                role=role,
                steps=maximum,
            )
            if role == "defense":
                config["learning_rate"] = values["outer_learning_rate"]
                config["custom_trainer_args"]["retain_coefficient"] = values[
                    "retain_coefficient"
                ]
                config["custom_trainer_args"]["attack_learning_rate"] = values[
                    "inner_learning_rate"
                ]
            config["wandb_name"] = name
            config["wandb_project"] = "jwk-tar-fixed-sampler-hparams"
            shutil.copy2(source, folder / "config.reference.yml")
            (folder / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
            case = {
                "name": name,
                "label": label,
                "condition": condition,
                "defense_hyperparameters": values,
                "role": role,
                "folder": str(folder),
                "config": str(folder / "config.yml"),
                "reference_config": str(source),
                "max_steps": maximum,
                "save_steps": list(saves),
                "state_steps": state_saves,
                "defense_reference": (
                    defense["base_model"] if role == "defense" else str(defended)
                ),
                "bootstrap_seed": None,
                "inner_steps_per_update": 16 if role == "defense" else None,
                "attack_learning_rate": (
                    None if role == "defense" else config["learning_rate"]
                ),
                "gaussian_sigma": None,
                "base_model": defense["base_model"],
            }
            cases[name] = case
            program["bundles"]["defenses" if role == "defense" else "attacks"].append(
                name
            )
            program["bundles"]["all"].append(name)
            program["bundles"][stem].append(name)
            dump(folder / "case.json", case)
            exports = {
                "TAR_CASE_NAME": name,
                "TAR_CASE_ROLE": role,
                "TAR_CASE_CONFIG": str(folder / "config.yml"),
                "DYNAMICS_SAVE_STEPS": ",".join(map(str, saves)),
                "DYNAMICS_STATE_STEPS": ",".join(map(str, state_saves)),
            }
            if role == "attack":
                exports["DYNAMICS_SOURCE_MANIFEST"] = str(
                    folder / "source-dataset.json"
                )
                settings = analysis_settings(run, program, case, probes)
                settings.update(
                    defense_checkpoint_root=str(defended),
                    defense_steps=list(DEFENSE_STEPS),
                    defense_frames=[0, 8, 16, 32, 64, 125],
                    broad_defense_frames=[0, 8, 32, 125],
                )
                dump(folder / "study.json", settings)
            (folder / "runtime.sh").write_text(
                "\n".join(
                    f"export {key}={shlex.quote(value)}"
                    for key, value in exports.items()
                )
                + "\n"
            )
    program["profile_reference_case"] = program["bundles"]["attacks"][0]
    program["source_sha256"] = {
        str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (run / "source").rglob("*")
        if path.is_file()
    }
    dump(run / "study.json", program)
    print(f"Prepared {len(matrix)} fresh defense/attack pairs; reused B at {control}")


def prepare_checkpoint_comparison(
    run, example, original, baseline, compute, checkpoint, *, model_id, revision
):
    """Reuse the original attacks and physical reference frames for a supplied model."""
    if not checkpoint.is_absolute() or not (checkpoint / "model.safetensors").is_file():
        raise ValueError("A downloaded absolute checkpoint with weights is required")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Pin the supplied checkpoint to a full Hub commit")
    if (run / "study.json").exists():
        raise FileExistsError("Use a new directory for a new comparison")
    run.mkdir(parents=True, exist_ok=True)
    for name in ("source/misc", "cases", "logs", "status", "cache"):
        (run / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        example / "src",
        run / "source/src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for filename in (
        "tar_compute.py",
        "tar_dynamics.py",
        "tar_dynamics_training.py",
        "tar_compute_compare.py",
        "tar_compute_evaluation.py",
        "run_tar_compute.sh",
        "run_tar_compute_landscapes.sh",
        "run_tar_compute_evaluation.sh",
    ):
        shutil.copy2(
            example / "scripts/misc" / filename, run / "source/misc" / filename
        )
    shutil.copy2(
        baseline / "source/misc/reference_get_ll.py",
        run / "source/misc/reference_get_ll.py",
    )
    shutil.copytree(compute / "source/lm_eval_tasks", run / "source/lm_eval_tasks")
    shutil.copytree(
        compute / "cache/attack",
        run / "cache/attack",
        copy_function=os.link,
        ignore=shutil.ignore_patterns("*.lock"),
    )
    probes = json.loads((compute / "probes.json").read_text())
    probes["expected_order_sha256_by_horizon"] = {}
    for maximum, control in (
        (128, baseline),
        (1000, compute / "cases/baseline-lr1e-6"),
    ):
        records = [
            json.loads(line)
            for line in (control / "models/attack/observed-batches.jsonl")
            .read_text()
            .splitlines()
        ]
        if [row["step"] for row in records] != list(range(1, maximum + 1)):
            raise ValueError(f"Incomplete attack-order control for horizon {maximum}")
        probes["expected_order_sha256_by_horizon"][str(maximum)] = hashlib.sha256(
            json.dumps([row["batch_sha256"] for row in records]).encode()
        ).hexdigest()
    dump(run / "probes.json", probes)
    original_config = yaml.safe_load((original / "configs/defense.yml").read_text())
    revisions = {d["path"]: d["revision"] for d in original_config["datasets"]}
    cases = {}
    program = {
        "state": "prepared",
        "kind": "supplied-checkpoint-comparison",
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=example, text=True
        ).strip(),
        "original_run": str(original),
        "baseline_dynamics": str(baseline),
        "base_model": original_config["base_model"],
        "supplied_checkpoint": {
            "model_id": model_id,
            "revision": revision,
            "path": str(checkpoint),
        },
        "comparison_controls": {
            "reproduced-1000": {
                "folder": str(compute / "cases/baseline-lr1e-6"),
                "schedule_steps": 1000,
            },
            "reproduced-128": {
                "folder": str(baseline),
                "schedule_steps": 128,
                "fixed_source_indices": probes["fixed_source_indices"],
                "forget_revision": probes["forget_revision"],
                "scores_file": str(baseline / "capability-scores.csv"),
                "provenance_note": "The legacy fixed probe is the first ten qualifying documents at the pinned original corpus revision; verified against its source and document hashes.",
            },
        },
        "shared_probes": str(run / "probes.json"),
        "profile_reference_case": "collaborator-lr1e-6-ms1000",
        "cases": cases,
        "bundles": {"attacks": []},
        "jobs": {},
        "attack_schedule": "Independent named LR1e-6 linear schedules over 128 and 1000 updates; no warmup. Their prefixes are different experiments.",
        "analysis": "Same compute-study fixed/held-out probes and physical random/reference bases; only the supplied defense endpoint is available. No defense history is inferred.",
    }
    for maximum in (128, 1000):
        name = f"collaborator-lr1e-6-ms{maximum}"
        folder = run / "cases" / name
        for child in (
            "models",
            "models/defense-history",
            "logs",
            "status",
            "geometry",
            "surfaces",
            "cache",
        ):
            (folder / child).mkdir(parents=True, exist_ok=True)
        (folder / "models/defense").symlink_to(checkpoint, target_is_directory=True)
        (folder / "models/defense-history/checkpoint-125").symlink_to(
            checkpoint, target_is_directory=True
        )
        (folder / "source").symlink_to(run / "source", target_is_directory=True)
        reference_path = (
            example
            / "configs/stages-wmdp/2_relearn"
            / f"{DEFENSE}--relearn-fft-lr1e-6-ms{maximum}/selected.yml"
        )
        reference = yaml.safe_load(reference_path.read_text())
        shutil.copy2(reference_path, folder / "config.reference.yml")
        config = case_config(
            reference,
            base_model=checkpoint,
            output=folder / "models/attack",
            prepared=run / "cache/attack",
            role="attack",
            steps=maximum,
        )
        for dataset in config["datasets"]:
            dataset["revision"] = revisions[dataset["path"]]
        config["wandb_name"] = name
        config["wandb_project"] = "jwk-tar-collaborator-checkpoint"
        (folder / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        saves = [step for step in ATTACK_STEPS if step <= maximum]
        state_saves = [step for step in (128, 512, 1000) if step <= maximum]
        case = {
            "name": name,
            "role": "attack",
            "folder": str(folder),
            "config": str(folder / "config.yml"),
            "reference_config": str(reference_path),
            "max_steps": maximum,
            "save_steps": saves,
            "state_steps": state_saves,
            "defense_reference": str(checkpoint),
            "bootstrap_seed": None,
            "inner_steps_per_update": None,
            "attack_learning_rate": config["learning_rate"],
            "gaussian_sigma": None,
            "base_model": program["base_model"],
            "end_defense_only": True,
        }
        cases[name] = case
        program["bundles"]["attacks"].append(name)
        dump(folder / "case.json", case)
        exports = {
            "TAR_CASE_NAME": name,
            "TAR_CASE_ROLE": "attack",
            "TAR_CASE_CONFIG": str(folder / "config.yml"),
            "DYNAMICS_SAVE_STEPS": ",".join(map(str, saves)),
            "DYNAMICS_STATE_STEPS": ",".join(map(str, state_saves)),
            "DYNAMICS_SOURCE_MANIFEST": str(folder / "source-dataset.json"),
        }
        (folder / "runtime.sh").write_text(
            "\n".join(
                f"export {key}={shlex.quote(value)}" for key, value in exports.items()
            )
            + "\n"
        )
        settings = analysis_settings(run, program, case, probes)
        if maximum == 128:
            settings["attack_steps"] = json.loads(
                (baseline / "study.json").read_text()
            )["attack_steps"]
        dump(folder / "study.json", settings)
    program["source_sha256"] = {
        str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (run / "source").rglob("*")
        if p.is_file()
    }
    dump(run / "study.json", program)
    print(f"Prepared two independent original-protocol attacks at {run}")


def checkpoint_file_identity(checkpoint, metadata, *, model_id, revision):
    """Check downloaded bytes against the pinned Hub blob declarations."""
    if metadata.get("sha") != revision or metadata.get("id") != model_id:
        raise ValueError("Hub metadata does not match the declared checkpoint revision")
    expected = {
        item["rfilename"]: item.get("lfs", {})
        for item in metadata["siblings"]
        if item["rfilename"].endswith(".safetensors")
    }
    if not expected:
        raise ValueError("Pinned Hub metadata has no weight blobs")
    identities = {}
    for filename, declaration in expected.items():
        if Path(filename).name != filename:
            raise ValueError("Only checkpoint-root weight files are supported")
        path = checkpoint / filename
        if path.stat().st_size != declaration.get("size"):
            raise ValueError(f"Downloaded weight size differs: {filename}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != declaration.get("sha256"):
            raise ValueError(f"Downloaded weight hash differs: {filename}")
        identities[filename] = {"size": path.stat().st_size, "sha256": digest}
    return identities


def verify_checkpoint_comparison(run):
    """Validate model provenance, parameter layout, and exact probe token IDs."""
    if not os.environ.get("FLUX_JOB_ID"):
        raise ValueError("Verify checkpoint bytes through launch_tuo.py")
    import torch
    from datasets import load_dataset
    from safetensors import safe_open
    from transformers import AutoTokenizer

    program = json.loads((run / "study.json").read_text())
    supplied = program["supplied_checkpoint"]
    checkpoint = Path(supplied["path"])
    metadata = json.loads((run / "input/hub-metadata.json").read_text())
    weights = checkpoint_file_identity(
        checkpoint,
        metadata,
        model_id=supplied["model_id"],
        revision=supplied["revision"],
    )
    reference = json.loads(
        (Path(program["baseline_dynamics"]) / "geometry/geometry.json").read_text()
    )
    shapes, dtype_scalars = {}, {}
    for filename in weights:
        with safe_open(checkpoint / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_slice(name)
                shape, dtype = tensor.get_shape(), tensor.get_dtype()
                shapes[name] = shape
                dtype_scalars[dtype] = dtype_scalars.get(dtype, 0) + math.prod(shape)
    if shapes != reference["shapes"]:
        raise ValueError(
            "Supplied checkpoint parameter layout differs from fixed bases"
        )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    base_tokenizer = AutoTokenizer.from_pretrained(
        program["base_model"], local_files_only=True
    )
    if tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise ValueError("Supplied checkpoint has different token-to-ID assignments")
    probes = json.loads((run / "probes.json").read_text())
    corpus = load_dataset(
        "cais/wmdp-bio-forget-corpus", split="train", revision=probes["forget_revision"]
    )
    tokens = []
    for index in probes["fixed_source_indices"] + probes["heldout_source_indices"]:
        text = corpus[index]["text"]
        actual = tokenizer(text, max_length=512, truncation=True)
        expected = base_tokenizer(text, max_length=512, truncation=True)
        if actual != expected:
            raise ValueError(f"Probe tokenization differs at source row {index}")
        tokens.append(
            {
                "source_index": index,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(actual["input_ids"], separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    for filename in (
        "training_config.yml",
        "eval_results.json",
        "config.json",
        "tokenizer_config.json",
        "README.md",
    ):
        shutil.copy2(checkpoint / filename, run / "input" / filename)
    published = yaml.safe_load((checkpoint / "training_config.yml").read_text())
    original = yaml.safe_load(
        (Path(program["original_run"]) / "configs/defense.yml").read_text()
    )
    scientific_keys = (
        "trainer_cls",
        "custom_trainer_args",
        "datasets",
        "micro_batch_size",
        "max_steps",
        "learning_rate",
        "warmup_steps",
        "seed",
        "data_seed",
        "sequence_len",
        "bf16",
    )
    result = {
        "state": "passed",
        "checkpoint": supplied,
        "weight_files": weights,
        "parameter_count": sum(dtype_scalars.values()),
        "stored_dtype_scalars": dtype_scalars,
        "parameter_layout_matches_shared_bases": True,
        "vocabulary_identical": True,
        "probe_tokenization_identical": True,
        "probe_token_identities": tokens,
        "defense_config_comparison": {
            key: {"supplied": published.get(key), "reproduction": original.get(key)}
            for key in scientific_keys
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "visible_gpus": torch.cuda.device_count(),
        "flux_job_id": os.environ["FLUX_JOB_ID"],
        "interpretation": "A multi-factor defense-checkpoint contrast under fixed attacks and probes; no intermediate collaborator defense states are available.",
    }
    if result["visible_gpus"] != 4:
        raise ValueError("Expected one outer process with all four GPUs visible")
    dump(run / "input-validation.json", result)
    print(json.dumps(result, indent=2), flush=True)


def matrix_positions(maximum, count=128):
    if maximum < 1 or count < 1:
        raise ValueError("Matrix timeline and sample count must be positive")
    selected = set(range(1, min(maximum, 16) + 1)) | {
        step for step in (32, 128, maximum) if step <= maximum
    }
    while len(selected) < min(maximum, count):
        ordered = sorted(selected)
        left, right = max(
            zip(ordered[:-1], ordered[1:]), key=lambda pair: pair[1] - pair[0]
        )
        selected.add((left + right) // 2)
    return sorted(selected)


def analysis_settings(run, program, case, probes):
    folder = Path(case["folder"])
    baseline = Path(program["baseline_dynamics"])
    family = case["name"].split("-lr")[0]
    final_defense = 500 if family == "long" else 125
    defense_root = (
        baseline / "models/defense"
        if family == "baseline" or family not in ("long", "deep")
        else folder / "models/defense"
    )
    primary = case["name"] in ("baseline-lr1e-6", "long-lr1e-6", "deep-lr1e-6")
    available = (
        [1, 2, 4, 8, 16, 32, 64, 96, 125, 192, 250, 375, 500]
        if family == "long"
        else DEFENSE_STEPS
    )
    defense_steps = available if primary else [final_defense]
    defense_frames = (
        (
            [0, 8, 16, 32, 64, 125, 250, 500]
            if family == "long"
            else [0, 8, 16, 32, 64, 125]
        )
        if primary
        else [final_defense]
    )
    broad_defense = (
        ([0, 32, 125, 250, 500] if family == "long" else [0, 8, 32, 125])
        if primary
        else [final_defense]
    )
    geometric = [
        step
        for step in (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1000)
        if step in case["save_steps"]
    ]
    settings = {
        "case_name": case["name"],
        "original_run": program["original_run"],
        "base_model": program["base_model"],
        "attack_anchor": str(
            folder / "models/attack/checkpoint-0"
            if case["gaussian_sigma"] is not None
            else folder / "models/defense"
        ),
        "defense_checkpoint_root": str(defense_root),
        "defense_steps": defense_steps,
        "defense_frames": defense_frames,
        "attack_steps": geometric,
        "actual_attack_steps": case["save_steps"],
        "attack_max_steps": case["max_steps"],
        "attack_frames": [0, 1, 8, 32, 128, 512, 1000],
        "attack_learning_rate": str(case["attack_learning_rate"]),
        "compare_original_weights": False,
        "allow_degenerate": True,
        "minimum_radius": 1e-3,
        "shared_random_basis": str(baseline / "geometry/random-basis.npy"),
        "reference_bases": {"reference": str(baseline / "geometry/attack-basis.npy")},
        "planes": ["random", "reference", "attack"],
        "plane_radii": {"random": 1.0, "reference": 1.0},
        "local_plane": "reference",
        "local_plane_label": "shared baseline attack directions",
        "local_radius": 0.15713339880469646,
        "surface_points": 17,
        "local_surface_points": 9,
        "direction_seed": 42,
        "broad_radius": 125,
        "broad_points": 9,
        "broad_defense_frames": broad_defense,
        "broad_attack_frames": [0, 128, 1000],
        "probe_documents": 10,
        "heldout_documents": 32,
        "fixed_source_indices": probes["fixed_source_indices"],
        "heldout_source_indices": probes["heldout_source_indices"],
        "forget_revision": probes["forget_revision"],
        "minibatch_surface_steps": [1, 32, 128, 1000],
        "matrix_steps": [0, 1, 8, 32, 128, 512, 1000],
        "matrix_batch_steps": matrix_positions(case["max_steps"]),
        "minibatch_comparison_steps": [0, 32, 128, 1000],
        "path_label_steps": [0, 1, 8, 32, 128, 512, 1000],
        "capability_scores_file": str(folder / "evaluation-scores.csv"),
    }
    if case.get("end_defense_only"):
        settings.update(
            defense_checkpoint_root=str(folder / "models/defense-history"),
            defense_steps=[125],
            defense_frames=[125],
            broad_defense_frames=[125],
        )
    for key in (
        "attack_frames",
        "broad_attack_frames",
        "minibatch_surface_steps",
        "matrix_steps",
        "minibatch_comparison_steps",
        "path_label_steps",
    ):
        settings[key] = [step for step in settings[key] if step <= case["max_steps"]]
    return settings


def prepare_analysis(run):
    """Freeze probes after all four distinct attacker-data cohorts are observed."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    program = json.loads((run / "study.json").read_text())
    cohorts = ["baseline-lr1e-5", "bootstrap-101", "bootstrap-102", "bootstrap-103"]
    for name in cohorts:
        status = json.loads((run / "cases" / name / "status/train.json").read_text())
        if status["state"] != "complete":
            raise ValueError(f"Wait for attacker-data cohort {name} to finish")

    def token_signature(tokens):
        return hashlib.sha256(
            json.dumps(tokens, separators=(",", ":")).encode()
        ).hexdigest()

    seen, orders = set(), {}
    for name, case in program["cases"].items():
        if case["role"] != "attack" or name.startswith("smoke-"):
            continue
        status_path = run / "cases" / name / "status/train.json"
        if (
            not status_path.exists()
            or json.loads(status_path.read_text())["state"] != "complete"
        ):
            continue
        records = [
            json.loads(line)
            for line in (run / "cases" / name / "models/attack/observed-batches.jsonl")
            .read_text()
            .splitlines()
        ]
        assert (
            len(records) == case["max_steps"]
        ), f"Incomplete observation trace: {name}"
        orders[name] = hashlib.sha256(
            json.dumps([r["batch_sha256"] for r in records]).encode()
        ).hexdigest()
        for record in records:
            batch = record["batch"]
            tokens = [
                token
                for token, keep in zip(
                    batch["input_ids"][0], batch["attention_mask"][0]
                )
                if keep
            ]
            seen.add(token_signature(tokens))
    clean_order = orders["baseline-lr1e-5"]
    for name, order in orders.items():
        if not name.startswith("bootstrap-") and order != clean_order:
            raise ValueError(
                f"Non-bootstrap case {name} has a different attack data order"
            )
    config = yaml.safe_load((run / "cases/baseline-lr1e-5/config.yml").read_text())
    revision = config["datasets"][0]["revision"]
    corpus = load_dataset(
        "cais/wmdp-bio-forget-corpus", split="train", revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(program["base_model"])
    fixed = []
    for index in range(len(corpus)):
        text = corpus[index]["text"]
        if isinstance(text, str) and len(text) > 50:
            fixed.append(index)
            tokens = tokenizer(text, max_length=512, truncation=True)["input_ids"]
            seen.add(token_signature(tokens))
            if len(fixed) == 10:
                break
    probe_path = run / "probes.json"
    if probe_path.exists():
        probes = json.loads(probe_path.read_text())
        assert (
            probes["fixed_source_indices"] == fixed
            and probes["forget_revision"] == revision
        )
        assert not any(value in seen for value in probes["heldout_token_sha256"])
    else:
        heldout, hashes, token_hashes = [], [], []
        candidates = list(range(1000, len(corpus))) + list(
            range(min(1000, len(corpus)))
        )
        for index in candidates:
            text = corpus[index]["text"]
            if not isinstance(text, str) or len(text) <= 50:
                continue
            tokens = tokenizer(text, max_length=512, truncation=True)["input_ids"]
            signature = token_signature(tokens)
            if signature in seen:
                continue
            seen.add(signature)
            heldout.append(index)
            hashes.append(hashlib.sha256(text.encode()).hexdigest())
            token_hashes.append(signature)
            if len(heldout) == 32:
                break
        assert (
            len(heldout) == 32
        ), "Insufficient unseen forget documents for common held-out probes"
        probes = {
            "forget_revision": revision,
            "fixed_source_indices": fixed,
            "heldout_source_indices": heldout,
            "heldout_sha256": hashes,
            "heldout_token_sha256": token_hashes,
            "observed_cohort_order_sha256": orders,
            "expected_nonbootstrap_order_sha256": clean_order,
        }
        dump(probe_path, probes)
    for name, case in program["cases"].items():
        if case["role"] != "attack" or name.startswith("smoke-"):
            continue
        folder = run / "cases" / name
        for child in ("geometry", "surfaces", "cache"):
            (folder / child).mkdir(exist_ok=True)
        if not (folder / "source").exists():
            (folder / "source").symlink_to(run / "source", target_is_directory=True)
        dump(folder / "study.json", analysis_settings(run, program, case, probes))
    program["shared_probes"] = str(probe_path)
    dump(run / "study.json", program)
    print("Frozen common probes and materialized sixteen landscape analyses")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run", type=Path)
    p.add_argument("--example", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p = sub.add_parser("list")
    p.add_argument("run", type=Path)
    p.add_argument("bundle")
    p = sub.add_parser("validate")
    p.add_argument("folder", type=Path)
    p = sub.add_parser("prepare-analysis")
    p.add_argument("run", type=Path)
    p = sub.add_parser("prepare-checkpoint-comparison")
    p.add_argument("run", type=Path)
    for name in ("example", "original", "baseline", "compute", "checkpoint"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--revision", required=True)
    p = sub.add_parser("prepare-hparam-study")
    p.add_argument("run", type=Path)
    for name in ("example", "original", "baseline", "compute"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument(
        "--matrix",
        type=Path,
        help="JSON file mapping condition names to the three factor values",
    )
    p = sub.add_parser("verify-checkpoint-comparison")
    p.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.example, args.original, args.baseline)
    elif args.stage == "list":
        study = json.loads((args.run / "study.json").read_text())
        print("\n".join(study["bundles"][args.bundle]))
    elif args.stage == "prepare-analysis":
        prepare_analysis(args.run)
    elif args.stage == "prepare-checkpoint-comparison":
        prepare_checkpoint_comparison(
            args.run,
            args.example,
            args.original,
            args.baseline,
            args.compute,
            args.checkpoint,
            model_id=args.model_id,
            revision=args.revision,
        )
    elif args.stage == "verify-checkpoint-comparison":
        verify_checkpoint_comparison(args.run)
    elif args.stage == "prepare-hparam-study":
        prepare_hparam_study(
            args.run,
            args.example,
            args.original,
            args.baseline,
            args.compute,
            json.loads(args.matrix.read_text()) if args.matrix is not None else None,
        )
    else:
        validate_case(args.folder)


if __name__ == "__main__":
    main()
