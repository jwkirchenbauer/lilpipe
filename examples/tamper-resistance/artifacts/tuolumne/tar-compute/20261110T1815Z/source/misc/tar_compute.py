"""Materialize and validate the controlled TAR compute/intervention study."""

import argparse
import copy
import hashlib
import json
import math
import os
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


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


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
        inner = max(r.get("inner_step_total", 0) for r in state["log_history"])
        assert inner == maximum * case["inner_steps_per_update"], inner
    else:
        assert all(
            math.isfinite(r["loss_after_update"]) and r["source_rows"] for r in records
        )
        if case["gaussian_sigma"] is not None:
            assert (model / "checkpoint-0/model.safetensors").exists()
            noise = json.loads((model / "noise.json").read_text())
            assert math.isfinite(noise["actual_l2_after_cast"])
    summary = {
        "state": "passed",
        "case": case["name"],
        "updates": maximum,
        "model_checkpoints": len(case["save_steps"]),
        "optimizer_checkpoints": case["state_steps"],
        "training_elapsed_seconds": records[-1]["elapsed_seconds"],
    }
    dump(folder / "training-validation.json", summary)
    print(json.dumps(summary))


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
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.run, args.example, args.original, args.baseline)
    elif args.stage == "list":
        study = json.loads((args.run / "study.json").read_text())
        print("\n".join(study["bundles"][args.bundle]))
    else:
        validate_case(args.folder)


if __name__ == "__main__":
    main()
