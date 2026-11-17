"""Freeze operational profiles and emit downstream launcher dry runs only."""

import argparse
import ast
import hashlib
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


TRAIN = "/usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4"
EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2"
LAUNCHER = "/p/vast1/kirchenb/llnl-tools/launch_tuo.py"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_under(folder):
    return [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]


def checksum_file(path, files):
    path.write_text("".join(f"{digest(item)}  {item}\n" for item in sorted(set(files))))


def freeze(run):
    program = read(run / "study.json")
    original, baseline, compute = (
        Path(program[key])
        for key in ("original_run", "baseline_dynamics", "compute_study")
    )
    example = run.parents[3]
    collaborator = example / "artifacts/tuolumne/tar-collaborator/20261116T1725Z"
    trace_source = collaborator / "source/profiles/analysis-v3"
    validation = {
        "state": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_comparison": {},
        "configurations": {},
        "cache_links": {},
    }
    for relative in (
        "trainers/tr_tar.py",
        "trainers/ul_me.py",
        "trainers/utils.py",
        "axolotl_plugins/custom_trainer_args.py",
        "data/__init__.py",
    ):
        current, before = (
            run / "source/src" / relative,
            baseline / "source/src" / relative,
        )
        same = current.read_bytes() == before.read_bytes()
        assert same, f"Active B trainer/data/plugin source changed: {relative}"
        validation["source_comparison"][relative] = {
            "original": str(before),
            "frozen": str(current),
            "sha256": digest(current),
            "byte_identical": same,
        }
    recorder = run / "source/misc/tar_dynamics_training.py"
    assert (
        recorder.read_bytes()
        == (compute / "source/misc/tar_dynamics_training.py").read_bytes()
    )
    validation["source_comparison"]["recorder"] = {
        "original": str(compute / "source/misc/tar_dynamics_training.py"),
        "frozen": str(recorder),
        "sha256": digest(recorder),
        "byte_identical": True,
    }
    trainer = ast.parse((run / "source/src/trainers/tr_tar.py").read_text())
    base = next(
        node
        for node in trainer.body
        if isinstance(node, ast.ClassDef) and node.name == "TARBaseTrainer"
    )
    init = next(
        node
        for node in base.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    defaults = {
        key.arg: value
        for key, value in zip(init.args.kwonlyargs, init.args.kw_defaults)
    }
    assert (
        isinstance(defaults["sample_attack_fn"], ast.Constant)
        and defaults["sample_attack_fn"].value is None
    )
    validation["optional_attack_branch"] = (
        "sample_attack_fn defaults to None in byte-identical B source; absent from every hparam config. The original AdamW branch remains active."
    )
    numerical = run / "source/misc/tar_dynamics.py"
    assert numerical.read_bytes() == (trace_source / "tar_dynamics.py").read_bytes()
    validation["numerical_source"] = {
        "sha256": digest(numerical),
        "byte_identical_to": str(trace_source / "tar_dynamics.py"),
        "precision": "Original mixed saved trainer weights; unchanged BF16 eager dynamics evaluation. No precision policy was changed.",
    }
    guard_tree = ast.parse((run / "source/misc/tar_compute.py").read_text())
    validator = next(
        node
        for node in guard_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_case"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_hparam_input_order"
        for node in ast.walk(validator)
    )
    validation["post_training_batch_guard"] = {
        "enabled_for_kind": program["kind"],
        "source_sha256": digest(run / "source/misc/tar_compute.py"),
        "required": "125 outer defense batches and 1000 attack batches match frozen step+canonical tensors before passed training validation",
    }
    runtime = {
        "base_model",
        "output_dir",
        "dataset_prepared_path",
        "trainer_cls",
        "wandb_name",
        "wandb_project",
        "save_steps",
        "save_strategy",
        "save_only_model",
        "save_total_limit",
    }
    for name, case in program["cases"].items():
        folder = run / "cases" / name
        config = yaml.safe_load((folder / "config.yml").read_text())
        reference = yaml.safe_load((folder / "config.reference.yml").read_text())
        science = json.loads(json.dumps(config))
        if case["role"] == "defense":
            assert "sample_attack_fn" not in config["custom_trainer_args"]
            values = program["hparam_matrix"][case["condition"]]
            assert config["learning_rate"] == values["outer_learning_rate"]
            assert (
                config["custom_trainer_args"]["retain_coefficient"]
                == values["retain_coefficient"]
            )
            assert (
                config["custom_trainer_args"]["attack_learning_rate"]
                == values["inner_learning_rate"]
            )
            science["learning_rate"] = reference["learning_rate"]
            for key in ("retain_coefficient", "attack_learning_rate"):
                science["custom_trainer_args"][key] = reference["custom_trainer_args"][
                    key
                ]
        assert {key: value for key, value in science.items() if key not in runtime} == {
            key: value for key, value in reference.items() if key not in runtime
        }
        validation["configurations"][name] = {
            "config_sha256": digest(folder / "config.yml"),
            "case_sha256": digest(folder / "case.json"),
            "only_requested_scientific_changes": True,
            "optimizer_state_milestones": case["state_steps"],
        }
    for role, source in program["prepared_cache_sources"].items():
        arrows = list((run / "cache" / role).rglob("*.arrow"))
        assert arrows
        for target in arrows:
            before = Path(source) / target.relative_to(run / "cache" / role)
            assert (target.stat().st_ino, target.stat().st_dev) == (
                before.stat().st_ino,
                before.stat().st_dev,
            )
        validation["cache_links"][role] = {
            "source": source,
            "arrow_files": len(arrows),
            "all_same_inodes": True,
        }
    assert (run / "probes.json").read_bytes() == (compute / "probes.json").read_bytes()
    validation["probes_sha256"] = digest(run / "probes.json")
    source_files = []
    for child in ("src", "misc", "lm_eval_tasks"):
        source_files.extend(files_under(run / "source" / child))
    inputs = [run / "matrix.json", run / "probes.json", *source_files]
    for case in program["cases"].values():
        folder = Path(case["folder"])
        inputs.extend(
            folder / child
            for child in (
                "case.json",
                "config.yml",
                "config.reference.yml",
                "runtime.sh",
            )
        )
        if (folder / "study.json").exists():
            inputs.append(folder / "study.json")
    checksum_file(run / "operational-inputs.sha256", inputs)
    profiles = {}
    for name, worker, environment in (
        ("training-v1", "run_tar_compute.sh", TRAIN),
        ("evaluation-v1", "run_tar_compute_evaluation.sh", EVAL),
        ("landscapes-v1", "run_tar_compute_landscapes.sh", EVAL),
    ):
        profile = run / "source/profiles" / name
        profile.mkdir(parents=True, exist_ok=False)
        target = run / "source/misc" / worker
        if name == "evaluation-v1":
            (profile / "misc").mkdir()
            shutil.copy2(
                example / "scripts/misc/tar_compute_evaluation.py",
                profile / "misc/tar_compute_evaluation.py",
            )
            shutil.copy2(run / "source/misc" / worker, profile / "misc" / worker)
            shutil.copytree(
                run / "source/src",
                profile / "src",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            target = profile / "misc" / worker
        elif name == "landscapes-v1":
            for filename in (
                "tar_dynamics.py",
                "tar_compute.py",
                "tar_compute_compare.py",
                "reference_get_ll.py",
            ):
                shutil.copy2(run / "source/misc" / filename, profile / filename)
            for filename in (worker, "trace_runner.py", "test_trace_payload.py"):
                shutil.copy2(trace_source / filename, profile / filename)
            target = profile / worker
        entry = profile / "run.sh"
        guard = run / "source/planning-v1/runtime_guard.py"
        entry.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
[[ $- == *i* ]]
[[ -n "${{FLUX_JOB_ID:-}}" ]]
profile=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
sha256sum --quiet -c "$profile/manifest.sha256"
sha256sum --quiet -c {shlex.quote(str(run / "operational-inputs.sha256"))}
export PYTHONDONTWRITEBYTECODE=1
mm_activate {shlex.quote(environment)}
python {shlex.quote(str(guard))} {shlex.quote(name.split('-v')[0])} "$@"
exec bash -i {shlex.quote(str(target))} "$@"
"""
        )
        metadata = {
            "entry": str(entry),
            "worker": str(target),
            "environment": environment,
            "canonical_source": str(run / "source"),
            "inputs_manifest": str(run / "operational-inputs.sha256"),
        }
        if name == "landscapes-v1":
            metadata.update(
                numerical_sha256=digest(profile / "tar_dynamics.py"),
                trace_wrapper_sha256=digest(profile / "trace_runner.py"),
                trace_source=str(trace_source),
                trace_note="Only periodic asynchronous stack dumps are suppressed; fatal-fault reporting remains enabled. No root cause is inferred.",
            )
        save(profile / "profile.json", metadata)
        checksum_file(
            profile / "manifest.sha256",
            [*files_under(profile), guard, run / "operational-inputs.sha256"],
        )
        profiles[name] = {
            **metadata,
            "manifest_sha256": digest(profile / "manifest.sha256"),
            "files": {
                str(path.relative_to(run)): digest(path)
                for path in files_under(profile)
            },
        }
        for path in files_under(profile):
            path.chmod(0o444)
    save(
        run / "operational-profiles.json",
        {
            "created_utc": validation["created_utc"],
            "profiles": profiles,
            "inputs_manifest_sha256": digest(run / "operational-inputs.sha256"),
            "note": "Canonical source/study/config files were not changed; parent owns submissions and job records.",
        },
    )
    save(run / "preparation-validation.json", validation)
    plan = []

    defenses = {
        item["case"]: item["job_id"] for item in read(run / "defense-jobs.json")
    }
    attacks = {item["case"]: item["job_id"] for item in read(run / "attack-jobs.json")}

    def add(
        name, mode, arguments, minutes, gpus, requires, qos="pbatch", dependency=None
    ):
        profile = profiles["evaluation-v1" if mode == "evaluation" else "landscapes-v1"]
        target = ["bash", "-i", profile["entry"], *arguments]
        argv = [
            "python",
            LAUNCHER,
            f"--environment={profile['environment']}",
            "--env_act_style=mm_activate",
            f"--output_dir={run / 'launches'}",
            f"--sub_output_dir_name={name}",
            f"--run_name=tar-{name}",
            "--nodes=1",
            "--tasks_per_node=1",
            f"--gpus_per_node={gpus}",
            "--cpus_per_task=96",
            f"--minutes={minutes}",
            f"--qos={qos}",
            "--bank=verival",
            "--torchrun=false",
            "--pass_run_name=false",
            "--quiet_launch=true",
            "--quiet_localization=true",
            "--squonce=false",
            "--rm_core_dumps=false",
            "--dryrun",
            "--",
            *target,
        ]
        if dependency:
            argv.insert(argv.index("--"), f"--dependency=afterok:{dependency}")
        plan.append(
            {
                "name": name,
                "mode": mode,
                "environment": profile["environment"],
                "minutes": minutes,
                "gpu_mask_count": gpus,
                "requires_completed": requires,
                "argv_dryrun": argv,
                "launch_directory": str(run / "launches" / name),
                "dryrun_log": str(run / "logs" / f"dryrun-{name}.log"),
            }
        )

    for name, case in program["cases"].items():
        if case["role"] == "attack":
            add(
                f"eval-{name}",
                "evaluation",
                ["evaluate", str(run), "--cases", name],
                180,
                4,
                [f"train:{name}"],
                dependency=attacks[name],
            )
    for suffix, conditions in (("r-i", ("r", "i")), ("o-rio", ("o", "rio"))):
        cases = [f"hparam-{condition}-lr1e-6" for condition in conditions]
        add(
            f"landscapes-{suffix}",
            "landscapes",
            ["group", str(run), *cases],
            180,
            2,
            [f"train:{case}" for case in cases],
        )
    reference = program["profile_reference_case"]
    add(
        "profiles-r",
        "profiles",
        ["profiles", str(run), reference],
        60,
        1,
        [f"train:{reference}"],
        dependency=attacks[reference],
    )
    add(
        "profiles-i-o",
        "profiles",
        ["profiles-group", str(run), "hparam-i-lr1e-6", "hparam-o-lr1e-6"],
        60,
        2,
        ["profiles-r", "train:hparam-i-lr1e-6", "train:hparam-o-lr1e-6"],
    )
    add(
        "profiles-rio",
        "profiles",
        ["profiles", str(run), "hparam-rio-lr1e-6"],
        60,
        1,
        ["profiles-r", "train:hparam-rio-lr1e-6"],
    )
    add(
        "checkpoint-report",
        "report",
        ["checkpoint-compare", str(run)],
        30,
        1,
        [item["name"] for item in plan],
        "pdebug",
    )
    save(
        run / "operational-dryrun-plan.json",
        {
            "state": "prepared-not-submitted",
            "jobs": plan,
            "evaluation_scope": "Only the four attack cases; step zero evaluates the paired defense endpoint. Standard WMDP milestones and MMLU at0/128/512/1000 remain unchanged. Defense checkpoint histories receive native CE/landscape analysis, not nine rounds of benchmarks.",
            "training": "Four D120 and four dependent A55 jobs were submitted by parent; see defense-jobs.json and attack-jobs.json. Do not regenerate or resubmit training jobs. Status and passed canonical-batch validation are required independently of scheduler dependencies.",
        },
    )
    print("Operational profiles frozen; canonical running training sources unchanged.")


def dryruns(run):
    plan = read(run / "operational-dryrun-plan.json")
    if (
        Path(os.environ.get("CONDA_PREFIX", "/missing")).resolve()
        != Path(EVAL).resolve()
    ):
        raise RuntimeError(
            "Activate the absolute EVAL prefix before downstream dry runs"
        )
    for item in plan["jobs"]:
        directory = Path(item["launch_directory"])
        if (directory / "launch.sh").exists():
            raise FileExistsError(
                f"Do not overwrite existing launcher files: {directory}"
            )
        assert "--dryrun" in item["argv_dryrun"]
        with Path(item["dryrun_log"]).open("x") as stream:
            subprocess.run(
                item["argv_dryrun"], stdout=stream, stderr=subprocess.STDOUT, check=True
            )
        output = Path(item["dryrun_log"]).read_text()
        assert "No jobs were launched." in output
        line = next(
            line for line in output.splitlines() if line.startswith("Submission: ")
        )
        item["submission_argv_after_review"] = shlex.split(
            line.removeprefix("Submission: ")
        )
        for filename in ("launch.sh", "subrunner.sh"):
            path = directory / filename
            subprocess.run(["bash", "-n", str(path)], check=True)
        print(f"DRY RUN ONLY: {item['name']}", flush=True)
    plan["state"] = "dryruns-complete-not-submitted"
    save(run / "operational-dryrun-plan.json", plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "dryruns"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else dryruns)(args.run.resolve())


if __name__ == "__main__":
    main()
