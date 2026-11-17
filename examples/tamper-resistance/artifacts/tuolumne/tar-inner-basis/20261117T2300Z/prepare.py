"""Freeze an observation-only replay of the completed local B defense."""

import hashlib
import json
import shutil
from pathlib import Path

import yaml


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    example = Path.cwd().resolve()
    run = Path(__file__).resolve().parent
    assert example.name == "tamper-resistance"
    baseline = (
        example
        / "artifacts/tuolumne/tar-k16-repro/20261109T1830Z/dynamics/baseline-k16-20261110T0400Z"
    )
    compute = example / "artifacts/tuolumne/tar-compute/20261110T1815Z"
    actual = compute / "cases/baseline-lr1e-6"
    source = run / "source"
    assert not (run / "study.json").exists(), "Preparation is one-time"
    shutil.copytree(
        baseline / "source/src",
        source / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(example / "src/trainers/tr_tar.py", source / "src/trainers/tr_tar.py")
    for name in (
        "tar_dynamics_training.py",
        "tar_basis_structure.py",
        "tar_compute.py",
        "tar_inner_recording.py",
        "tar_inner_geometry.py",
    ):
        target = source / "misc" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example / "scripts/misc" / name, target)
    for name in ("test_tar_inner_recording.py", "test_tar_inner_geometry.py"):
        target = source / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example.parents[1] / "tests" / name, target)
    case_folder = run / "cases/baseline-inner-recording"
    case_folder.mkdir(parents=True)
    original_config = baseline / "configs/defense.yml"
    config = yaml.safe_load(original_config.read_text())
    changed = dict(config)
    changed.update(
        trainer_cls="tar_inner_recording.RecordedInnerTARMETrainer",
        output_dir=str(case_folder / "models/defense"),
        wandb_name="baseline-inner-basis-observation-20261117T2300Z",
    )
    allowed = {"trainer_cls", "output_dir", "wandb_name"}
    assert {key: value for key, value in changed.items() if key not in allowed} == {
        key: value for key, value in config.items() if key not in allowed
    }
    (case_folder / "config.yml").write_text(yaml.safe_dump(changed, sort_keys=False))
    shutil.copy2(original_config, case_folder / "config.reference.yml")
    saves = [1, 2, 4, 8, 16, 32, 64, 96, 125]
    outer_calls = [1, 32, 64, 96, 125]
    inner_steps = [1, 2, 4, 8, 12, 16]
    actual_settings = json.loads((actual / "study.json").read_text())
    reference_steps = actual_settings["actual_attack_steps"]
    reference_anchor = actual_settings["attack_anchor"]
    references = [
        {
            "name": f"actual_{step}",
            "anchor": reference_anchor,
            "path": str(actual / f"models/attack/checkpoint-{step}"),
        }
        for step in reference_steps
    ]
    assert all(Path(item["path"]).is_dir() for item in references)
    reference_spec = {"version": 1, "references": references}
    dump(run / "references.json", reference_spec)
    case = {
        "name": "baseline-inner-recording",
        "role": "defense",
        "folder": str(case_folder),
        "config": str(case_folder / "config.yml"),
        "max_steps": 125,
        "save_steps": saves,
        "state_steps": [32, 64, 96, 125],
        "inner_steps_per_update": 16,
        "defense_reference": config["base_model"],
    }
    dump(case_folder / "case.json", case)
    dump(
        run / "analysis-spec.json",
        {
            "version": 1,
            "aggregates": [
                str(run / f"inner/outer-{step:06d}.json") for step in outer_calls
            ],
            "actual_views": {
                "matched": [
                    {"name": f"actual_{step}", "step": step} for step in inner_steps
                ],
                "long": [
                    {"name": f"actual_{step}", "step": step}
                    for step in actual_settings["attack_steps"]
                ],
            },
            "actual_states": [
                {"name": f"actual_{step}", "step": step} for step in reference_steps
            ],
        },
    )
    dump(
        run / "study.json",
        {
            "kind": "fixed-sampler-hparam-study",
            "purpose": "observation-only inner-versus-existing-actual trajectory comparison",
            "baseline_dynamics": str(baseline),
            "compute_study": str(compute),
            "base_model": config["base_model"],
            "cases": {case["name"]: case},
            "outer_calls": outer_calls,
            "inner_steps": inner_steps,
            "existing_actual_case": str(actual),
            "new_attacks": [],
            "allowed_config_changes": sorted(allowed),
            "recording": "Transient inner iterates only; persist aggregate Grams/cross-Grams, not inner checkpoints, deltas or basis arrays.",
            "anchor_convention": "Inner call t starts at outer state t-1; the archived actual attack starts at final outer state125. Report anchor-gap norms; no same-state causal claim.",
            "precision": "Raw saved-parameter and BF16-cast endpoint displacements reported separately.",
        },
    )
    dump(
        run / "preparation-validation.json",
        {
            "state": "passed",
            "checks": [
                "same pinned base checkpoint",
                "original B hyperparameters/cache/device map/seeds/schedule unchanged",
                "only observer class and output labels changed",
                "all existing reference checkpoints present",
            ],
            "reference_config_sha256": hashlib.sha256(
                original_config.read_bytes()
            ).hexdigest(),
        },
    )
    paths = [
        run / "prepare.py",
        run / "references.json",
        run / "analysis-spec.json",
        run / "study.json",
        case_folder / "config.yml",
        case_folder / "config.reference.yml",
        case_folder / "case.json",
    ]
    paths += sorted(path for path in source.rglob("*") if path.is_file())
    (run / "source-manifest.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in paths
        )
    )
    print(
        json.dumps(
            {
                "state": "prepared",
                "run": str(run),
                "reference_checkpoints": len(references),
                "source_files": len(paths),
            }
        )
    )


if __name__ == "__main__":
    main()
