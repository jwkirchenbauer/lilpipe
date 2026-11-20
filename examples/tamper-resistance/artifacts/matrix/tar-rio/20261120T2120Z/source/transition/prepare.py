"""Stage fresh CUDA measurements around unchanged RIO training artifacts."""

import copy
import hashlib
import json
from pathlib import Path
import shutil


EXAMPLE = Path("/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance")
ORIGIN = EXAMPLE / "artifacts/tuolumne/tar-rio/20261119T1620Z"
TARGET = EXAMPLE / "artifacts/matrix/tar-rio/20261120T2120Z"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link(target, source):
    target.symlink_to(source)


def stage_case(root, name, source, *, attack):
    folder = root / "cases" / name
    folder.mkdir(parents=True)
    if not attack:
        # Defense cases only supply training provenance and immutable states.
        folder.rmdir()
        link(folder, source)
        return
    for directory in ("status", "logs", "cache", "surfaces", "geometry"):
        (folder / directory).mkdir()
    for filename in (
        "case.json",
        "config.yml",
        "runtime.sh",
        "runtime.json",
        "training-validation.json",
        "source-dataset.json",
    ):
        link(folder / filename, source / filename)
    link(folder / "models", source / "models")
    link(folder / "source", TARGET / "source")
    link(folder / "status/train.json", source / "status/train.json")
    settings = read(source / "study.json")
    old_settings = copy.deepcopy(settings)
    settings["cpu_landscape_anchor"] = True
    settings["capability_scores_file"] = str(folder / "evaluation-scores.csv")
    write(folder / "study.json", settings)
    write(
        folder / "measurement-provenance.json",
        {
            "training_folder": str(source),
            "origin_settings_sha256": sha(source / "study.json"),
            "measurement_backend": "CUDA12.6/H100",
            "changed_settings": {
                k: {"old": old_settings.get(k), "new": v}
                for k, v in settings.items()
                if old_settings.get(k) != v
            },
            "models_are_read_only_inputs": True,
        },
    )
    geometry_status = source / "status/geometry.json"
    if geometry_status.is_file() and read(geometry_status).get("state") == "complete":
        for path in (source / "geometry").iterdir():
            if not path.is_file():
                continue
            if path.suffix == ".npy":
                link(folder / "geometry" / path.name, path)
            else:
                shutil.copy2(path, folder / "geometry" / path.name)
        write(
            folder / "status/geometry.json",
            {
                "state": "complete",
                "reused_saved_parameter_geometry": str(source),
                "reason": "Preserve literal directions from the same recorded weights",
            },
        )


def main():
    assert not TARGET.exists(), "Use a fresh target; never mutate submitted source"
    TARGET.mkdir(parents=True)
    for name in ("cases", "source", "logs", "status", "launches", "cache"):
        (TARGET / name).mkdir()
    source = TARGET / "source"
    shutil.copytree(
        ORIGIN / "source/misc",
        source / "misc",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    link(source / "src", ORIGIN / "source/src")
    link(source / "lm_eval_tasks", ORIGIN / "source/lm_eval_tasks")
    overlays = (
        "tar_dynamics.py",
        "tar_compute_evaluation.py",
        "tar_compute_compare.py",
        "run_tar_compute.sh",
        "run_tar_compute_evaluation.sh",
        "run_tar_compute_landscapes.sh",
    )
    for name in overlays:
        shutil.copy2(EXAMPLE / "scripts/misc" / name, source / "misc" / name)
    shutil.copy2(
        ORIGIN / "source/analysis-runtime/trace_runner.py",
        source / "misc/trace_runner.py",
    )
    shutil.copytree(
        Path(__file__).parent,
        source / "transition",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    program = read(ORIGIN / "study.json")
    program["origin_run"] = str(ORIGIN)
    program["measurement_backend"] = "matrix-cuda126"
    program["training_backend_note"] = (
        "Completed K4/K64/fidelity training reused from ROCm; remaining K256 training "
        "moves to CUDA after a four-update placement/numerical pilot. User accepts "
        "expected floating-point variation. Scientific configurations stay fixed."
    )
    for name, case in program["cases"].items():
        stage_case(
            TARGET, name, ORIGIN / "cases" / name, attack=case["role"] == "attack"
        )
    shutil.copy2(ORIGIN / "probes.json", TARGET / "probes.json")
    write(
        TARGET / "status/probes.json",
        {"state": "complete", "reused": str(ORIGIN / "probes.json")},
    )
    # Put the reused control in its own small evaluation study, avoiding an extra
    # experimental condition in the eight-case final RIO comparison.
    control_source = Path(program["comparison_controls"]["rio-k16"]["folder"])
    control_root = TARGET / "control-study"
    (control_root / "cases").mkdir(parents=True)
    for name in ("status", "logs", "cache"):
        (control_root / name).mkdir()
    link(control_root / "source", source)
    shutil.copy2(ORIGIN / "probes.json", control_root / "probes.json")
    control_name = "hparam-rio-lr1e-6"
    stage_case(control_root, control_name, control_source, attack=True)
    control_folder = control_root / "cases" / control_name
    for key in ("folder", "attack"):
        program["comparison_controls"]["rio-k16"][key] = str(control_folder)
    program["comparison_controls"]["rio-k16"]["scores_file"] = str(
        control_folder / "evaluation-scores.csv"
    )
    program["reused_k16_rio_attack"] = str(control_folder)
    source_hashes = {}
    for section in ("misc", "src", "lm_eval_tasks"):
        for path in sorted((source / section).rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
            ):
                source_hashes[str(path.relative_to(TARGET))] = sha(path)
    program["source_sha256"] = source_hashes
    write(TARGET / "study.json", program)
    control_program = copy.deepcopy(program)
    control_program.update(
        cases={control_name: read(control_source / "case.json")},
        evaluation_cases=[control_name],
        profile_reference_case=control_name,
    )
    write(control_root / "study.json", control_program)
    write(
        TARGET / "migration.json",
        {
            "origin": str(ORIGIN),
            "target": str(TARGET),
            "origin_study_sha256": sha(ORIGIN / "study.json"),
            "origin_probes_sha256": sha(ORIGIN / "probes.json"),
            "source_sha256": source_hashes,
            "scientific_hyperparameters_changed": False,
            "cuda_measurements_have_fresh_status_and_result_directories": True,
            "new_control_folder": str(control_folder),
            "production_training_folder": str(ORIGIN / "cases/rio-k256-defense"),
        },
    )
    print(TARGET)


if __name__ == "__main__":
    main()
