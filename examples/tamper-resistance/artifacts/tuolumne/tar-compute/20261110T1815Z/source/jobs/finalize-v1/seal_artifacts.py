#!/usr/bin/env python3
"""Validate this completed run and seal only reviewable scientific artifacts.

Run through the authorized launcher after study.json has been marked complete.
--list only selects/stats files; it neither validates nor hashes file contents.
Outputs are exclusive-create: artifact-manifest.json and commit-artifacts.txt.
No model/basis/optimizer files or token-bearing JSONL traces are selected.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


sys.dont_write_bytecode = True
LIMIT = 32 * 1024 * 1024
OUTPUTS = {"artifact-manifest.json", "commit-artifacts.txt"}
CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".sbatch",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".json",
    ".txt",
    ".lock",
    ".cfg",
    ".ini",
}
CASE_SUFFIXES = {".png", ".pdf", ".gif", ".csv", ".json", ".yml", ".yaml"}
PRUNE = {
    "__pycache__",
    ".git",
    "cache",
    "caches",
    "logs",
    "models",
    "optimizer",
    "tokenizer",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(run, relative):
    value = PurePosixPath(relative)
    require(
        isinstance(relative, str)
        and relative == value.as_posix()
        and not value.is_absolute()
        and value.parts
        and ".." not in value.parts
        and "\n" not in relative
        and "\r" not in relative,
        f"Not a normalized RUN-relative path: {relative!r}",
    )
    return run.joinpath(*value.parts)


def bounded_stat(run, path):
    relative = path.relative_to(run)
    current = run
    for part in relative.parts:
        current = current / part
        require(
            not stat.S_ISLNK(current.lstat().st_mode), f"Symlink refused: {current}"
        )
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode), f"Not a regular file: {path}")
    require(metadata.st_size <= LIMIT, f"File exceeds 32 MiB: {relative}")
    return metadata


def read_json(run, path):
    bounded_stat(run, path)
    return json.loads(path.read_text())


def read_rows(run, path):
    bounded_stat(run, path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(run, path):
    before = bounded_stat(run, path)
    checksum = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"File changed while hashing: {path}",
    )
    return checksum.hexdigest()


def walk_files(root):
    if root.is_symlink() or not root.is_dir():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in PRUNE and not (base / name).is_symlink()
        )
        for name in sorted(files):
            path = base / name
            if not path.is_symlink() and not name.startswith("."):
                yield path


def select_files(run):
    selected = set()

    def add(path):
        if (
            path.is_symlink()
            or path.name in OUTPUTS
            or path.name.startswith(
                ("core", "samples", "tokenizer", "optimizer", "observed-batches")
            )
        ):
            return
        bounded_stat(run, path)
        selected.add(path)

    for path in run.iterdir():
        if (
            not path.is_symlink()
            and path.is_file()
            and path.suffix in {".md", ".csv", ".json"}
        ):
            add(path)
    for path in walk_files(run / "comparisons"):
        if path.suffix in {".png", ".pdf"}:
            add(path)
    for path in walk_files(run / "source"):
        if path.suffix in CODE_SUFFIXES or path.name == "_default_template_yaml":
            add(path)
    for path in walk_files(run / "status"):
        if path.suffix == ".json":
            add(path)
    require(not (run / "cases").is_symlink(), "Symlink refused: cases")
    for folder in sorted((run / "cases").iterdir()):
        if folder.is_symlink() or not folder.is_dir():
            continue
        for path in folder.iterdir():
            if (
                not path.is_symlink()
                and path.is_file()
                and (path.suffix in CASE_SUFFIXES or path.name == "runtime.sh")
            ):
                add(path)
        for child in ("surfaces", "status"):
            for path in walk_files(folder / child):
                if path.suffix in {".csv", ".json"} or (
                    child == "surfaces" and path.name == "minibatch-losses.npz"
                ):
                    add(path)
        geometry = folder / "geometry/geometry.json"
        if not geometry.is_symlink() and geometry.exists():
            add(geometry)
        for path in walk_files(folder / "evals"):
            if path.name.startswith("results_") and path.suffix == ".json":
                add(path)
    return sorted(selected)


def load_module(run, relative, name):
    path = local_path(run, relative)
    bounded_stat(run, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def integer(value):
    number = float(value)
    require(
        math.isfinite(number) and number.is_integer(), f"Invalid integer: {value!r}"
    )
    return int(number)


def finite(value):
    result = float(value)
    require(math.isfinite(result), f"Nonfinite value: {value!r}")
    return result


def indexed(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(
            integer(row[field]) if field in {"step", "attack_step"} else row[field]
            for field in fields
        )
        require(key not in result, f"{label}: duplicate {key}")
        result[key] = row
    return result


def complete_status(run, folder, stage):
    require(
        read_json(run, folder / f"status/{stage}.json").get("state") == "complete",
        f"{folder.name}: {stage} incomplete",
    )


def check_lattices(compare, rows, fields, expected, coordinates, label, losses):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(
            integer(row[field]) if field == "step" else row[field] for field in fields
        )
        require(key in expected, f"{label}: unexpected group {key}")
        groups[key].append(row)
    require(set(groups) == set(expected), f"{label}: missing groups")
    centers = {}
    for key, group in groups.items():
        axis = coordinates(key)
        for field in losses(key):
            grid = compare.rectangular_grid(group, axis, axis, field)
            require(
                not grid["issues"]
                and grid["measured"] == grid["expected"]
                and len(group) == grid["expected"],
                f"{label} {key} {field}: incomplete/invalid lattice: {grid['issues']}",
            )
        center = [
            row
            for row in group
            if abs(finite(row["c0"])) < 1e-10 and abs(finite(row["c1"])) < 1e-10
        ]
        require(len(center) == 1, f"{label} {key}: missing/duplicate center")
        centers[key] = finite(center[0]["probe_loss"])
    return centers


def validate_surfaces(run, folder, settings, compare):
    import numpy as np

    geom = read_json(run, folder / "geometry/geometry.json")
    actual = indexed(
        read_rows(run, folder / "surfaces/actual-losses.csv"),
        ("phase", "step"),
        folder.name,
    )
    attack_steps = settings.get("actual_attack_steps", settings["attack_steps"])
    expected_actual = {
        ("defense", step) for step in [0] + settings["defense_steps"]
    } | {("attack", step) for step in [0] + attack_steps}
    require(
        set(actual) == expected_actual,
        f"{folder.name}: incomplete actual checkpoint losses",
    )
    for row in actual.values():
        finite(row["probe_loss"])
        finite(row["heldout_loss"])
    fixed = read_rows(run, folder / "surfaces/fixed-grids.csv")
    planes = {(plane,) for plane in settings["planes"]}
    centers = check_lattices(
        compare,
        fixed,
        ("plane",),
        planes,
        lambda key: np.linspace(
            -settings.get("plane_radii", {}).get(key[0], geom["radius"]),
            settings.get("plane_radii", {}).get(key[0], geom["radius"]),
            settings["surface_points"],
        ),
        folder.name + " fixed",
        lambda key: ["probe_loss"]
        + [f"batch_{step}_loss" for step in settings["minibatch_surface_steps"]],
    )
    start_loss = finite(actual[("attack", 0)]["probe_loss"])
    require(
        all(abs(value - start_loss) < 1e-5 for value in centers.values()),
        f"{folder.name}: fixed centers disagree",
    )
    counts = {"fixed": len(fixed)}
    for filename, prefix, resolution, radius in (
        (
            "local-grids.csv",
            "",
            settings["local_surface_points"],
            settings.get("local_radius", geom["radius"] * 0.2),
        ),
        (
            "random-evolution.csv",
            "broad_",
            settings.get("broad_points", 9),
            settings.get("broad_radius", 125),
        ),
    ):
        frames = {("defense", step) for step in settings[prefix + "defense_frames"]} | {
            ("attack", step) for step in settings[prefix + "attack_frames"]
        }
        rows = read_rows(run, folder / "surfaces" / filename)
        centers = check_lattices(
            compare,
            rows,
            ("phase", "step"),
            frames,
            lambda key: np.linspace(-radius, radius, resolution),
            folder.name + " " + filename,
            lambda key: (
                ["probe_loss", "minibatch_loss"]
                if not prefix and key[0] == "attack"
                else ["probe_loss"]
            ),
        )
        require(
            all(
                key in actual and abs(value - finite(actual[key]["probe_loss"])) < 1e-5
                for key, value in centers.items()
            ),
            f"{folder.name}: {filename} centers disagree",
        )
        counts[filename] = len(rows)
    projected = indexed(
        read_rows(run, folder / "surfaces/projected-losses.csv"),
        ("plane", "step"),
        folder.name + " projections",
    )
    require(
        set(projected)
        == {
            (plane, step)
            for plane in settings["planes"]
            for step in [0] + settings["attack_steps"]
        },
        f"{folder.name}: incomplete projections",
    )
    for row in projected.values():
        for field in (
            "c0",
            "c1",
            "distance",
            "captured_fraction",
            "off_plane_distance",
            "projected_loss",
        ):
            finite(row[field])
    matrix_path = folder / "surfaces/minibatch-losses.npz"
    bounded_stat(run, matrix_path)
    with zipfile.ZipFile(matrix_path) as archive:
        require(
            sum(item.file_size for item in archive.infolist()) <= LIMIT,
            f"{folder.name}: expanded matrix exceeds 32 MiB",
        )
    with np.load(matrix_path, allow_pickle=False) as matrix:
        steps = settings.get(
            "matrix_steps",
            [0] + settings.get("actual_attack_steps", settings["attack_steps"]),
        )
        batches = settings.get(
            "matrix_batch_steps",
            list(range(1, settings.get("attack_max_steps", 128) + 1)),
        )
        require(
            np.array_equal(matrix["steps"], steps)
            and np.array_equal(matrix["batch_steps"], batches),
            f"{folder.name}: matrix axes disagree",
        )
        require(
            matrix["losses"].shape == (len(steps), len(batches))
            and np.isfinite(matrix["losses"]).all(),
            f"{folder.name}: invalid minibatch matrix",
        )
    return counts


def validate(run, selected):
    program = read_json(run, run / "study.json")
    require(
        program.get("state") == "complete",
        "Parent must mark study.json complete before sealing",
    )
    cases = {
        name: case
        for name, case in program["cases"].items()
        if not name.startswith("smoke-")
    }
    attacks = {name for name, case in cases.items() if case["role"] == "attack"}
    require(
        len(cases) == 18 and len(attacks) == 16,
        "Expected 18 scientific cases and 16 attacks",
    )
    comparison = read_json(run, run / "comparison-validation.json")
    require(
        comparison.get("coverage") == "complete"
        and comparison.get("complete_scientific_cases") == 18
        and comparison.get("expected_scientific_cases") == 18
        and comparison.get("step_profiles_required") is True,
        "Comparison coverage is not complete 18/18 with profiles required",
    )
    for field in ("comparability", "step_profile_comparability"):
        require(set(comparison.get(field, {})) == attacks, f"Incomplete {field}")
        for name, check in comparison[field].items():
            flags = (
                ("shared_probes_verified", "local_comparable", "broad_comparable")
                if field == "comparability"
                else ("step_profiles_complete", "step_profiles_comparable")
            )
            require(
                not check.get("issues")
                and all(check.get(flag) is True for flag in flags),
                f"{name}: invalid {field}",
            )
    registry = program.get("source_sha256", {})
    require(registry, "Missing source registry")
    selected_set = set(selected)
    for relative, expected in registry.items():
        path = local_path(run, relative)
        require(
            relative.startswith("source/") and path in selected_set,
            f"Unselected/non-source registry entry: {relative}",
        )
        require(
            digest(run, path) == expected,
            f"Registered source SHA256 changed: {relative}",
        )
    compare_source = program["comparison_report_source"]
    require(compare_source in registry, "Comparison source is not registered")
    compare = load_module(run, compare_source, "seal_compare")
    candidates = [key for key in registry if key.endswith("/tar_compute_evaluation.py")]
    require(candidates, "Missing frozen evaluation plan")
    evaluation_source = max(
        candidates, key=lambda key: tuple(int(x) for x in re.findall(r"\d+", key))
    )
    evaluation = load_module(run, evaluation_source, "seal_evaluation")
    groups = evaluation.frozen_groups(run, program)
    result_data = {}
    expected_scores, measured_scores, counts = {}, {}, {}
    root_scores = indexed(
        read_rows(run, run / "evaluation-scores.csv"),
        ("case", "attack_step", "task"),
        "root evaluation scores",
    )
    for name, case in cases.items():
        folder = run / "cases" / name
        require(
            evaluation.load_case(run, program, name)
            == read_json(run, folder / "case.json"),
            f"{name}: case snapshot changed",
        )
        for stage in ("train", "evaluation"):
            complete_status(run, folder, stage)
        local_scores = indexed(
            read_rows(run, folder / "evaluation-scores.csv"),
            ("attack_step", "task"),
            name + " evaluation scores",
        )
        keys = {
            (step, task)
            for step, tasks in evaluation.evaluation_plan(case)
            for task in tasks
        }
        require(set(local_scores) == keys, f"{name}: incomplete evaluation rows")
        for step, tasks in evaluation.evaluation_plan(case):
            complete_status(run, folder, f"evaluation-checkpoint-{step}")
            status = read_json(
                run, folder / f"status/evaluation-checkpoint-{step}.json"
            )
            for task in tasks:
                files = status.get("tasks", {}).get(task, [])
                require(
                    files and all(Path(path) in selected_set for path in files),
                    f"{name} {step}: missing selected raw results for {task}",
                )
                key = (name, step, task)
                score = finite(local_scores[(step, task)]["jwk_percent"])
                require(0 <= score <= 100, f"{key}: invalid score")
                for filename in files:
                    path = Path(filename)
                    if path not in result_data:
                        result_data[path] = read_json(run, path)
                    verified = evaluation.verified_task(
                        result_data[path],
                        task,
                        evaluation.checkpoint_model(folder, case, step),
                        groups[task],
                    )
                    require(
                        abs(verified["jwk_percent"] - score) < 1e-10,
                        f"{key}: score differs from full raw result",
                    )
                expected_scores[key] = score
        if case["role"] != "attack":
            continue
        for stage in (
            "analysis",
            "geometry",
            "surfaces",
            "random-frames",
            "report",
            "step-profiles",
            "step-profile-values",
            "step-profile-report",
        ):
            complete_status(run, folder, stage)
        settings = read_json(run, folder / "study.json")
        signature = read_json(run, folder / "status/analysis-input.json")
        require(
            signature.get("settings_sha256") == digest(run, folder / "study.json"),
            f"{name}: analysis settings changed",
        )
        counts[name] = validate_surfaces(run, folder, settings, compare)
        for filename in (
            "summary.json",
            "center-validation.json",
            "random-landscape-evolution.png",
            "random-landscape-evolution.pdf",
            "random-landscape-evolution.gif",
            "early-update-profiles.png",
            "early-update-profiles.pdf",
            "first-update-landscape.png",
            "first-update-landscape.pdf",
        ):
            path = folder / filename
            require(
                path in selected_set and bounded_stat(run, path).st_size > 0,
                f"{name}: missing rendered output {filename}",
            )
        profile = read_json(run, folder / "surfaces/step-profiles.json")
        require(
            profile.get("state") == "complete"
            and profile.get("source_sha256") in registry.values(),
            f"{name}: incomplete/unregistered profile source",
        )
        protocol = profile["protocol"]
        require(
            profile["identity"]
            == hashlib.sha256(
                json.dumps(
                    protocol, sort_keys=True, separators=(",", ":"), default=str
                ).encode()
            ).hexdigest()
            and protocol.get("settings") == settings
            and protocol.get("source_sha256") == profile["source_sha256"],
            f"{name}: profile protocol changed",
        )
        rows = read_rows(run, folder / "surfaces/step-profiles.csv")
        require(len(rows) == 200, f"{name}: expected 200 profile samples")
        profile_keys = {
            (integer(row["update_step"]), finite(row["fraction"])) for row in rows
        }
        require(
            profile_keys
            == {
                (step, float(t))
                for step in range(1, 9)
                for t in compare.PROFILE_FRACTIONS
            },
            f"{name}: incomplete profile lattice",
        )
        for row in rows:
            for field in ("signed_displacement", "probe_loss", "minibatch_loss"):
                finite(row[field])
            step = integer(row["update_step"])
            fraction = finite(row["fraction"])
            checks = profile["steps"][str(step)]
            require(
                math.isclose(
                    float(row["signed_displacement"]),
                    fraction * finite(checks["update_l2"]),
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                ),
                f"{name}: profile displacement differs",
            )
            if fraction in (0, 1):
                endpoint = checks["actual_before" if fraction == 0 else "actual_after"]
                require(
                    all(
                        abs(finite(row[field]) - finite(endpoint[field])) <= 1e-4
                        for field in ("probe_loss", "minibatch_loss")
                    ),
                    f"{name}: profile endpoint differs",
                )
        grid = read_rows(run, folder / "surfaces/first-update-grid.csv")
        if profile.get("first_update_plane", {}).get("status") == "fitted":
            for field in ("probe_loss", "minibatch_loss"):
                checked = compare.rectangular_grid(
                    grid, compare.PROFILE_C0, compare.PROFILE_C1, field
                )
                require(
                    len(grid) == 289
                    and not checked["issues"]
                    and checked["measured"] == 289,
                    f"{name}: incomplete first-update grid",
                )
                actual = finite(profile["steps"]["1"]["actual_before"][field])
                require(
                    abs(float(checked["values"][8, 8]) - actual) <= 1e-4,
                    f"{name}: first-update grid center differs",
                )
                proof = profile["first_update_plane"]
                require(
                    abs(
                        finite(proof["plane_endpoint_losses"][field])
                        - finite(profile["steps"]["1"]["actual_after"][field])
                    )
                    <= 1e-4,
                    f"{name}: first-update plane endpoint differs",
                )
        else:
            require(not grid, f"{name}: skipped first-update plane has grid data")
    for key, row in root_scores.items():
        require(
            row["role"] == cases[key[0]]["role"], f"{key}: root evaluation role differs"
        )
        measured_scores[key] = finite(row["jwk_percent"])
    require(
        set(root_scores) == set(expected_scores)
        and all(
            abs(measured_scores[key] - score) < 1e-10
            for key, score in expected_scores.items()
        ),
        "Root/per-case evaluation rows disagree",
    )
    for family in compare.FAMILIES:
        for kind in (
            "trajectories",
            "local-reference",
            "local-delta",
            "broad-random",
            "early-update-profiles",
            "first-update-landscape",
        ):
            for suffix in ("png", "pdf"):
                path = run / "comparisons" / f"{family}-{kind}.{suffix}"
                require(
                    path in selected_set and bounded_stat(run, path).st_size > 0,
                    f"Missing comparison figure: {path.name}",
                )
    return {
        "scientific_cases": len(cases),
        "profile_cases": len(attacks),
        "evaluation_task_checkpoints": len(expected_scores),
        "verified_source_files": len(registry),
        "comparison_source": compare_source,
        "evaluation_plan_source": evaluation_source,
        "surface_points": counts,
    }


def publish_new(path, payload):
    require(len(payload) <= LIMIT, f"Output exceeds 32 MiB: {path.name}")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix="." + path.name + ".", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list RUN-relative paths and stat sizes",
    )
    args = parser.parse_args()
    run = Path(__file__).resolve().parents[3]
    selected = select_files(run)
    total = sum(bounded_stat(run, path).st_size for path in selected)
    if args.list:
        for path in selected:
            print(
                f"{bounded_stat(run, path).st_size}\t{path.relative_to(run).as_posix()}"
            )
        print(
            json.dumps(
                {"files": len(selected), "bytes": total, "per_file_limit": LIMIT}
            ),
            file=sys.stderr,
        )
        return
    require(
        not any(os.path.lexists(run / name) for name in OUTPUTS),
        "Seal outputs already exist; refusing to overwrite",
    )
    snapshot = {path: bounded_stat(run, path) for path in selected}
    validation = validate(run, selected)
    entries = []
    for path in selected:
        checksum = digest(run, path)
        after, before = bounded_stat(run, path), snapshot[path]
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
            f"Artifact changed during validation/sealing: {path}",
        )
        entries.append(
            {
                "path": path.relative_to(run).as_posix(),
                "bytes": after.st_size,
                "sha256": checksum,
            }
        )
    require(
        select_files(run) == selected, "Selected artifact set changed while sealing"
    )
    total = sum(entry["bytes"] for entry in entries)
    manifest = {
        "schema_version": 1,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": run.name,
        "per_file_limit_bytes": LIMIT,
        "selected_files": len(entries),
        "selected_bytes": total,
        "validation": validation,
        "excluded": "Weights, bases, optimizer state, tokenizers, caches, logs, core files, samples, all JSONL traces, and symlinks. Manifest/list are excluded from their own hashes.",
        "files": entries,
    }
    commit_paths = sorted([entry["path"] for entry in entries] + list(OUTPUTS))
    publish_new(run / "commit-artifacts.txt", ("\n".join(commit_paths) + "\n").encode())
    publish_new(
        run / "artifact-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()
    )
    print(
        json.dumps(
            {
                "state": "sealed",
                "files": len(entries),
                "bytes": total,
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
