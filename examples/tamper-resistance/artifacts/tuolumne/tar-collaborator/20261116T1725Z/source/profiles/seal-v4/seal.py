"""Verify completed evidence and select small reproducible artifacts for Git."""

import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

run = Path(sys.argv[1]).resolve()
assert os.environ.get("FLUX_JOB_ID"), "Seal through launch_tuo.py"
example = Path.cwd()
repository = example.parents[1]
active_launch = Path(os.environ["OUTPUT_DIR"]).resolve()
assert active_launch.parent == run / "launches"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def publish(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


program = read(run / "study.json")
comparison = module(
    "seal_comparison", run / "source/profiles/analysis-v2/tar_compute_compare.py"
)
evaluation = module("seal_evaluation", run / "source/misc/tar_compute_evaluation.py")
groups = evaluation.frozen_groups(run, program)
probes = read(run / "probes.json")
for filename in (
    "input-validation.json",
    "prepared-data-validation.json",
    "profile-implementation-validation.json",
):
    assert read(run / filename)["state"] == "passed", filename
assert read(run / "static/validation.json")["state"] == "passed"
assert read(run / "status/static.json")["state"] == "complete"
assert read(run / "checkpoint-comparison-validation.json")["coverage"] == "complete"
assert read(run / "status/final-report.json")["state"] == "complete"
assert (run / "checkpoint-report.md").is_file()
assert (run / "figure-index.md").is_file()
for extension in ("png", "pdf"):
    figures = list((run / "checkpoint-comparisons/final").glob("*." + extension))
    assert len(figures) == 10 and all(path.stat().st_size > 1000 for path in figures)
assert (run / "findings.md").is_file()
required_images = (
    "attack-paths.png",
    "surfaces-and-checkpoints.png",
    "projection-fidelity.png",
    "loss-and-capability.png",
    "minibatch-dynamics.png",
    "minibatch-landscapes.png",
    "local-minibatch-landscapes.png",
    "random-landscape-evolution.png",
    "random-landscape-evolution.gif",
    "local-landscape-evolution.gif",
    "attack-landscape-evolution.png",
    "first-update-landscape.png",
    "early-update-profiles.png",
)
counts, endpoints, rejected = {}, {}, []
for name, case in program["cases"].items():
    folder = run / "cases" / name
    case = evaluation.load_case(run, program, name)
    for phase in (
        "train",
        "evaluation",
        "analysis",
        "geometry",
        "surfaces",
        "random-frames",
        "report",
        "step-profiles",
        "step-profile-values",
        "step-profile-report",
    ):
        assert read(folder / "status" / (phase + ".json"))["state"] == "complete", (
            name,
            phase,
        )
    assert read(folder / "training-validation.json")["state"] == "passed"
    planned = evaluation.evaluation_plan(case)
    for step, tasks in planned:
        found, invalid = evaluation.collect_results(
            folder / "evals" / f"checkpoint-{step}",
            evaluation.checkpoint_model(folder, case, step),
            groups,
        )
        assert all(task in found for task in tasks), (name, step)
        rejected.extend(invalid)
    _, _, profile_check = comparison._profile_adapter_record(run, name)
    assert profile_check["step_profiles_comparable"], profile_check
    record = comparison.checkpoint_record(name, folder, case["max_steps"])
    assert not comparison.checkpoint_validation(record, program, probes)["issues"]
    settings = read(folder / "study.json")
    count = {"benchmark_groups": sum(len(tasks) for _, tasks in planned)}
    for filename, expected, fields in (
        (
            "fixed-grids.csv",
            len(settings["planes"]) * settings["surface_points"] ** 2,
            ("plane", "c0", "c1"),
        ),
        (
            "local-grids.csv",
            (len(settings["defense_frames"]) + len(settings["attack_frames"]))
            * settings["local_surface_points"] ** 2,
            ("phase", "step", "c0", "c1"),
        ),
        (
            "random-evolution.csv",
            (
                len(settings["broad_defense_frames"])
                + len(settings["broad_attack_frames"])
            )
            * settings["broad_points"] ** 2,
            ("phase", "step", "c0", "c1"),
        ),
        ("first-update-grid.csv", 17**2, ("c0", "c1")),
    ):
        data = rows(folder / "surfaces" / filename)
        assert len(data) == expected, (name, filename, len(data), expected)
        assert len({tuple(row[field] for field in fields) for row in data}) == expected
        count[filename] = len(data)
    centers = read(folder / "center-validation.json")
    assert centers and all(
        row["count"] > 0 and row["maximum_absolute_difference"] < 1e-5
        for row in centers
    )
    for image in required_images:
        assert (folder / image).stat().st_size > 1000, (name, image)
    observations = [
        json.loads(line)
        for line in (folder / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["step"] for row in observations] == list(
        range(1, case["max_steps"] + 1)
    )
    assert all(
        hashlib.sha256(json.dumps(row["batch"], sort_keys=True).encode()).hexdigest()
        == row["batch_sha256"]
        for row in observations
    )
    order = hashlib.sha256(
        json.dumps([row["batch_sha256"] for row in observations]).encode()
    ).hexdigest()
    assert order == probes["expected_order_sha256_by_horizon"][str(case["max_steps"])]
    shutil.copy2(
        folder / "models/attack/observed-batches.jsonl",
        folder / "observed-batches.jsonl",
    )
    final = folder / "models/attack" / f"checkpoint-{case['max_steps']}"
    state = read(final / "trainer_state.json")
    assert state["global_step"] == case["max_steps"]
    shutil.copy2(final / "trainer_state.json", folder / "terminal-trainer-state.json")
    weights = final / "model.safetensors"
    endpoints[name] = {
        "path": str(weights),
        "size": weights.stat().st_size,
        "sha256": digest(weights),
    }
    counts[name] = count
assert sum(value["benchmark_groups"] for value in counts.values()) == 54
proofs = {}
for relative, expected in program["source_sha256"].items():
    assert digest(run / relative) == expected, relative
for manifest in sorted((run / "source/profiles").rglob("manifest.sha256")):
    for line in manifest.read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert digest(filename.strip()) == expected, filename
    proofs[str(manifest.relative_to(run))] = digest(manifest)
publish(run / "endpoint-checkpoint-identities.json", endpoints)
publish(
    run / "source-validation.json", {"state": "passed", "profile_manifests": proofs}
)
publish(
    run / "completion.json",
    {
        "state": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "cases": counts,
        "benchmark_groups": 54,
        "focus_test_methods_passed": 50,
        "full_training_updates": [128, 1000],
        "profile_cached_measurements_reused_after_adapter_fix": True,
        "grid_records": sum(
            value[filename]
            for value in counts.values()
            for filename in (
                "fixed-grids.csv",
                "local-grids.csv",
                "random-evolution.csv",
                "first-update-grid.csv",
            )
        ),
        "prior_rejected_evaluation_artifacts": rejected,
    },
)
program["state"] = "complete"
publish(run / "study.json", program)
extensions = {
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".png",
    ".pdf",
    ".gif",
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".toml",
    ".lock",
    ".sha256",
    ".npz",
    ".jinja",
}
exclude = {"models", "cache", "__pycache__", "wandb", "renderings-v2", "renderings-v6"}
selected = []
for path in sorted(run.rglob("*")):
    relative = path.relative_to(run)
    if not path.is_file() or any(part in exclude for part in relative.parts):
        continue
    if path.name in {
        "artifact-manifest.sha256",
        "curated-files.txt",
        "curated-pathspec.nul",
    }:
        continue
    if "logs" in relative.parts:
        if path.resolve().is_relative_to(active_launch):
            # Our stdout continues growing after the manifest is written.
            continue
        preserve = path.name in {
            "prepare-tests.log",
            "prepare-tests-v2.log",
            "control-schema-preflight.log",
            "profile-implementation-audit.log",
            "first-profile-validation-128.log",
            "first-profile-validation-1000.log",
            "trace-wrapper-test.log",
            "final-format-check.log",
            "seal-record-preflight.log",
        }
        if not preserve and not (
            relative.parts[0] == "launches" and path.suffix == ".out"
        ):
            continue
    elif path.suffix not in extensions and not str(relative).startswith(
        "source/lm_eval_tasks/"
    ):
        continue
    assert path.stat().st_size < 50_000_000, path
    selected.append(path)
(run / "curated-files.txt").write_text("".join(str(path) + "\n" for path in selected))
manifest = run / "artifact-manifest.sha256"
manifest.write_text(
    "# Verify from examples/tamper-resistance. Bulk model/basis/optimizer files are excluded.\n"
    + "".join(f"{digest(path)}  {path.relative_to(example)}\n" for path in selected)
)
stage = selected + [manifest]
(run / "curated-pathspec.nul").write_bytes(
    b"".join(str(path.relative_to(repository)).encode() + b"\0" for path in stage)
)
print(
    json.dumps(
        {
            "state": "complete",
            "curated_files": len(stage),
            "curated_bytes": sum(path.stat().st_size for path in stage),
            "counts": counts,
        },
        indent=2,
    ),
    flush=True,
)
