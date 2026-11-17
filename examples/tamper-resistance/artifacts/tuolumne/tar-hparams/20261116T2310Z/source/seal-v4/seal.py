"""Gate, curate and hash the completed main and factorial TAR studies."""

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse


LOSS_MATRIX_PATHS = frozenset(
    f"cases/hparam-{condition}-lr1e-6/surfaces/minibatch-losses.npz"
    for condition in ("r", "i", "o", "rio")
)
LOSS_MATRIX_STEPS = [0, 1, 8, 32, 128, 512, 1000]
LOSS_MATRIX_MAXIMUM_BYTES = 65_536


class GateError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise GateError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def allowed_loss_matrix(path, root, config):
    if root is None or config is None:
        return False
    policy = config.get("curated_minibatch_loss_matrices", {})
    require(
        set(policy.get("relative_paths", [])) == LOSS_MATRIX_PATHS
        and len(policy["relative_paths"]) == len(LOSS_MATRIX_PATHS),
        "Loss-matrix allowlist must name exactly the four main attack matrices",
    )
    path, root = Path(path), Path(root)
    return (
        path.is_relative_to(root)
        and path.relative_to(root).as_posix() in LOSS_MATRIX_PATHS
    )


def validate_loss_matrix(path, root, config):
    """Read only the four bounded CE matrices, never arbitrary NumPy payloads."""
    import numpy as np

    path, root = Path(path), Path(root)
    require(
        allowed_loss_matrix(path, root, config), f"NPZ path is not allowlisted: {path}"
    )
    policy = config["curated_minibatch_loss_matrices"]
    maximum = policy["maximum_file_bytes"]
    unpacked_maximum = policy["maximum_uncompressed_bytes"]
    require(
        0 < maximum <= LOSS_MATRIX_MAXIMUM_BYTES
        and 0 < unpacked_maximum <= LOSS_MATRIX_MAXIMUM_BYTES,
        "Loss-matrix byte bounds cannot exceed 64 KiB",
    )
    require(
        path.is_file()
        and not path.is_symlink()
        and path.resolve() == path.absolute()
        and 0 < path.stat().st_size <= maximum,
        f"Missing/symlink/large loss matrix: {path}",
    )
    original_identity = identity(path)
    with path.open("rb") as stream:
        archive_bytes = stream.read(maximum + 1)
    require(
        0 < len(archive_bytes) <= maximum
        and len(archive_bytes) == original_identity[2],
        "Loss archive changed or exceeded its read bound",
    )
    settings_path = path.parents[1] / "study.json"
    settings = read(settings_path)
    steps, batch_steps = settings.get("matrix_steps"), settings.get(
        "matrix_batch_steps"
    )
    require(
        steps == LOSS_MATRIX_STEPS
        and all(type(step) is int for step in steps)
        and settings.get("attack_max_steps", 1000) == 1000,
        "Unexpected loss-matrix model-state axis",
    )
    require(
        isinstance(batch_steps, list)
        and 1 <= len(batch_steps) <= 256
        and all(type(step) is int and 1 <= step <= 1000 for step in batch_steps)
        and batch_steps == sorted(set(batch_steps))
        and batch_steps[0] == 1
        and batch_steps[-1] == 1000,
        "Invalid loss-matrix batch axis",
    )
    expected = {
        "losses.npy": ((len(steps), len(batch_steps)), np.dtype("float64")),
        "steps.npy": ((len(steps),), np.dtype("int64")),
        "batch_steps.npy": ((len(batch_steps),), np.dtype("int64")),
    }
    arrays = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            require(
                len(members) == 3
                and {member.filename for member in members} == set(expected),
                "Loss archive must contain only losses/steps/batch_steps",
            )
            unpacked = sum(member.file_size for member in members)
            require(
                unpacked <= unpacked_maximum
                and all(
                    0 < member.file_size <= unpacked_maximum
                    and not member.flag_bits & 1
                    and member.compress_type
                    in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    for member in members
                ),
                "Large/encrypted/unsupported loss archive payload",
            )
            for member in members:
                with archive.open(member) as stream:
                    payload = stream.read(unpacked_maximum + 1)
                require(
                    len(payload) == member.file_size
                    and len(payload) <= unpacked_maximum,
                    "Loss archive exceeds bounded payload",
                )
                stream = io.BytesIO(payload)
                version = np.lib.format.read_magic(stream)
                require(version in {(1, 0), (2, 0)}, "Unsupported NPY header version")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream, max_header_size=4096)
                expected_shape, expected_dtype = expected[member.filename]
                require(
                    shape == expected_shape
                    and not fortran
                    and dtype == expected_dtype
                    and not dtype.hasobject,
                    f"Wrong loss-matrix shape/dtype: {member.filename}",
                )
                count = math.prod(shape)
                require(
                    stream.tell() + count * dtype.itemsize == len(payload),
                    "Loss NPY data length differs from its canonical header",
                )
                arrays[member.filename] = np.frombuffer(
                    payload, dtype=dtype, count=count, offset=stream.tell()
                ).reshape(shape)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        if isinstance(error, GateError):
            raise
        raise GateError(f"Invalid bounded loss archive: {path}: {error}") from error
    losses = arrays["losses.npy"]
    require(
        np.array_equal(arrays["steps.npy"], steps)
        and np.array_equal(arrays["batch_steps.npy"], batch_steps),
        "Loss-matrix axes differ from frozen case settings",
    )
    require(
        np.isfinite(losses).all() and (losses >= 0).all(),
        "Loss matrix contains nonfinite/negative CE",
    )
    require(
        identity(path) == original_identity, "Loss archive changed during validation"
    )
    return {
        "path": path.relative_to(root).as_posix(),
        "shape": list(losses.shape),
        "dtype": str(losses.dtype),
        "file_bytes": len(archive_bytes),
        "uncompressed_bytes": unpacked,
        "steps": steps,
        "batch_steps": batch_steps,
        "settings_sha256": digest(settings_path),
        "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "minimum_ce": float(losses.min()),
        "maximum_ce": float(losses.max()),
    }


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state(path, expected="complete"):
    value = read(path)
    require(
        value.get("state") == expected,
        f"{path}: expected {expected}, found {value.get('state')}",
    )
    return value


def load_module(path, name):
    path = Path(path)
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = original_path


def batch_signature(path, maximum, role):
    """Hash actual canonical step+batch tensors; ignore stored hashes/metrics."""
    signature = hashlib.sha256()
    required = {"input_ids", "attention_mask", "labels"}
    if role == "defense":
        required.add("metadata_ids")
    count = 0
    with Path(path).open() as stream:
        for count, line in enumerate(stream, 1):
            record = json.loads(line)
            batch = record.get("batch")
            require(
                record.get("step") == count
                and isinstance(batch, dict)
                and required <= batch.keys(),
                f"{path}: invalid canonical batch at row {count}",
            )
            signature.update(
                json.dumps(
                    [count, batch],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
    require(count == maximum, f"{path}: expected {maximum} batches, found {count}")
    return signature.hexdigest()


def training_gate(folder, case, reference, cache):
    role, maximum = case["role"], case["max_steps"]
    require(
        maximum == (125 if role == "defense" else 1000),
        f"{folder}: unexpected training horizon",
    )
    state(folder / "status/train.json")
    proof = state(folder / "training-validation.json", "passed")
    require(
        proof.get("updates") == maximum
        and proof.get("model_checkpoints") == len(case["save_steps"])
        and sorted(proof.get("optimizer_checkpoints", []))
        == sorted(case["state_steps"]),
        f"{folder}: incomplete checkpoint/optimizer coverage",
    )
    order = proof.get("input_order_validation", {})
    require(
        order.get("state") == "passed"
        and order.get("batches") == maximum
        and Path(order.get("reference_observations", "/missing")).resolve()
        == reference.resolve(),
        f"{folder}: exact input-order proof missing or references wrong baseline",
    )
    key = (str(reference), maximum, role)
    if key not in cache:
        cache[key] = batch_signature(reference, maximum, role)
    observed = folder / "models" / role / "observed-batches.jsonl"
    actual = batch_signature(observed, maximum, role)
    require(
        actual == order.get("canonical_step_batch_sha256") == cache[key],
        f"{folder}: saved batches/order no longer match passed baseline proof",
    )
    return {
        "role": role,
        "updates": maximum,
        "canonical_step_batch_sha256": actual,
        "reference": str(reference),
        "proof_sha256": digest(folder / "training-validation.json"),
    }


def verify_small_manifest(manifest, maximum, fallback=None):
    result = []
    for line in Path(manifest).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, filename = line.split(maxsplit=1)
        filename = filename.strip().removeprefix("*")
        path = Path(filename)
        if not path.is_absolute():
            path = Path(manifest).parent / path
            if not path.exists() and fallback is not None:
                path = Path(fallback) / filename
        require(
            path.is_file()
            and path.stat().st_size <= maximum
            and path.suffix
            not in {".npy", ".npz", ".safetensors", ".bin", ".pt", ".pth", ".arrow"},
            f"Refuse missing/large/tensor source-manifest member: {path}",
        )
        actual = digest(path)
        require(actual == expected, f"Frozen manifest mismatch: {path}")
        result.append({"path": str(path), "sha256": actual})
    require(result, f"Empty source manifest: {manifest}")
    return {"manifest_sha256": digest(manifest), "members": result}


def source_gate(studies, maximum):
    proofs = {}
    for root in studies:
        program = read(root / "study.json")
        for relative, expected in program.get("source_sha256", {}).items():
            path = root / relative
            require(
                path.is_file()
                and path.stat().st_size <= maximum
                and digest(path) == expected,
                f"Declared frozen source changed: {path}",
            )
        manifests = {
            path
            for path in (root / "source").rglob("*")
            if path.name in {"manifest.sha256", "SHA256SUMS", "source-checksums.sha256"}
        }
        manifests |= {
            root / name
            for name in ("operational-inputs.sha256", "manifest.sha256")
            if (root / name).exists()
        }
        require(manifests, f"No frozen manifests in {root}")
        for path in sorted(manifests):
            proofs[str(path)] = verify_small_manifest(path, maximum, root)
    return proofs


def evaluation_gate(hr, program, evaluator):
    attacks = [
        name for name, case in program["cases"].items() if case["role"] == "attack"
    ]
    require(
        set(evaluator.select_cases(program)) == set(attacks),
        "Evaluation report must cover exactly the four attack cases",
    )
    groups = evaluator.frozen_groups(hr, program)
    coverage = read(hr / "evaluation-verification.json")["coverage"]
    require(
        {row["case"] for row in coverage} == set(attacks)
        and all(
            row.get("complete")
            and row.get("measured_task_checkpoints")
            == row.get("expected_task_checkpoints")
            for row in coverage
        ),
        "Root evaluation coverage is incomplete or includes unplanned defense benchmarks",
    )
    required_files, result = set(), {}
    for name in attacks:
        folder, case = hr / "cases" / name, program["cases"][name]
        scores = rows(folder / "evaluation-scores.csv")
        indexed = {
            (int(row["attack_step"]), row["task"]): float(row["jwk_percent"])
            for row in scores
        }
        expected = {
            (step, task)
            for step, tasks in evaluator.evaluation_plan(case)
            for task in tasks
        }
        require(
            len(indexed) == len(scores) and set(indexed) == expected,
            f"{name}: score CSV coverage differs from declared milestones",
        )
        for step, tasks in evaluator.evaluation_plan(case):
            marker = state(folder / "status" / f"evaluation-checkpoint-{step}.json")
            require(
                marker.get("step") == step
                and set(tasks) <= set(marker.get("tasks", {})),
                f"{name}: checkpoint evaluation status is incomplete at {step}",
            )
            found, _ = evaluator.collect_results(
                folder / "evals" / f"checkpoint-{step}",
                evaluator.checkpoint_model(folder, case, step),
                groups,
            )
            for task in tasks:
                require(
                    task in found
                    and math.isclose(
                        indexed[(step, task)],
                        found[task]["jwk_percent"],
                        rel_tol=0,
                        abs_tol=1e-10,
                    ),
                    f"{name}: raw full-task result missing or disagrees with CSV at {step}/{task}",
                )
                required_files.update(
                    Path(path).resolve() for path in found[task]["files"]
                )
        result[name] = {"task_checkpoints": len(expected), "complete": True}
    return result, required_files


def checkpoint_gate(hr, program, compare):
    probes = read(hr / "probes.json")
    records, profiles = {}, {}
    for name, control in program["comparison_controls"].items():
        records[name] = compare.checkpoint_record(
            name, control["folder"], control["schedule_steps"], control
        )
    for name, case in program["cases"].items():
        if case["role"] != "attack":
            continue
        records[name] = compare.checkpoint_record(
            name, hr / "cases" / name, case["max_steps"]
        )
        profile, profile_program, validation = compare._profile_adapter_record(hr, name)
        require(
            validation.get("step_profiles_complete")
            and validation.get("step_profiles_comparable")
            and not validation.get("issues"),
            f"{name}: profile cache is incomplete/incompatible: {validation.get('issues')}",
        )
        compare._profile_source_path(
            profile_program, profile["step_profiles"]["source_sha256"]
        )
        profiles[name] = validation
    canonical = {}
    for name, record in records.items():
        validation = compare.checkpoint_validation(record, program, probes)
        require(
            not validation.get("issues")
            and validation.get("physical_comparable")
            and validation.get("heldout_matches_root"),
            f"{name}: canonical numerical coverage failed: {validation.get('issues')}",
        )
        canonical[name] = validation
    saved = read(hr / "checkpoint-comparison-validation.json")
    require(
        saved.get("coverage") == "complete"
        and set(saved.get("cases", {})) == set(records),
        "Final checkpoint report is absent/partial or has incorrect cases",
    )
    for name, validation in saved["cases"].items():
        require(
            not validation.get("issues")
            and validation.get("physical_comparable")
            and validation.get("heldout_matches_root"),
            f"{name}: saved checkpoint report validation failed",
        )
    expected_pairs = compare.checkpoint_pairs(
        records,
        [name for name in records if name in profiles],
        list(program["comparison_controls"]),
    )
    require(
        {row["name"] for row in saved.get("pairs", [])}
        == {row["name"] for row in expected_pairs},
        "Schedule-matched checkpoint report pairs are incomplete",
    )
    required = set()
    for pair in expected_pairs:
        for kind in (
            "local-reference",
            "local-delta",
            "broad-random",
            "broad-delta",
            "trajectories",
        ):
            for extension in ("png", "pdf"):
                required.add(
                    hr
                    / "checkpoint-comparisons/final"
                    / f"{pair['name']}-{kind}.{extension}"
                )
    return {"canonical": canonical, "profiles": profiles}, required


def aggregate_gate(output):
    proof = state(output / "aggregate-validation.json")
    require(
        set(proof.get("conditions", [])) == {"B", "R", "I", "O", "RIO"}
        and proof.get("numeric_validation_rows") == 5,
        "Five-condition aggregate coverage is incomplete",
    )
    coverage = rows(output / "numeric-validation-coverage.csv")
    require(
        len(coverage) == 5
        and {row["condition"] for row in coverage} == {"B", "R", "I", "O", "RIO"}
        and all(
            int(row["issues"]) == 0 and row["physical_comparable"].lower() == "true"
            for row in coverage
        ),
        "Aggregate canonical validation rows failed",
    )
    fixed = rows(output / "fixed-plane-coverage.csv")
    require(
        len(fixed) == 15
        and {(row["condition"], row["plane"]) for row in fixed}
        == {
            (condition, plane)
            for condition in ("B", "R", "I", "O", "RIO")
            for plane in ("random", "reference", "attack")
        }
        and all(
            int(row["measured_values"]) == int(row["grid_points"]) ** 2 for row in fixed
        ),
        "Aggregate fixed-plane grids are incomplete",
    )
    figures = rows(output / "figure-index.csv")
    expected = {
        "end-defense-fixed-random",
        "end-defense-fixed-reference",
        "end-defense-fixed-attack",
        "broad-random-r125",
        "local-reference-matched-attack",
        "defense-history-local-reference",
    }
    require(
        len(figures) == len({row["figure"] for row in figures})
        and expected <= {row["figure"] for row in figures},
        "Required aggregate figures missing",
    )
    required = set()
    for row in figures:
        require(
            row["status"] == "complete"
            and math.isfinite(float(row["color_min"]))
            and math.isfinite(float(row["color_max"]))
            and float(row["color_min"]) < float(row["color_max"]),
            "Invalid aggregate shared-scale figure metadata",
        )
        declared = {
            "broad-random-r125": {0, 128, 1000},
            "local-reference-matched-attack": {0, 1, 8, 32, 128, 512, 1000},
            "defense-history-local-reference": {0, 8, 32, 125},
        }.get(row["figure"])
        if declared is not None:
            require(
                {int(value) for value in row["frames"].split(";")} == declared
                and int(row["panels"]) == 5 * len(declared),
                f"Aggregate frame coverage is incomplete: {row['figure']}",
            )
        else:
            require(
                int(row["panels"]) >= 5, f"Aggregate omits a condition: {row['figure']}"
            )
        for extension in ("png", "pdf"):
            required.add(output / "figures" / f"{row['figure']}.{extension}")
    return proof, required


def artifact_hash_gate(folder, metadata):
    artifacts = metadata.get("artifacts", {})
    require(artifacts, f"{folder}: artifact hashes missing")
    required = set()
    for name, expected in artifacts.items():
        path = folder / name
        require(
            path.is_file() and digest(path) == expected,
            f"Report artifact changed/missing: {path}",
        )
        required.add(path)
    return required


def static_gates(hr, fr, config):
    main_output = hr / "static/comparison"
    main = state(main_output / "validation.json")
    require(
        main.get("all_inputs_complete")
        and set(main.get("cases", {})) == {"B", "R", "I", "O", "RIO", "HF context"},
        "Six-condition static report is incomplete",
    )
    required = artifact_hash_gate(main_output, main)
    for cell, item in main["cases"].items():
        require(
            item.get("points") == 648
            and item.get("coordinate_validation", {}).get(
                "fp32_execution_coefficients_bitwise_match_reference"
            )
            and digest(Path(item["folder"]) / "grid.csv") == item["grid_sha256"],
            f"{cell}: static comparison grid identity/coordinate evidence failed",
        )
    state(hr / "status/static-comparison.json")
    output = fr / config["factorial_output"]
    final = state(output / "validation.json")
    state(fr / "status/static-factorial-report.json")
    require(
        final.get("cells") == 8
        and final.get("points_per_cell") == 648
        and set(final.get("proofs", {}))
        == {"B", "R", "I", "O", "RI", "RO", "IO", "RIO"},
        "Full-factorial endpoint report is incomplete",
    )
    helper = load_module(fr / config["factorial_source"], "seal_factorial_static")
    spec_path = fr / "source/comparison-v1/inputs.json"
    records, fresh = helper.load_records(read(spec_path))
    require(
        len(records) == 8
        and final.get("source_sha256") == digest(fr / config["factorial_source"])
        and final.get("input_spec_sha256") == digest(spec_path)
        and final.get("definitions") == helper.DEFINITIONS,
        "Factorial report source/design identity changed",
    )
    for field in ("common_protocol", "evaluated_batch_sha256", "defense_order_sha256"):
        require(
            final.get(field) == fresh.get(field),
            f"Factorial static proof changed: {field}",
        )
    required |= artifact_hash_gate(output, final)
    return {
        "main_six_static": "complete",
        "full_eight_static": "complete",
        "factorial_current_identities": fresh,
    }, required


def gate_all(hr, fr, repository, config):
    required = (
        {hr / relative for relative in config["required_main"]}
        | {fr / relative for relative in config["required_factorial"]}
        | {repository / relative for relative in config.get("required_repository", [])}
    )
    for path in required:
        require(
            path.is_file() and path.stat().st_size > 0,
            f"Required final artifact missing/empty: {path}",
        )
    programs = [read(root / "study.json") for root in (hr, fr)]
    require(
        all(
            program.get("kind") == "fixed-sampler-hparam-study" for program in programs
        ),
        "Unexpected study kind",
    )
    sources = source_gate((hr, fr), config["maximum_file_bytes"])
    cache, training, loss_matrices = {}, {}, {}
    for root, program, conditions, roles in (
        (hr, programs[0], config["main_conditions"], ("defense", "attack")),
        (fr, programs[1], config["factorial_conditions"], ("defense",)),
    ):
        for condition in conditions:
            for role in roles:
                name = f"hparam-{condition.lower()}-{'defense' if role == 'defense' else 'lr1e-6'}"
                case = program["cases"][name]
                require(case["role"] == role, f"Wrong declared role: {name}")
                reference = (
                    Path(program["baseline_dynamics"])
                    / "models/defense/observed-batches.jsonl"
                    if role == "defense"
                    else Path(program["compute_study"])
                    / "cases/baseline-lr1e-6/models/attack/observed-batches.jsonl"
                )
                training[str(root / "cases" / name)] = training_gate(
                    root / "cases" / name, case, reference, cache
                )
                if root == hr and role == "attack":
                    matrix = root / "cases" / name / "surfaces/minibatch-losses.npz"
                    loss_matrices[name] = validate_loss_matrix(matrix, hr, config)
                    for stage in config["main_attack_statuses"]:
                        state(root / "cases" / name / "status" / f"{stage}.json")
                    state(root / "cases" / name / "surfaces/validation.json", "passed")
                    state(
                        root
                        / "cases"
                        / name
                        / "surfaces/random-evolution-validation.json",
                        "passed",
                    )
                    required.update(
                        root / "cases" / name / relative
                        for relative in config["required_case_artifacts"]
                    )
                    required.update(
                        root / "cases" / name / f"{stem}.{extension}"
                        for stem in config["required_case_figures"]
                        for extension in ("png", "pdf")
                    )
    compare = load_module(hr / config["compare_source"], "seal_compare")
    evaluator = load_module(hr / config["evaluation_source"], "seal_evaluation")
    evaluation, raw_results = evaluation_gate(hr, programs[0], evaluator)
    numerical, checkpoint_figures = checkpoint_gate(hr, programs[0], compare)
    aggregate, aggregate_figures = aggregate_gate(hr / config["aggregate_output"])
    static, static_files = static_gates(hr, fr, config)
    required |= raw_results | checkpoint_figures | aggregate_figures | static_files
    for path in required:
        require(
            path.is_file() and path.stat().st_size > 0,
            f"Required final output missing/empty: {path}",
        )
    return {
        "training": training,
        "retained_minibatch_loss_matrices": loss_matrices,
        "evaluation": evaluation,
        "checkpoint_validation": numerical,
        "aggregate_five": aggregate,
        "static": static,
        "source_manifests": sources,
        "scope": {
            "main": "four D125 and four A1000; full native/evaluation/profile analysis",
            "factorial": "RI/RO/IO D125 and endpoint static only; no attack/benchmark requirements",
        },
    }, required


def markdown_targets(path):
    for target in re.findall(
        r"!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)", Path(path).read_text()
    ):
        target = target.strip("<>")
        parsed = urlparse(target)
        if parsed.scheme or target.startswith("#"):
            continue
        local = unquote(target.split("#", 1)[0])
        if local:
            yield (Path(path).parent / local).resolve()


def exclusion(path, repository, config, loss_matrix_root=None):
    path = Path(path)
    if not path.is_relative_to(repository) or not path.resolve().is_relative_to(
        repository
    ):
        return "outside repository"
    relative = path.relative_to(repository)
    parts = relative.parts
    if path.is_symlink():
        return "symlink payload"
    if any(
        part in config["excluded_directories"]
        or part.startswith(".")
        or part.startswith("renderings-")
        or part == "partial"
        for part in parts
    ):
        return "excluded directory/cache/intermediate"
    if path.suffix == ".npz" and allowed_loss_matrix(path, loss_matrix_root, config):
        try:
            validate_loss_matrix(path, loss_matrix_root, config)
        except GateError as error:
            return f"invalid allowlisted CE matrix: {error}"
        return None
    if path.suffix in config["excluded_suffixes"] or ".partial." in path.name:
        return "excluded arrays/weights/intermediate format"
    if not path.is_file() or (
        path.stat().st_size == 0
        and not ("source" in parts and path.name == "__init__.py")
    ):
        return "missing/empty/lock"
    if path.stat().st_size > config["maximum_file_bytes"]:
        return "large file"
    if path.suffix in {".log", ".out", ".err"}:
        if path.stat().st_size > config["maximum_log_bytes"]:
            return "large worker stdout/log"
        if ("seal" in path.name.lower() and "source" not in parts) or (
            "launches" in parts
            and any(
                "seal" in part.lower() for part in parts[parts.index("launches") + 1 :]
            )
        ):
            return "sealing stdout is live and not self-curated"
        return None
    if path.suffix == ".lock" and path.name != "uv.lock":
        return "lock file"
    if "source" in parts or path.name in {"manifest.sha256", "SHA256SUMS"}:
        return None
    if path.suffix not in {
        ".py",
        ".sh",
        ".json",
        ".csv",
        ".png",
        ".pdf",
        ".md",
        ".yml",
        ".yaml",
        ".toml",
        ".lock",
        ".txt",
        ".sha256",
    }:
        return "outside curated formats"
    return None


def select_payload(hr, fr, repository, config, required):
    selected, excluded = set(), []
    outputs = {hr / name for name in config["main_final_outputs"]}
    for root in (hr, fr):
        for folder, directories, files in os.walk(root, followlinks=False):
            directories[:] = [
                name
                for name in directories
                if name not in config["excluded_directories"]
                and not name.startswith(".")
                and not name.startswith("renderings-")
                and name != "partial"
                and not (Path(folder) / name).is_symlink()
            ]
            for name in files:
                path = Path(folder) / name
                if path in outputs:
                    continue
                reason = exclusion(path, repository, config, hr)
                if reason:
                    if (
                        path.suffix in {".log", ".out", ".err"}
                        and path.exists()
                        and path.stat().st_size
                    ):
                        excluded.append(
                            {
                                "path": path.relative_to(repository).as_posix(),
                                "reason": reason,
                                "bytes": path.stat().st_size,
                            }
                        )
                else:
                    selected.add(path)
    for path in required:
        reason = exclusion(path, repository, config, hr)
        require(
            reason is None, f"Required artifact cannot be curated ({reason}): {path}"
        )
        selected.add(path)
    # Check new presentation reports recursively; historical sealed reports are linked but not rewritten or expanded.
    pending = [
        hr / "report.md",
        hr / "figure-index.md",
        hr / "checkpoint-report.md",
        hr / "static/comparison/report.md",
        hr / "static/comparison/findings.md",
        fr / config["factorial_output"] / "report.md",
    ]
    visited = set()
    while pending:
        report = pending.pop()
        if report in visited:
            continue
        visited.add(report)
        require(report.is_file(), f"Required report missing: {report}")
        for target in markdown_targets(report):
            if target in outputs:
                continue
            reason = exclusion(target, repository, config, hr)
            require(
                reason is None,
                f"Broken/excluded local report link ({reason}): {report} -> {target}",
            )
            selected.add(target)
            if target.suffix == ".md" and any(
                target.is_relative_to(root) for root in (hr, fr)
            ):
                pending.append(target)
    for root in (hr, fr):
        require(
            any(
                path.is_relative_to(root / "launches")
                and path.suffix in {".out", ".err", ".log"}
                for path in selected
            ),
            f"No small actual launch log curated for {root}; preserve a bounded completed launch log",
        )
    return selected, excluded


def identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def hashable_payload(path, maximum, loss_matrix_root=None, config=None):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= maximum,
        f"Refuse missing/symlink/large payload: {path}",
    )
    if path.suffix == ".npz":
        require(
            allowed_loss_matrix(path, loss_matrix_root, config),
            f"Refuse non-allowlisted NumPy payload: {path}",
        )
        validate_loss_matrix(path, loss_matrix_root, config)
    else:
        require(
            path.suffix
            not in {
                ".npy",
                ".safetensors",
                ".bin",
                ".pt",
                ".pth",
                ".arrow",
                ".parquet",
            },
            f"Refuse tensor payload: {path}",
        )


def stable_hashes(selected, maximum=50_000_000, loss_matrix_root=None, config=None):
    hashes, identities = {}, {}
    for path in sorted(selected):
        hashable_payload(path, maximum, loss_matrix_root, config)
        before = identity(path)
        hashes[path] = digest(path)
        require(identity(path) == before, f"Artifact changed while hashing: {path}")
        identities[path] = before
    return hashes, identities


def verify_hashes(
    hashes, identities=None, maximum=50_000_000, loss_matrix_root=None, config=None
):
    for path, expected in hashes.items():
        hashable_payload(path, maximum, loss_matrix_root, config)
        require(digest(path) == expected, f"Curated artifact hash mismatch: {path}")
        if identities is not None:
            require(
                identity(path) == identities[path],
                f"Artifact replaced or changed during sealing: {path}",
            )


def manifest_entries(path, repository):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, relative = line.split(maxsplit=1)
        relative = Path(relative.strip())
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and re.fullmatch("[0-9a-f]{64}", expected),
            "Manifest must use safe repository-relative paths and SHA256 hashes",
        )
        absolute = repository / relative
        require(absolute not in result, f"Duplicate manifest path: {relative}")
        result[absolute] = expected
    require(result, "Empty curated artifact manifest")
    return result


def verify_bundle(hr, repository):
    manifest = hr / "artifact-manifest.sha256"
    tracked = hr / "tracked-artifacts.txt"
    validation = state(hr / "seal-validation.json")
    require(
        validation.get("payload_verified_before_publication") is True,
        "Payload was not verified before publication",
    )
    entries = manifest_entries(manifest, repository)
    paths = {
        repository / line
        for line in tracked.read_text().splitlines()
        if line and not line.startswith("#")
    }
    require(paths == set(entries) | {manifest}, "Tracked list and manifest disagree")
    verify_hashes(
        entries,
        maximum=validation["exclusion_policy"]["maximum_file_bytes"],
        loss_matrix_root=hr,
        config=validation["exclusion_policy"],
    )
    return {
        "state": "verified",
        "manifest": str(manifest),
        "manifest_sha256": digest(manifest),
        "hashed_files": len(entries),
        "tracked_entries": len(paths),
    }


def publish_bundle(hr, fr, repository, config, gates, selected, excluded):
    outputs = {name: hr / name for name in config["main_final_outputs"]}
    require(
        not any(path.exists() for path in outputs.values()),
        "Final seal files already exist; use verify rather than overwriting a seal",
    )
    hashes, identities = stable_hashes(
        selected, config["maximum_file_bytes"], hr, config
    )
    verify_hashes(hashes, identities, config["maximum_file_bytes"], hr, config)
    manifest = outputs["artifact-manifest.sha256"]
    tracked = outputs["tracked-artifacts.txt"]
    validation = outputs["seal-validation.json"]
    paths = sorted(
        path.relative_to(repository).as_posix()
        for path in set(selected) | set(outputs.values())
    )
    proof = {
        "state": "complete",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "main_study": str(hr),
        "factorial_study": str(fr),
        "flux_job_id": os.environ.get("FLUX_JOB_ID"),
        "gates": gates,
        "payload_verified_before_publication": True,
        "curated_payload_files": len(selected),
        "curated_payload_bytes": sum(path.stat().st_size for path in selected),
        "excluded_logs": excluded,
        "exclusion_policy": {
            key: config[key]
            for key in (
                "excluded_directories",
                "excluded_suffixes",
                "maximum_file_bytes",
                "maximum_log_bytes",
                "curated_minibatch_loss_matrices",
            )
        },
        "selected_launch_logs": [
            path.relative_to(repository).as_posix()
            for path in sorted(selected)
            if "launches" in path.parts and path.suffix in {".log", ".out", ".err"}
        ],
        "manifest_self_hash": "Manifest is tracked but cannot hash itself. All other tracked files are hashed. Final verification runs before the wrapper returns; its allocation stdout is not self-curated.",
        "source_and_reports_modified": False,
        "study_completion_states_modified": False,
    }
    with tempfile.TemporaryDirectory(prefix=".seal-stage-", dir=hr) as directory:
        staging = Path(directory)
        temporary_validation = staging / validation.name
        temporary_validation.write_text(
            json.dumps(proof, indent=2, allow_nan=False) + "\n"
        )
        temporary_tracked = staging / tracked.name
        temporary_tracked.write_text("\n".join(paths) + "\n")
        hashes[validation], hashes[tracked] = digest(temporary_validation), digest(
            temporary_tracked
        )
        temporary_manifest = staging / manifest.name
        temporary_manifest.write_text(
            "# SHA256 of curated files; paths relative to the lilpipe repository root.\n# The manifest itself is tracked but not self-hashed.\n"
            + "".join(
                f"{expected}  {path.relative_to(repository).as_posix()}\n"
                for path, expected in sorted(hashes.items())
            )
        )
        # Publish only after every required artifact and existing payload hash passed.
        verify_hashes(
            {
                path: expected
                for path, expected in hashes.items()
                if path not in {validation, tracked}
            },
            identities,
            config["maximum_file_bytes"],
            hr,
            config,
        )
        for source, target in (
            (temporary_validation, validation),
            (temporary_tracked, tracked),
            (temporary_manifest, manifest),
        ):
            os.replace(source, target)
    return verify_bundle(hr, repository)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "seal", "verify"))
    parser.add_argument("main_study", type=Path)
    parser.add_argument("factorial_study", type=Path)
    args = parser.parse_args()
    require(
        bool(os.environ.get("FLUX_JOB_ID")),
        "Run this helper only in an allocated launcher job",
    )
    sys.dont_write_bytecode = True
    example = Path.cwd().resolve()
    repository = example.parents[1]
    require(
        example.name == "tamper-resistance" and repository.name == "lilpipe",
        "Run from examples/tamper-resistance",
    )
    hr, fr = args.main_study.resolve(), args.factorial_study.resolve()
    require(
        hr != fr
        and all(root.is_relative_to(example / "artifacts") for root in (hr, fr)),
        "Unexpected study roots",
    )
    config = read(Path(__file__).with_name("requirements.json"))
    if args.mode == "verify":
        result = verify_bundle(hr, repository)
    else:
        gates, required = gate_all(hr, fr, repository, config)
        selected, excluded = select_payload(hr, fr, repository, config, required)
        if args.mode == "check":
            result = {
                "state": "ready",
                "payload_files": len(selected),
                "payload_bytes": sum(path.stat().st_size for path in selected),
                "excluded_logs": excluded,
                "gates": gates,
            }
        else:
            result = publish_bundle(
                hr, fr, repository, config, gates, selected, excluded
            )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
