"""Validate and hash the curated TAR basis-structure evidence bundle."""

import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def markdown_targets(path):
    text = Path(path).read_text()
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split(" ", 1)[0].strip("<>")
        parsed = urlparse(target)
        if parsed.scheme or target.startswith("#"):
            continue
        local = unquote(target.split("#", 1)[0])
        if local:
            yield (Path(path).parent / local).resolve()


def verify_frozen_manifests(source_dir):
    manifests = {}
    for manifest in sorted(Path(source_dir).rglob("manifest.sha256")):
        members = []
        for line in manifest.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            expected, filename = line.split(maxsplit=1)
            filename = Path(filename.strip())
            if not filename.is_absolute():
                filename = manifest.parent / filename
            actual = digest(filename)
            assert actual == expected, f"Frozen source hash changed: {filename}"
            members.append({"path": str(filename), "sha256": actual})
        manifests[str(manifest)] = {"sha256": digest(manifest), "members": members}
    assert manifests, "No source manifests found"
    return manifests


def main():
    run = Path(sys.argv[1]).resolve()
    assert os.environ.get("FLUX_JOB_ID"), "Seal must run through launch_tuo.py"
    example = Path.cwd().resolve()
    repository = example.parents[1]
    assert example.name == "tamper-resistance" and repository.name == "lilpipe"
    assert run.is_relative_to(example / "artifacts"), run

    required = [
        "report.md",
        "figure-index.md",
        "study.json",
        "jobs.json",
        "auxiliary-jobs.json",
        "token-audit/report.md",
        "token-audit/results.json",
        "token-audit/input-identity.json",
        "token-audit/basis-rows.csv",
        "token-audit/basis-columns.csv",
        "token-audit/loading-offset-rows.csv",
        "token-audit/top-tokens.csv",
        "token-audit/column-concentration.png",
        "token-audit/column-concentration.pdf",
        "module-replay/losses.csv",
        "module-replay/validation.json",
        "module-replay/module-only-ce-replay.png",
        "module-replay/module-only-ce-replay.pdf",
        "anchor-sensitivity/anchor-analysis.json",
        "anchor-sensitivity/attack-overlaps.csv",
        "anchor-sensitivity/attack-capture.csv",
        "anchor-sensitivity/parameter-plane-energy.csv",
        "anchor-sensitivity/anchor-sensitivity.png",
        "TAR_REPRODUCTION_PLAN.md",
        "TAR_COLLABORATOR_CHECKPOINT.md",
        "tests/test_tar_basis_structure.py",
        "examples/tamper-resistance/scripts/misc/tar_basis_structure.py",
        "concentration-jobs.json",
    ]
    outputs = run
    for relative in required:
        candidate = (
            repository / relative
            if relative.startswith(("TAR_", "tests/", "examples/"))
            else outputs / relative
        )
        assert candidate.is_file(), f"Required curated artifact missing: {candidate}"

    for phase in ("local-defense-attack", "collaborator-attack"):
        case_dir = run / "cases" / phase
        for phase_status in ("scan", "structure", "analysis", "report"):
            status = read(case_dir / "status" / f"{phase_status}.json")
            assert status.get("state") == "complete", (phase, phase_status, status)
        assert read(case_dir / "report-validation.json")["state"] == "passed"
        assert read(case_dir / "input-identity.json").get("signature")
        assert read(case_dir / "concentration.json")["state"] == "complete"
        assert read(case_dir / "subspaces.json")["state"] == "complete"
        assert (case_dir / "gram.npz").is_file()

    local_subspaces = read(run / "cases/local-defense-attack/subspaces.json")
    assert any(
        row.get("native_gram_matches") is True
        for row in local_subspaces.get("native_validations", [])
    ), "Local attack Gram must match the original native geometry"
    assert read(run / "status/token-audit.json")["state"] == "complete"
    assert read(run / "status/replay.json")["state"] == "complete"
    token_results = read(run / "token-audit/results.json")
    assert token_results["state"] == "complete"
    replay = read(run / "module-replay/validation.json")
    assert replay["state"] == "passed"
    required_replay_checks = {
        (case, step)
        for case in ("local-defense-attack", "collaborator-attack")
        for step in (0, 1, 1000)
    }
    found_replay_checks = set()
    for item in replay["native_endpoint_checks"]:
        assert max(item["errors"].values()) <= 1e-4, item
        found_replay_checks.add((item["case"], item["step"]))
    assert required_replay_checks <= found_replay_checks
    fixed_hashes = {
        item["probe_sha256"]["fixed"] for item in replay["native_endpoint_checks"]
    }
    heldout_hashes = {
        item["probe_sha256"]["heldout"] for item in replay["native_endpoint_checks"]
    }
    assert len(fixed_hashes) == len(heldout_hashes) == 1

    anchor = read(run / "anchor-sensitivity/anchor-analysis.json")
    assert anchor["state"] == "complete"
    assert all(
        item["verified"] for item in anchor["start_endpoint_full_span_invariance"]
    )

    for markdown in (
        run / "report.md",
        run / "figure-index.md",
        run / "token-audit/report.md",
    ):
        for target in markdown_targets(markdown):
            assert (
                target.is_file()
            ), f"Broken local Markdown link: {markdown} -> {target}"

    source_dir = run / "source"
    source_proofs = verify_frozen_manifests(source_dir)
    selected = set()

    def add(path):
        path = Path(path).resolve()
        assert path.is_file(), f"Missing selected artifact: {path}"
        assert path.is_relative_to(repository), f"Artifact outside repository: {path}"
        selected.add(path)

    for relative in required:
        add(
            repository / relative
            if relative.startswith(("TAR_", "tests/", "examples/"))
            else run / relative
        )
    for relative in (
        "figure-index.md",
        "study.json",
        "jobs.json",
        "auxiliary-jobs.json",
        "status/token-audit.json",
        "status/replay.json",
        "token-audit/report.md",
        "token-audit/results.json",
        "token-audit/input-identity.json",
        "token-audit/basis-rows.csv",
        "token-audit/basis-columns.csv",
        "token-audit/loading-offset-rows.csv",
        "token-audit/top-tokens.csv",
        "module-replay/losses.csv",
        "module-replay/validation.json",
        "anchor-sensitivity/anchor-analysis.json",
        "anchor-sensitivity/attack-overlaps.csv",
        "anchor-sensitivity/attack-capture.csv",
        "anchor-sensitivity/parameter-plane-energy.csv",
    ):
        add(run / relative)
    for name in ("local-defense-attack", "collaborator-attack"):
        case_dir = run / "cases" / name
        import numpy as np

        addition = case_dir / "displacement-concentration"
        assert read(addition / "validation.json")["state"] == "passed"
        assert read(addition / "status/analysis.json")["state"] == "complete"
        assert (
            read(addition / "status/concentration_comparison.json")["state"]
            == "complete"
        )
        results = read(addition / "results.json")
        assert results["state"] == "complete"
        prior_trajectory = case_dir / "trajectory-energy.csv"
        if prior_trajectory.exists():
            with prior_trajectory.open() as stream:
                prior_rows = list(csv.DictReader(stream))
            for record in results["records"]:
                if record["measure"] != "trajectory":
                    continue
                expected_modules = {}
                for row in prior_rows:
                    if (
                        row["trajectory"] == record["group"]
                        and row["precision"] == record["precision"]
                    ):
                        category = row["category"]
                        expected_modules[category] = expected_modules.get(
                            category, 0
                        ) + float(row["energy_fraction"])
                assert expected_modules
                for category, expected in expected_modules.items():
                    assert math.isclose(
                        record["module_shares"][category],
                        expected,
                        rel_tol=2e-6,
                        abs_tol=1e-12,
                    )
        with np.load(addition / "coordinate-histograms.npz") as histograms:
            assert len(histograms["counts"]) == len(results["records"])
            for record, counts, masses in zip(
                results["records"], histograms["counts"], histograms["masses"]
            ):
                assert 0 < counts.sum() <= record["coordinates"]
                assert int(counts.sum()) == record["nonzero_coordinates"]
                assert math.isclose(float(masses.sum()), record["energy"], rel_tol=2e-6)
                assert math.isclose(
                    sum(record["module_shares"].values()), 1, rel_tol=2e-6
                )
                for bound in record["energy_targets"].values():
                    assert (
                        0
                        <= bound["coordinate_fraction_lower"]
                        <= bound["coordinate_fraction_upper"]
                        <= 1
                    )
        for relative in (
            "job.json",
            "input-identity.json",
            "environment.json",
            "gpu-preflight.json",
            "results.json",
            "summary.csv",
            "validation.json",
            "coordinate-histograms.npz",
            "coordinate-comparison.png",
            "coordinate-comparison.pdf",
            "module-comparison.png",
            "module-comparison.pdf",
            "status/analysis.json",
            "status/concentration_comparison.json",
            "analysis.log",
        ):
            add(addition / relative)
        for relative in (
            "job.json",
            "environment.json",
            "gpu-preflight.json",
            "input-identity.json",
            "report-validation.json",
            "concentration.json",
            "gram.npz",
            "subspaces.json",
            "module-structure.csv",
            "parameter-structure.csv",
            "snapshot-capture.csv",
            "token-frequencies.csv",
            "token-summary.csv",
            "top-token-rows.csv",
            "token-row-energy.npz",
        ):
            add(case_dir / relative)
        for relative in (
            "scan.json",
            "structure.json",
            "analysis.json",
            "report.json",
        ):
            add(case_dir / "status" / relative)
        for relative in (
            "local-defense-attack/trajectory-energy.csv",
            "local-defense-attack/block-defense-attack-alignment.csv",
        ):
            candidate = run / "cases" / relative
            if candidate.is_file():
                add(candidate)
        for figure in case_dir.glob("*.png"):
            add(figure)
        for figure in case_dir.glob("*.pdf"):
            add(figure)
    for relative in (
        "module-replay/module-only-ce-replay.png",
        "module-replay/module-only-ce-replay.pdf",
        "anchor-sensitivity/anchor-sensitivity.png",
    ):
        add(run / relative)
    for source_file in source_dir.rglob("*"):
        if source_file.is_file() and source_file.suffix in {".py", ".sh"}:
            add(source_file)
    for manifest in source_dir.rglob("manifest.sha256"):
        add(manifest)
    for launch_script in (run / "launches").glob("*/launch.sh"):
        add(launch_script)
    for worker in (run / "launches").glob("*/subrunner.sh"):
        add(worker)
    for relative in (
        "logs/unit-tests.log",
        "logs/anchor-tests.log",
        "logs/token-audit-self-test.log",
        "logs/token-audit-report.log",
        "logs/hparam-prepare-tests.log",
        "logs/hparam-prepare-tests-v2.log",
        "logs/hparam-report-tests.log",
        "logs/format.log",
        "logs/replay-format.log",
        "logs/hparam-report-format.log",
        "logs/concentration-tests.log",
    ):
        add(run / relative)
    for log in (run / "logs").glob("*-dryrun.log"):
        add(log)
    for log in (run / "logs").glob("*-submission.log"):
        if "seal" not in log.name:
            add(log)

    # Every report link target is curated so those links remain valid after checkout.
    for markdown in (
        run / "report.md",
        run / "figure-index.md",
        run / "token-audit/report.md",
    ):
        for target in markdown_targets(markdown):
            add(target)

    tracked_list = run / "tracked-artifacts.txt"
    manifest = run / "artifact-manifest.sha256"
    validation_path = run / "seal-validation.json"
    tracked_paths = set(selected) | {tracked_list, manifest, validation_path}
    payload_bytes = sum(path.stat().st_size for path in selected)
    for path in selected:
        assert path.stat().st_size < 50_000_000, path

    publish(
        validation_path,
        {
            "state": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "run": str(run),
            "flux_job_id": os.environ["FLUX_JOB_ID"],
            "case_states": {
                name: {
                    phase: read(run / "cases" / name / "status" / f"{phase}.json")[
                        "state"
                    ]
                    for phase in ("scan", "structure", "analysis", "report")
                }
                for name in ("local-defense-attack", "collaborator-attack")
            },
            "token_audit": "complete",
            "module_replay": "passed",
            "anchor_full_span_invariance": "passed",
            "displacement_concentration": "passed",
            "source_manifests": source_proofs,
            "curated_payload_files": len(selected),
            "curated_payload_bytes": payload_bytes,
            "excluded_classes": [
                "model checkpoints and optimizer state",
                "cache directories and per-parameter scan/structure caches",
                "native basis NPY files",
                "original coordinate-histograms NPZ files (new comparison histograms are curated)",
                "token-audit/arrays.npz",
                "launch job stdout and worker stdout logs",
                "token-audit-values.log and module-replay.log",
            ],
            "manifest_self_hash": "intentionally omitted; the manifest is listed but cannot hash itself",
        },
    )
    relative_paths = sorted(
        path.relative_to(repository).as_posix() for path in tracked_paths
    )
    tracked_list.write_text(
        "# Paths are relative to the repository root (lilpipe); use as git pathspec input.\n"
        "# artifact-manifest.sha256 is listed but intentionally does not self-hash.\n"
        + "\n".join(relative_paths)
        + "\n"
    )
    lines = [
        "# SHA256 of curated files; verify from the repository root with sha256sum --check.",
        "# artifact-manifest.sha256 itself is listed in tracked-artifacts.txt but is not self-hashed.",
    ]
    for path in sorted(tracked_paths - {manifest}):
        lines.append(f"{digest(path)}  {path.relative_to(repository).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "state": "complete",
                "tracked_entries": len(tracked_paths),
                "hashed_entries": len(tracked_paths) - 1,
                "payload_bytes": payload_bytes,
                "manifest": str(manifest),
                "tracked_artifacts": str(tracked_list),
                "validation": str(validation_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
