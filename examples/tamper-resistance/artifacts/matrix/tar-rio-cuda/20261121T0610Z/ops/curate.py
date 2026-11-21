"""Archive bounded completed outputs while excluding models and benchmark data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT.parents[3]
EXCLUDED_DIRS = {"models", "evals", "cache", "__pycache__", "wandb"}
SUFFIXES = {
    ".py",
    ".sh",
    ".json",
    ".md",
    ".csv",
    ".yml",
    ".yaml",
    ".png",
    ".pdf",
    ".gif",
    ".npz",
    ".log",
    ".out",
    ".txt",
    ".sha256",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    assert json.loads((ROOT / "status/report.json").read_text())["state"] == "complete"
    accounting = json.loads((ROOT / "status/poll.json").read_text())
    assert accounting["all_complete"] and not accounting["errors"]
    provenance = json.loads((ROOT / "report-provenance.json").read_text())
    preflight = json.loads((ROOT / "ops/report-preflight-validation.json").read_text())
    assert provenance["state"] == "complete" and provenance["reused_inputs_unchanged"]
    assert (
        provenance["report_worker_sha256"]
        == preflight["report_worker_sha256"]
        == digest(ROOT / "ops/finish_report.py")
    )
    overlaps = []
    for path in sorted(
        (ROOT / "cases").glob("*/status/profile-overlap-readiness.json")
    ):
        proof = json.loads(path.read_text())
        current = {name: digest(Path(name)) for name in proof["immutable_loss_sha256"]}
        assert current == proof["immutable_loss_sha256"], path
        for basis in proof["bases"]:
            stat = Path(basis["path"]).stat()
            assert (
                stat.st_size == basis["bytes"] and stat.st_mtime_ns == basis["mtime_ns"]
            ), path
        overlaps.append(
            {
                "case": proof["case"],
                "readiness_sha256": digest(path),
                "immutable_loss_sha256": current,
                "bases_unchanged": True,
            }
        )
    if overlaps:
        (ROOT / "status/profile-overlap-verification.json").write_text(
            json.dumps({"state": "complete", "cases": overlaps}, indent=2) + "\n"
        )
    files, links, skipped_logs, scalar_archives = [], {}, [], []
    manifest_path = ROOT / "curation.json"
    for directory, directories, names in os.walk(ROOT, followlinks=False):
        parent = Path(directory)
        keep = []
        for name in directories:
            path = parent / name
            if name in EXCLUDED_DIRS:
                continue
            if path.is_symlink():
                links[str(path.relative_to(EXAMPLE))] = os.readlink(path)
            else:
                keep.append(name)
        directories[:] = keep
        for name in names:
            path = parent / name
            if path == manifest_path or path.suffix not in SUFFIXES:
                continue
            relative = str(path.relative_to(EXAMPLE))
            if path.is_symlink():
                links[relative] = os.readlink(path)
                continue
            size = path.stat().st_size
            if size > 5 * 1024**2:
                if path.suffix in {".log", ".out"}:
                    skipped_logs.append(
                        {"path": relative, "bytes": size, "sha256": digest(path)}
                    )
                    continue
                raise ValueError(
                    f"Review oversized scientific output before archiving: {path}"
                )
            if path.suffix == ".npz":
                assert name == "minibatch-losses.npz" and size < 1024**2, path
                with np.load(path, allow_pickle=False) as data:
                    assert all(
                        data[key].dtype.kind in "iuf" and np.isfinite(data[key]).all()
                        for key in data.files
                    ), path
                    scalar_archives.append(
                        {
                            "path": relative,
                            "arrays": {
                                key: {
                                    "shape": list(data[key].shape),
                                    "dtype": str(data[key].dtype),
                                }
                                for key in data.files
                            },
                        }
                    )
            if path.suffix == ".md":
                for destination in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                    destination = unquote(destination.split("#", 1)[0]).strip("<>")
                    if (
                        not destination
                        or "://" in destination
                        or destination.startswith("mailto:")
                    ):
                        continue
                    target = (
                        Path(destination)
                        if destination.startswith("/")
                        else path.parent / destination
                    )
                    assert target.exists(), (path, destination)
            files.append({"path": relative, "bytes": size, "sha256": digest(path)})
    files.sort(key=lambda item: item["path"])
    manifest = {
        "state": "complete",
        "scope": "Completed CUDA RIO K comparison: reports, figures, scalar loss summaries, validation and execution provenance.",
        "files": files,
        "links": dict(sorted(links.items())),
        "total_file_bytes": sum(item["bytes"] for item in files),
        "scalar_loss_archives": scalar_archives,
        "oversized_logs_not_committed": skipped_logs,
        "excluded_directories": sorted(EXCLUDED_DIRS),
        "directory_symlinks_followed": False,
        "csv_bytes_preserved": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.stage:
        paths = (
            [item["path"] for item in files]
            + list(links)
            + [str(manifest_path.relative_to(EXAMPLE))]
        )
        for offset in range(0, len(paths), 100):
            subprocess.run(
                ["git", "add", "-f", "--", *paths[offset : offset + 100]],
                cwd=EXAMPLE,
                check=True,
            )
    print(
        json.dumps(
            {
                "files": len(files),
                "links": len(links),
                "bytes": manifest["total_file_bytes"],
                "scalar_loss_archives": len(scalar_archives),
                "oversized_logs": len(skipped_logs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
