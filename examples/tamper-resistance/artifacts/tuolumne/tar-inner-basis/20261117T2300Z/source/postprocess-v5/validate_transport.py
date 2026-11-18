"""Check that every aggregate used the declared exact local copy of archived inputs."""

import hashlib
import json
import os
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    run = Path(sys.argv[1]).resolve()
    spec = read(run / "source/reference-local-v5/analysis-spec.json")
    records = [read(Path(path)) for path in spec["aggregates"]]
    used_paths = {Path(record["reference_spec_path"]).resolve() for record in records}
    assert len(used_paths) == 1
    used_path = used_paths.pop()
    assert used_path.is_relative_to(run / "cache-provenance")
    proof_path = used_path.parent / "provenance.json"
    proof = read(proof_path)
    assert proof["status"] == "cleaned"
    source_spec = run / "references.json"
    assert Path(proof["spec"]) == source_spec
    assert proof["spec_sha256"] == hashlib.sha256(source_spec.read_bytes()).hexdigest()
    original = read(source_spec)["references"]
    cached = read(used_path)["references"]
    assert len(original) == len(cached) == 26
    for old, new in zip(original, cached):
        assert old["name"] == new["name"]
        for field in ("anchor", "path"):
            assert (
                new[field] == proof["source_folders"][str(Path(old[field]).resolve())]
            )
    for record in records:
        assert record["status"] == "complete" and record["references"] == cached
        assert (
            record["references_sha256"]
            == hashlib.sha256(json.dumps(cached, sort_keys=True).encode()).hexdigest()
        )
    for entry in proof["files"]:
        current = Path(entry["source"]).stat()
        actual = {
            "size": current.st_size,
            "dev": current.st_dev,
            "ino": current.st_ino,
            "mtime_ns": current.st_mtime_ns,
            "ctime_ns": current.st_ctime_ns,
        }
        assert actual == entry["source_stat"], entry["source"]
        assert len(entry["sha256"]) == 64
    result = {
        "state": "passed",
        "reference_checkpoints": 26,
        "source_files": len(proof["files"]),
        "source_bytes": proof["total_bytes"],
        "cache_proof": str(proof_path),
        "cache_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "source_spec_sha256": proof["spec_sha256"],
        "local_copies_verified_during_staging": True,
        "source_identities_unchanged_after_training": True,
        "local_cache_cleaned": True,
    }
    output = run / "reference-transport-validation.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
