"""Stage immutable reference checkpoint files on two local storage tiers."""

import argparse
import hashlib
import json
import os
import shutil
import socket
import stat
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUFFER = 8 * 1024**2
TIERS = ("ssd", "ram")
MARKER = ".tar-reference-owner.json"


def _json(path, value):
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    tmp.replace(path)


def _ident(info):
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Expected a regular checkpoint file")
    return dict(
        size=info.st_size,
        dev=info.st_dev,
        ino=info.st_ino,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _nearest(path):
    return next(parent for parent in (path, *path.parents) if parent.exists())


def _files(folder):
    files = sorted(folder.glob("*.safetensors"))
    index = folder / "model.safetensors.index.json"
    if index.exists():
        names = {p.name for p in files}
        mapping = json.loads(index.read_text())["weight_map"]
        if not mapping or not set(mapping.values()) <= names:
            raise ValueError(f"Invalid checkpoint index in {folder}")
        files.append(index)
    elif not (folder / "model.safetensors").is_file():
        raise ValueError(f"Missing model.safetensors or its index in {folder}")
    return files


def _copy(row):
    source, dest = Path(row["source"]), Path(row["destination"])
    with source.open("rb") as src, dest.open("xb") as dst:
        if _ident(os.fstat(src.fileno())) != row["source_stat"]:
            raise ValueError(f"Source changed before copy: {source}")
        digest = hashlib.sha256()
        while chunk := src.read(BUFFER):
            digest.update(chunk)
            dst.write(chunk)
    with dest.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if _ident(source.stat()) != row["source_stat"] or actual != digest.hexdigest():
        raise ValueError(f"Unstable or corrupt transfer: {source}")
    if dest.stat().st_size != row["source_stat"]["size"]:
        raise ValueError(f"Short transfer: {dest}")
    row.update(sha256=actual, destination_stat=_ident(dest.stat()))
    logical = Path(row["logical"])
    if logical != dest:
        logical.symlink_to(dest)


def stage(
    spec,
    ssd_root,
    ram_root,
    output,
    *,
    reserve_ssd_bytes=20 * 1024**3,
    max_ram_bytes=240_000_000_000,
    max_model_bytes=450_000_000_000,
):
    if min(reserve_ssd_bytes, max_ram_bytes, max_model_bytes) < 0:
        raise ValueError("Cache capacity limits must be nonnegative")
    output = Path(output).resolve()
    spec = Path(spec).resolve()
    if output.exists():
        raise FileExistsError(output)
    raw = spec.read_bytes()
    config = json.loads(raw)
    refs = config["references"]
    if not refs:
        raise ValueError("Empty reference spec")
    token = uuid.uuid4().hex
    parents = dict(ssd=Path(ssd_root).resolve(), ram=Path(ram_root).resolve())
    roots = {k: p / f"tar-reference-{token}-{k}" for k, p in parents.items()}
    mounts = {k: _nearest(p) for k, p in parents.items()}
    capacities = {
        "ssd": max(0, shutil.disk_usage(mounts["ssd"]).free - reserve_ssd_bytes),
        "ram": min(max_ram_bytes, shutil.disk_usage(mounts["ram"]).free),
    }
    folders = sorted(
        {Path(ref[k]).resolve(strict=True) for ref in refs for k in ("anchor", "path")}
    )
    names = {folder: f"checkpoint-{i:04d}" for i, folder in enumerate(folders)}
    rows = []
    for folder in folders:
        for p in _files(folder):
            rows.append(
                {
                    "source": str(p),
                    "source_stat": _ident(p.stat()),
                    "relative": str(Path(names[folder]) / p.name),
                }
            )
    total = sum(r["source_stat"]["size"] for r in rows)
    if total > max_model_bytes or total > sum(capacities.values()):
        raise ValueError(
            f"Insufficient capacity for {total} bytes with tier limits {capacities} and total cap {max_model_bytes}"
        )
    if (
        mounts["ssd"].stat().st_dev == mounts["ram"].stat().st_dev
        and total > capacities["ssd"]
    ):
        raise ValueError("Both tiers share a filesystem with insufficient space")
    remaining = capacities.copy()
    for r in sorted(rows, key=lambda r: -r["source_stat"]["size"]):
        size = r["source_stat"]["size"]
        tier = next((k for k in TIERS if remaining[k] >= size), None)
        if tier is None:
            raise ValueError(f"Insufficient space for intact shard {r['source']}")
        remaining[tier] -= size
        r.update(
            tier=tier,
            destination=str(roots[tier] / r["relative"]),
            logical=str(roots["ssd"] / r["relative"]),
        )
    provenance = output / "provenance.json"
    owner = dict(
        token=token,
        host=socket.gethostname(),
        uid=os.getuid(),
        provenance=str(provenance),
    )
    out = dict(
        schema_version=1,
        status="staging",
        owner=owner,
        spec=str(spec),
        spec_sha256=hashlib.sha256(raw).hexdigest(),
        roots={k: str(p) for k, p in roots.items()},
        source_folders={str(k): str(roots["ssd"] / v) for k, v in names.items()},
        capacities=capacities,
        planned_bytes={k: capacities[k] - remaining[k] for k in TIERS},
        total_bytes=total,
        files=rows,
    )
    print(
        f"Staging {total} bytes in {len(rows)} files: {out['planned_bytes']}",
        flush=True,
    )
    output.mkdir(parents=True, mode=0o700)
    _json(provenance, out)
    created = []
    try:
        for directory in roots.values():
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir(mode=0o700)
            created.append(directory)
            _json(directory / MARKER, owner)
        for r in rows:
            for key in ("logical", "destination"):
                Path(r[key]).parent.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, _ in enumerate(executor.map(_copy, rows), 1):
                if i % 8 == 0 or i == len(rows):
                    print(f"Verified {i}/{len(rows)} cache files", flush=True)
        for r in rows:
            if _ident(Path(r["source"]).stat()) != r["source_stat"]:
                raise ValueError(f"Source changed during staging: {r['source']}")
        for ref in refs:
            for k in ("anchor", "path"):
                ref[k] = str(roots["ssd"] / names[Path(ref[k]).resolve(strict=True)])
        _json(output / "references.json", config)
        out["status"] = "complete"
        _json(provenance, out)
    except BaseException as err:
        out.update(status="failed", error=str(err))
        _json(provenance, out)
        # All roots below were exclusively created by this call.
        for directory in created:
            shutil.rmtree(directory)
        raise
    return provenance


def cleanup(provenance):
    path = Path(provenance).resolve(strict=True)
    doc = json.loads(path.read_text())
    owner = doc["owner"]
    token = owner["token"]
    if (
        doc["status"] != "complete"
        or len(token) != 32
        or uuid.UUID(hex=token).hex != token
        or owner
        != dict(
            token=token,
            host=socket.gethostname(),
            uid=os.getuid(),
            provenance=str(path),
        )
    ):
        raise ValueError("Invalid cache owner, host, location or status")
    roots = {k: Path(doc["roots"][k]) for k in TIERS}
    expected = {k: {p / MARKER} for k, p in roots.items()}
    for tier, root in roots.items():
        info = root.lstat()
        if (
            root.resolve() != root
            or root.name != f"tar-reference-{token}-{tier}"
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or (root / MARKER).is_symlink()
            or json.loads((root / MARKER).read_text()) != owner
        ):
            raise ValueError(f"Refusing to remove unowned cache root {root}")
    for r in doc["files"]:
        rel = Path(r["relative"])
        tier = r["tier"]
        target = roots[tier] / rel
        logical = roots["ssd"] / rel
        if (
            len(rel.parts) != 2
            or rel.is_absolute()
            or not rel.parts[0].startswith("checkpoint-")
            or not (
                rel.name.endswith(".safetensors")
                or rel.name == "model.safetensors.index.json"
            )
            or str(target) != r["destination"]
            or str(logical) != r["logical"]
            or target.resolve() != target
            or logical.resolve() != target
        ):
            raise ValueError("Cache file outside its declared layout")
        if (
            _ident(Path(r["source"]).stat()) != r["source_stat"]
            or _ident(target.lstat()) != r["destination_stat"]
        ):
            raise ValueError(f"Changed source or cached file: {r['source']}")
        expected[tier].update((target, target.parent))
        expected["ssd"].update((logical, logical.parent))
    for k, root in roots.items():
        if set(root.rglob("*")) != expected[k]:
            raise ValueError(f"Unexpected files in {root}")
    for root in roots.values():
        shutil.rmtree(root)
    doc["status"] = "cleaned"
    _json(path, doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    stage_args = subs.add_parser("stage")
    for option in ("spec", "ssd-root", "ram-root", "output"):
        stage_args.add_argument("--" + option, required=True)
    stage_args.add_argument("--reserve-ssd-bytes", type=int, default=20 * 1024**3)
    stage_args.add_argument("--max-ram-bytes", type=int, default=240_000_000_000)
    stage_args.add_argument("--max-model-bytes", type=int, default=450_000_000_000)
    subs.add_parser("cleanup").add_argument("--provenance", required=True)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "stage":
        stage(**args)
    else:
        cleanup(**args)


if __name__ == "__main__":
    main()
