import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_FILE = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_reference_cache.py"
)
module_spec = importlib.util.spec_from_file_location("tar_reference_cache", MODULE_FILE)
cache = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cache)


@pytest.fixture
def example(tmp_path, monkeypatch):
    ssd, ram, archive = (tmp_path / k for k in ("ssd", "ram", "source"))
    for p in (ssd, ram, archive):
        p.mkdir()
    actual_stat = Path.stat

    def stat_dev(path, *a, **kw):
        if path == ssd:
            return SimpleNamespace(st_dev=90, st_mode=0o40755)
        return actual_stat(path, *a, **kw)

    monkeypatch.setattr(Path, "stat", stat_dev)
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=140 if path == ssd else 1000),
    )
    folders = []
    for i in range(2):
        folder = archive / str(i)
        folder.mkdir()
        if i:
            shard = "model-00001.safetensors"
            (folder / shard).write_bytes(b"b" * 100)
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": shard}})
            )
        else:
            (folder / "model.safetensors").write_bytes(b"a" * 70)
        (folder / "ignored.dat").write_bytes(b"ignore")
        folders.append(folder)
    alias = tmp_path / "alias"
    alias.symlink_to(folders[0], target_is_directory=True)
    ref = {"name": "first", "anchor": str(folders[0]), "path": str(folders[1])}
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "references": [ref, dict(ref, name="alias", anchor=str(alias))],
                "version": 1,
            }
        )
    )
    return dict(
        spec=spec,
        ssd_root=ssd,
        ram_root=ram,
        output=tmp_path / "proof",
        reserve_ssd_bytes=20,
        max_ram_bytes=300,
        max_model_bytes=400,
    )


def test_roundtrip_and_private_cleanup(example):
    provenance = cache.stage(**example)
    doc = json.loads(provenance.read_text())
    assert doc["status"] == "complete"
    assert len(doc["files"]) == 3 and doc["planned_bytes"]["ram"] > 0
    assert all(Path(p).stat().st_mode & 0o777 == 0o700 for p in doc["roots"].values())
    config = json.loads((example["output"] / "references.json").read_text())
    assert config["references"][0]["anchor"] == config["references"][1]["anchor"]
    assert config["version"] == 1
    for row in doc["files"]:
        contents = Path(row["source"]).read_bytes()
        assert (
            Path(row["logical"]).read_bytes()
            == Path(row["destination"]).read_bytes()
            == contents
        )
        assert hashlib.sha256(contents).hexdigest() == row["sha256"]
        assert not (Path(row["logical"]).parent / "ignored.dat").exists()
    with pytest.raises(FileExistsError):
        cache.stage(**example)
    cache.cleanup(provenance)
    assert json.loads(provenance.read_text())["status"] == "cleaned"
    assert all(not Path(p).exists() for p in doc["roots"].values())
    assert all(Path(row["source"]).exists() for row in doc["files"])


@pytest.mark.parametrize("case", ["ram_limit", "total_limit"])
def test_capacity_precedes_writes(example, case):
    example["max_ram_bytes" if case == "ram_limit" else "max_model_bytes"] = 20
    with pytest.raises(ValueError, match="capacity"):
        cache.stage(**example)
    assert not example["output"].exists()
    assert not list(example["ssd_root"].iterdir())
    assert not list(example["ram_root"].iterdir())


@pytest.mark.parametrize(
    "case", ["source", "destination", "arbitrary_root", "extra_file"]
)
def test_cleanup_refuses_changes_before_deletion(example, case):
    provenance = cache.stage(**example)
    doc = json.loads(provenance.read_text())
    if case in ("source", "destination"):
        Path(doc["files"][0][case]).write_bytes(b"changed")
    elif case == "arbitrary_root":
        doc["roots"]["ssd"] = str(example["ssd_root"])
        provenance.write_text(json.dumps(doc))
    else:
        (Path(doc["roots"]["ssd"]) / "unexpected").touch()
    with pytest.raises(ValueError):
        cache.cleanup(provenance)
    assert Path(doc["roots"]["ram"]).is_dir()
    assert Path(doc["files"][0]["logical"]).exists()
