"""Release a pending profile only after its immutable analysis inputs exist."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readiness(folder):
    settings = json.loads((folder / "study.json").read_text())
    for phase in ("train", "evaluation", "geometry"):
        assert (
            json.loads((folder / "status" / (phase + ".json")).read_text())["state"]
            == "complete"
        ), phase
    registered = json.loads((folder / "status/analysis-input.json").read_text())
    assert registered["settings_sha256"] == digest(folder / "study.json")
    geometry_path = folder / "geometry/geometry.json"
    geometry = json.loads(geometry_path.read_text())
    bases = []
    for plane in geometry["planes"]:
        path = folder / "geometry" / (plane + "-basis.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (2, geometry["parameters"])
        stat = path.stat()
        bases.append(
            dict(
                path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
            )
        )
        del array
    actual_path = folder / "surfaces/actual-losses.csv"
    matrix_path = folder / "surfaces/minibatch-losses.npz"
    before = {str(p): digest(p) for p in (actual_path, matrix_path)}
    with actual_path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {("defense", step) for step in [0, *settings["defense_steps"]]}
    expected |= {("attack", step) for step in [0, *settings["actual_attack_steps"]]}
    assert len(rows) == len(expected) == 37
    assert {(row["phase"], int(row["step"])) for row in rows} == expected
    for row in rows:
        assert all(
            math.isfinite(float(row[key])) for key in ("probe_loss", "heldout_loss")
        )
        if row["phase"] == "attack":
            if int(row["step"]) > 0:
                assert math.isfinite(float(row["last_batch_loss"]))
            if int(row["step"]) < settings["attack_max_steps"]:
                assert math.isfinite(float(row["next_batch_loss"]))
    with np.load(matrix_path, allow_pickle=False) as matrix:
        assert np.array_equal(matrix["steps"], settings["matrix_steps"])
        assert np.array_equal(matrix["batch_steps"], settings["matrix_batch_steps"])
        assert matrix["losses"].shape == (7, 128)
        assert np.isfinite(matrix["losses"]).all()
    assert before == {str(p): digest(p) for p in (actual_path, matrix_path)}
    return dict(
        actual_rows=len(rows),
        matrix_shape=[7, 128],
        settings_sha256=registered["settings_sha256"],
        geometry_sha256=digest(geometry_path),
        bases=bases,
        immutable_loss_sha256=before,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="+")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--profile-attempt", default="retry-v2")
    parser.add_argument("--analysis-attempt", default="peer-v2")
    args = parser.parse_args()
    path = ROOT / "matrix-jobs.json"
    ledger = json.loads(path.read_text())
    reports = [
        row
        for row in ledger.values()
        if row["phase"] == "report" and not row.get("superseded")
    ]
    assert len(reports) == 1, "Expected one current final-report allocation"
    proofs = []
    for name in args.case:
        key = "profiles:" + name + ":" + args.profile_attempt
        row = ledger[key]
        assert not row.get("superseded")
        folder = Path(row["run"]) / "cases" / name
        proof = readiness(folder)
        details = subprocess.check_output(
            ["scontrol", "show", "job", row["job_id"], "-o"], text=True
        )
        assert "JobState=PENDING " in details
        analysis = ledger["analysis:" + name + ":" + args.analysis_attempt]["job_id"]
        final = subprocess.check_output(
            ["scontrol", "show", "job", reports[0]["job_id"], "-o"], text=True
        )
        assert (
            "afterok:" + analysis in final
        ), "Final report must retain the full-analysis gate"
        proof.update(
            case=name,
            profile_job=row["job_id"],
            analysis_job=analysis,
            verified_epoch=time.time(),
            before_dependencies=row["dependencies"],
            applied=args.apply,
        )
        if args.apply:
            subprocess.run(
                ["scontrol", "update", "JobId=" + row["job_id"], "Dependency=0"],
                check=True,
            )
            row["dependencies"] = []
            row.setdefault("dependency_amendments", []).append(
                dict(
                    reason="Immutable geometry and complete checkpoint/minibatch losses published; overlap disjoint profile work. Final report retains full analysis.",
                    readiness=proof,
                )
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(ledger, indent=2) + "\n")
            temporary.replace(path)
            (folder / "status/profile-overlap-readiness.json").write_text(
                json.dumps(proof, indent=2) + "\n"
            )
        proofs.append(proof)
    output = (
        ROOT
        / "source/transition"
        / (
            "profile-overlap-"
            + "-".join(args.case)
            + ("-applied" if args.apply else "-dryrun")
            + ".json"
        )
    )
    output.write_text(json.dumps(proofs, indent=2) + "\n")
    print(json.dumps(proofs, indent=2), flush=True)


if __name__ == "__main__":
    main()
