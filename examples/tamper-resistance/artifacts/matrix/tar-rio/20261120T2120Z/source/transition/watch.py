"""Poll Slurm work and the remaining original Flux requests every 15 minutes."""

import argparse
import collections
import csv
import io
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path(
    "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z"
)


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def poll():
    jobs = read(ROOT / "matrix-jobs.json")
    active = {key: row for key, row in jobs.items() if not row.get("superseded")}
    ids = [row["job_id"] for row in active.values()]
    rows, errors = [], []
    if ids:
        response = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                "matrix",
                "sacct",
                "-X",
                "-j",
                ",".join(ids),
                "--parsable2",
                "--format=JobID,State%30,Elapsed,Submit,Start,End,Reason",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        scheduler = {
            row["JobID"]: row
            for row in csv.DictReader(io.StringIO(response.stdout), delimiter="|")
        }
        for key, declared in active.items():
            job = scheduler.get(declared["job_id"], {})
            state = job.get("State", "UNKNOWN").split()[0]
            scientific = read(Path(declared["status_file"])).get("state")
            terminal = state in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
            }
            done = state == "COMPLETED" and scientific == "complete"
            rows.append(
                {
                    "key": key,
                    "backend": "slurm",
                    "job_id": declared["job_id"],
                    "allocation_state": state,
                    "phase_state": scientific,
                    "done": done,
                    "scheduler": job,
                    "status_file": declared["status_file"],
                }
            )
            if terminal and not done:
                errors.append(
                    key + ": " + state + ", scientific state " + str(scientific)
                )
    replaced = read(ROOT / "cancellations.json")
    excluded = {row["job_id"] for row in replaced.get("jobs", [])}
    original = read(ORIGIN / "batch-operation-jobs.json")
    original.update(
        {
            "aux-new-anchor-parity": {
                "job_id": "f3m5LNyUCuVq",
                "status_file": str(
                    ORIGIN / "pilots/rio-prefix-new-anchor/status/train.json"
                ),
            },
            "aux-precision-audit": {
                "job_id": "f3m6GBJfJki3",
                "status_file": str(
                    ORIGIN / "source/precision-audit/results/precision-audit.json"
                ),
            },
        }
    )
    original = {
        key: row for key, row in original.items() if row["job_id"] not in excluded
    }
    if original:
        response = subprocess.run(
            [
                "flux",
                "--root",
                "jobs",
                "--json",
                *[row["job_id"] for row in original.values()],
            ],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        )
        payload = json.loads(response.stdout)
        scheduler = {row["jobid"]: row for row in payload.get("jobs", [payload])}
        for key, declared in original.items():
            job = scheduler[declared["job_id"]]
            if "status_file" in declared:
                scientific = read(Path(declared["status_file"])).get("state")
            else:
                phase = {"eval": "evaluation", "profiles": "step-profiles"}.get(
                    declared.get("phase"), declared.get("phase", "train")
                )
                case = declared.get("case", declared["target"][-1])
                scientific = read(
                    ORIGIN / "cases" / case / "status" / (phase + ".json")
                ).get("state")
            done = job["state"] == "INACTIVE" and scientific == "complete"
            rows.append(
                {
                    "key": key,
                    "backend": "flux",
                    "job_id": declared["job_id"],
                    "allocation_state": job["state"],
                    "phase_state": scientific,
                    "done": done,
                }
            )
            if (
                job["state"] == "INACTIVE"
                and not done
                and time.time() - job.get("t_inactive", 0) > 60
            ):
                errors.append(key + ": inactive, scientific state " + str(scientific))
    counts = {
        backend: dict(
            collections.Counter(
                row["allocation_state"] for row in rows if row["backend"] == backend
            )
        )
        for backend in ("slurm", "flux")
    }
    result = {
        "observed_epoch": time.time(),
        "counts": counts,
        "errors": errors,
        "jobs": rows,
        "superseded_original_requests": len(excluded),
    }
    target = ROOT / "status/poll.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    print(
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        counts,
        "completed",
        sum(row["done"] for row in rows),
        "/",
        len(rows),
        flush=True,
    )
    if errors:
        raise RuntimeError("ROOTERROR: " + "; ".join(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        poll()
        if not args.watch:
            return
        subprocess.run(["sleep", "900"], check=True)


if __name__ == "__main__":
    main()
