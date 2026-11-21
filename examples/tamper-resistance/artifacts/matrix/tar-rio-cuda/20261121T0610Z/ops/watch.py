"""Read scheduler and scientific completion markers for the CUDA follow-up."""

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main():
    ledger = json.loads((ROOT / "matrix-jobs.json").read_text())
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
            ",".join(row["job_id"] for row in ledger.values()),
            "--parsable2",
            "--format=JobID,State%30,Elapsed,Submit,Start,End,Reason",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    scheduler = {
        row["JobID"]: row
        for row in csv.DictReader(io.StringIO(response.stdout), delimiter="|")
    }
    jobs = []
    errors = []
    for key, declared in ledger.items():
        if declared.get("superseded"):
            continue
        job = scheduler.get(declared["job_id"], {})
        state = job.get("State", "UNKNOWN").split()[0]
        path = Path(declared["status_file"])
        try:
            scientific = json.loads(path.read_text()).get("state")
        except (FileNotFoundError, json.JSONDecodeError):
            scientific = None
        done = state == "COMPLETED" and scientific == "complete"
        jobs.append(
            {
                "key": key,
                "job_id": declared["job_id"],
                "allocation_state": state,
                "phase_state": scientific,
                "done": done,
                "scheduler": job,
            }
        )
        if (
            state
            in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
            }
            and not done
        ):
            errors.append(f"{key}: {state}, scientific state {scientific}")
    required = {
        f"{phase}:rio-k{k}-cuda-{suffix}"
        for k in (16, 64)
        for phase, suffix in [
            ("train", "defense"),
            ("train", "lr1e-6"),
            ("eval", "lr1e-6"),
            ("analysis", "lr1e-6"),
            ("profiles", "lr1e-6"),
        ]
    }
    required.add("report")
    active = {row["key"] for row in jobs}
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_complete": required <= active and all(row["done"] for row in jobs),
        "missing_declared_phases": sorted(required - active),
        "jobs": jobs,
        "errors": errors,
    }
    (ROOT / "status/poll.json").write_text(json.dumps(result, indent=2) + "\n")
    with (ROOT / "logs/poll-history.jsonl").open("a") as stream:
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "time": result["checked_at_utc"],
                "all_complete": result["all_complete"],
                "done": sum(row["done"] for row in jobs),
                "declared": len(jobs),
                "missing_phases": result["missing_declared_phases"],
                "errors": errors,
            }
        ),
        flush=True,
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
