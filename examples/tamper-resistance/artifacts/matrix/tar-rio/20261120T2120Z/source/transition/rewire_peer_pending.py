"""Rewire pending consumers before canceling superseded one-GPU analyses."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    path = ROOT / "matrix-jobs.json"
    ledger = json.loads(path.read_text())
    replacements = {}
    keys = []
    for key, row in ledger.items():
        if row.get("superseded") or row["phase"] != "analysis":
            continue
        peer = ledger.get(key + ":peer")
        if peer:
            replacements[row["job_id"]] = peer["job_id"]
            keys.append(key)
    assert replacements, "No pending replacements"
    by_id = {row["job_id"]: row for row in ledger.values()}
    plan = {"replacements": replacements, "updates": [], "canceled": []}
    plan_path = ROOT / "source/transition" / (
        "peer-rewire-" + "-".join(replacements.values()) + ".json"
    )

    def save():
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        temporary.replace(path)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    for key, row in ledger.items():
        if row.get("superseded") or not set(row.get("dependencies", [])).intersection(replacements):
            continue
        details = subprocess.check_output(
            ["scontrol", "show", "job", row["job_id"], "-o"], text=True
        )
        assert "JobState=PENDING " in details, key
        before = list(row["dependencies"])
        after, verified = [], []
        for job in before:
            if job in replacements:
                after.append(replacements[job])
                continue
            parent = by_id[job]
            status = Path(parent["status_file"])
            if status.exists() and json.loads(status.read_text()).get("state") == "complete":
                verified.append(job)
            else:
                after.append(job)
        subprocess.run(
            ["scontrol", "update", "JobId=" + row["job_id"],
             "Dependency=afterok:" + ":".join(after)], check=True
        )
        row["dependencies"] = after
        amendment = dict(before=before, after=after, completed_markers_verified=verified)
        row.setdefault("dependency_amendments", []).append(amendment)
        plan["updates"].append(dict(key=key, job_id=row["job_id"], **amendment))
        save()

    for key in keys:
        row = ledger[key]
        details = subprocess.check_output(
            ["scontrol", "show", "job", row["job_id"], "-o"], text=True
        )
        assert "JobState=PENDING " in details, key
        row.update(superseded=True, superseded_by=replacements[row["job_id"]],
                   supersession_reason="Exact peer pilot passed; switch pending analysis to two-GPU residency.")
    save()
    assert not any(
        not row.get("superseded") and set(row.get("dependencies", [])).intersection(replacements)
        for row in ledger.values()
    )
    for job in replacements:
        subprocess.run(["scancel", job], check=True)
        plan["canceled"].append(job)
        save()
    plan["state"] = "complete"
    save()
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == "__main__":
    main()
