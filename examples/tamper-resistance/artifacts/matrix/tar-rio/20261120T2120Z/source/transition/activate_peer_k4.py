"""Replace the K4 analysis gate after the exact two-GPU pilot passes."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    ledger_path = ROOT / "matrix-jobs.json"
    ledger = json.loads(ledger_path.read_text())
    peer = ledger["analysis:rio-k4-lr1e-6:peer"]
    new = peer["job_id"]
    closure_key = "analysis:rio-k4-lr1e-6:closure"
    old = ledger[closure_key]["job_id"]
    original = ledger["analysis:rio-k4-lr1e-6"]["job_id"]
    pilot = json.loads(Path(peer["pilot_result"]).read_text())
    assert pilot["state"] == "complete" and all(pilot["gates"].values())
    assert new == "516314" and old == "516227" and original == "516068"

    plan = {"new_gate": new, "old_gate": old, "old_worker": original, "updates": []}
    plan_path = ROOT / "source/transition/peer-k4-activation.json"

    def save():
        tmp = ledger_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger, indent=2) + "\n")
        tmp.replace(ledger_path)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    by_id = {row["job_id"]: row for row in ledger.values()}
    for key, row in ledger.items():
        if row.get("superseded") or old not in row.get("dependencies", []):
            continue
        details = subprocess.check_output(
            ["scontrol", "show", "job", row["job_id"], "-o"], text=True
        )
        assert "JobState=PENDING " in details, key
        before = list(row["dependencies"])
        after, verified = [], []
        for job in before:
            if job == old:
                after.append(new)
                continue
            parent = by_id[job]
            status = Path(parent["status_file"])
            # Completed producer markers are independently checked again by report.
            if status.exists() and json.loads(status.read_text()).get("state") == "complete":
                verified.append(job)
            else:
                after.append(job)
        dependency = "afterok:" + ":".join(after)
        subprocess.run(
            ["scontrol", "update", "JobId=" + row["job_id"], "Dependency=" + dependency],
            check=True,
        )
        row["dependencies"] = after
        amendment = dict(before=before, after=after, completed_markers_verified=verified)
        row.setdefault("dependency_amendments", []).append(amendment)
        plan["updates"].append(dict(key=key, job_id=row["job_id"], **amendment))
        save()

    assert not any(
        not row.get("superseded") and old in row.get("dependencies", [])
        for row in ledger.values()
    )
    for key in (closure_key, "analysis:rio-k4-lr1e-6"):
        ledger[key].update(
            superseded=True,
            superseded_by=new,
            supersession_reason="Exact peer pilot passed; reuse completed rows with GPU-resident centers and directions.",
        )
    plan["state"] = "rewired"
    save()
    # Cancel the pending closure first, then release the afterany peer continuation.
    for job in (old, original):
        subprocess.run(["scancel", job], check=True)
        plan.setdefault("canceled", []).append(job)
        save()
    plan["state"] = "complete"
    save()
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == "__main__":
    main()
