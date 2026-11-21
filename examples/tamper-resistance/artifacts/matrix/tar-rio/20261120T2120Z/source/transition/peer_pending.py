"""Prepare or submit peer replacements; dependency rewiring is a separate step."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dryrun", "submit"))
    args = parser.parse_args()
    ledger = json.loads((ROOT / "matrix-jobs.json").read_text())
    gate = ledger["analysis:rio-k4-lr1e-6:peer"]["job_id"]
    pilot = ROOT / "pilots/tar-rio-peer-landscape-pilot/results-516308/result.json"
    plan = {"gate": gate, "pilot": str(pilot), "candidates": [], "deferred": []}
    for key, row in ledger.items():
        if row.get("superseded") or row["phase"] != "analysis":
            continue
        name = row["case"]
        if "analysis:" + name + ":peer" in ledger:
            continue
        folder = Path(row["run"]) / "cases" / name
        status = folder / "status/train.json"
        if not status.exists() or json.loads(status.read_text()).get("state") != "complete":
            plan["deferred"].append(key)
            continue
        command = [
            sys.executable,
            str(ROOT / "source/transition/submit_peer.py"),
            "--run", row["run"], "--case", name, "--pilot", str(pilot),
            "--dependency", gate,
        ]
        plan["candidates"].append(dict(old_key=key, old_job=row["job_id"], argv=command))
    target = ROOT / "source/transition" / ("peer-pending-" + args.action + ".json")
    target.write_text(json.dumps(plan, indent=2) + "\n")
    for candidate in plan["candidates"]:
        command = candidate["argv"] + (["--dryrun"] if args.action == "dryrun" else [])
        print(candidate["old_key"], flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
