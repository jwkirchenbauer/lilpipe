"""Restore the reviewed analysis DAG after the explicit-binding pilot."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "retry-v2"
PEER_ATTEMPT = "peer-v2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("analyses", "profiles", "report"))
    parser.add_argument("action", choices=("dryrun", "submit"))
    args = parser.parse_args()
    ledger = json.loads((ROOT / "matrix-jobs.json").read_text())
    gate = ledger["analysis:rio-k4-lr1e-6:" + PEER_ATTEMPT]
    assert not gate.get("superseded")
    plan_path = ROOT / "source/transition" / ("restored-" + args.stage + "-plan.json")
    if args.action == "dryrun":
        commands = []
        for subdir in (".", "control-study", "backend-control"):
            run = (ROOT / subdir).resolve()
            program = json.loads((run / "study.json").read_text())
            for name, case in program["cases"].items():
                if case["role"] != "attack" or name == "rio-k4-lr1e-6":
                    continue
                if args.stage == "analyses":
                    command = [
                        sys.executable,
                        str(ROOT / "source/transition/submit_peer.py"),
                        "--run",
                        str(run),
                        "--case",
                        name,
                        "--pilot",
                        gate["pilot_result"],
                        "--attempt",
                        PEER_ATTEMPT,
                        "--dependency",
                        gate["job_id"],
                    ]
                elif args.stage == "profiles":
                    reference = ledger["profiles:rio-k4-lr1e-6"]
                    assert (
                        json.loads(Path(reference["status_file"]).read_text())["state"]
                        == "complete"
                    )
                    own = ledger["analysis:" + name + ":" + PEER_ATTEMPT]
                    assert not own.get("superseded")
                    command = [
                        sys.executable,
                        str(ROOT / "source/transition/launch.py"),
                        "profiles",
                        name,
                        "--run",
                        str(run),
                        "--attempt",
                        ATTEMPT,
                        "--minutes",
                        "180",
                        "--dependency",
                        own["job_id"],
                    ]
                else:
                    continue
                commands.append(dict(case=name, argv=command))
        if args.stage == "report":
            dependencies, completed = [], []
            for key, row in ledger.items():
                if row.get("superseded") or row["phase"] == "report":
                    continue
                status = Path(row["status_file"])
                if (
                    status.exists()
                    and json.loads(status.read_text()).get("state") == "complete"
                ):
                    completed.append(
                        dict(key=key, job_id=row["job_id"], status_file=str(status))
                    )
                else:
                    dependencies.append(row["job_id"])
            commands = [
                dict(
                    case="final",
                    argv=[
                        sys.executable,
                        str(ROOT / "source/transition/launch.py"),
                        "report",
                        "final",
                        "--run",
                        str(ROOT),
                        "--attempt",
                        ATTEMPT,
                        "--minutes",
                        "120",
                        "--dependency",
                        *dependencies,
                    ],
                )
            ]
        assert len(commands) == (1 if args.stage == "report" else 9)
        plan = dict(stage=args.stage, reviewed=False, commands=commands)
        if args.stage == "report":
            plan["completed_markers_verified"] = completed
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    else:
        plan = json.loads(plan_path.read_text())
        assert plan["stage"] == args.stage and plan["reviewed"] is True
    for item in plan["commands"]:
        print(item["case"], flush=True)
        subprocess.run(
            item["argv"] + (["--dryrun"] if args.action == "dryrun" else []), check=True
        )


if __name__ == "__main__":
    main()
