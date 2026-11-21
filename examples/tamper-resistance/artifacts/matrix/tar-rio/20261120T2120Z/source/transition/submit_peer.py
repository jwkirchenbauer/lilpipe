"""Submit one reviewed peer-landscape analysis on Matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1"
LAUNCHER = "/p/vast1/kirchenb/llnl-tools/launch_matrix.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def passing_pilot(pilot_path):
    pilot = json.loads(Path(pilot_path).read_text())
    gates = pilot.get("gates")
    if pilot.get("state") != "complete" or not isinstance(gates, dict) or not gates:
        raise ValueError("Pilot must be complete and contain passing gates")
    if any(value is not True for value in gates.values()):
        raise ValueError("Pilot gates are not all passing")
    return pilot


def validate_case(run, case):
    folder = run / "cases" / case
    for name in ("status", "logs", "cache", "models", "geometry", "surfaces"):
        if not (folder / name).is_dir():
            raise ValueError(f"Missing prepared case directory: {folder / name}")
    for name in ("study.json", "case.json"):
        if not (folder / name).is_file():
            raise ValueError(f"Missing prepared case metadata: {folder / name}")
    train_path = folder / "status" / "train.json"
    if (
        not train_path.is_file()
        or json.loads(train_path.read_text()).get("state") != "complete"
    ):
        raise ValueError(f"Training is not complete: {train_path}")
    return folder


def build_command(
    run, case, pilot, dependencies, dependency_mode, minutes, attempt="peer-v2"
):
    command = [
        sys.executable,
        LAUNCHER,
        "--run_name=tar-rio-analysis-" + case + "-" + attempt,
        "--partition=pbatch",
        "--account=verival",
        "--nodes=1",
        "--tasks_per_node=1",
        "--gpus_per_node=2",
        "--gpus_per_task=2",
        "--cpus_per_task=56",
        "--exclusive=false",
        "--torchrun=false",
        "--launch_once_per_node=true",
        "--environment=" + EVAL,
        "--env_act_style=mm_activate",
        "--cuda_module=",
        "--minutes=" + str(minutes),
        "--output_dir=" + str(ROOT / "launches"),
        "--sub_output_dir_name=analysis-" + case + "-" + attempt,
        "--pass_run_name=false",
        "--quiet_launch=true",
        "--squonce=false",
    ]
    if dependencies:
        command.append("--dependency=" + dependency_mode + ":" + ":".join(dependencies))
    target = [
        "bash",
        "-i",
        str(ROOT / "source/peer-landscape/analysis.sh"),
        str(run),
        case,
        str(pilot),
    ]
    return command + ["--", *target]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--attempt", default="peer-v2")
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--dependency", nargs="*", default=[])
    parser.add_argument(
        "--dependency-mode", choices=("afterok", "afterany"), default="afterok"
    )
    parser.add_argument("--minutes", type=int, default=360)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    if not args.run.is_absolute() or not args.pilot.is_absolute():
        parser.error("--run and --pilot must be absolute paths")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.case) or args.case in (".", ".."):
        parser.error("--case must be a safe case name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.attempt) or args.attempt in (
        ".",
        "..",
    ):
        parser.error("--attempt must be a safe suffix")
    if args.minutes < 1:
        parser.error("--minutes must be positive")
    if not args.pilot.is_file():
        parser.error(f"Pilot result does not exist: {args.pilot}")
    if any(not job.isdecimal() for job in args.dependency):
        parser.error("Dependency IDs must be numeric Slurm job IDs")

    pilot_data = passing_pilot(args.pilot)
    folder = validate_case(args.run, args.case)
    pilot_digest = sha256(args.pilot)
    key = "analysis:" + args.case + ":" + args.attempt
    ledger_path = ROOT / "matrix-jobs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if key in ledger and not args.dryrun:
        raise ValueError("Already submitted: " + key)

    command = build_command(
        args.run,
        args.case,
        args.pilot,
        args.dependency,
        args.dependency_mode,
        args.minutes,
        args.attempt,
    )
    if args.dryrun:
        command.insert(command.index("--"), "--dryrun")
    environment = os.environ.copy()
    environment["SBATCH_MEM_PER_NODE"] = "224G"
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(result.stdout, end="", flush=True)
    log = (
        ROOT
        / "logs"
        / (
            "analysis-"
            + args.case
            + "-"
            + args.attempt
            + "-"
            + ("dryrun" if args.dryrun else "submit")
            + ".log"
        )
    )
    log.write_text(result.stdout)
    if args.dryrun:
        return

    match = re.search(r"with job_id=(\d+)\.", result.stdout)
    if not match:
        raise RuntimeError("No submitted job ID in launcher output")
    old_parent_ids = list(args.dependency)
    status_file = folder / "status" / "analysis.json"
    ledger[key] = {
        "job_id": match.group(1),
        "backend": "slurm",
        "cluster": "matrix",
        "partition": "pbatch",
        "account": "verival",
        "phase": "analysis-peer",
        "case": args.case,
        "attempt": args.attempt,
        "run": str(args.run),
        "minutes": args.minutes,
        "gpus": 2,
        "cpus": 56,
        "host_memory": "224G",
        "environment": EVAL,
        "status_file": str(status_file),
        "peer_status_pattern": str(folder / "status" / "analysis-peer-<job_id>.json"),
        "pilot_result": str(args.pilot),
        "pilot_result_sha256": pilot_digest,
        "pilot_state": pilot_data["state"],
        "dependencies": old_parent_ids,
        "dependency_mode": args.dependency_mode,
        "old_parent_job_ids": old_parent_ids,
        "argv": command,
        "log": str(log),
    }
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(ledger_path)


if __name__ == "__main__":
    main()
