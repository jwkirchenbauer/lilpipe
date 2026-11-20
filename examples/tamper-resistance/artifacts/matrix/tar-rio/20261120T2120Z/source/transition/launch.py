"""Submit one reviewed native-Slurm RIO production phase on Matrix."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1"
TRAIN = "/usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1"


def require_case_directories(run, name):
    folder = run / "cases" / name
    for directory in ("status", "logs", "cache", "models"):
        if not (folder / directory).is_dir():
            raise ValueError(f"Missing prepared case directory: {folder / directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("eval", "analysis", "profiles", "train", "pilot", "report")
    )
    parser.add_argument("case")
    parser.add_argument("--run", type=Path, default=ROOT)
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--eval-processes", type=int, choices=(1, 4), default=4)
    parser.add_argument("--dependency", nargs="*", default=[])
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    assert args.run.is_absolute()
    if args.phase not in ("pilot", "report"):
        require_case_directories(args.run, args.case)
    key = args.phase + ":" + args.case
    ledger_path = ROOT / "matrix-jobs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if key in ledger and not args.dryrun:
        raise ValueError("Already submitted: " + key)
    training = args.phase in ("train", "pilot")
    prefix = TRAIN if training else EVAL
    gpus = 4 if args.phase in ("eval", "train", "pilot") else 1
    if args.phase == "eval":
        gpus = args.eval_processes
    minutes = (
        args.minutes
        or {
            "eval": 150,
            "analysis": 240,
            "profiles": 180,
            "train": 240,
            "pilot": 45,
            "report": 120,
        }[args.phase]
    )
    target = [
        "bash",
        "-i",
        str(ROOT / "source/transition/entry.sh"),
        args.phase,
        str(args.run),
        args.case,
    ]
    if args.phase == "eval" and args.eval_processes == 1:
        target = [
            "bash",
            "-i",
            str(ROOT / "source/transition/eval_onegpu.sh"),
            str(args.run),
            args.case,
        ]
    if args.phase == "pilot":
        target = [
            "bash",
            "-i",
            str(ROOT / "source/transition/training_pilot.sh"),
            str(ROOT),
        ]
    elif args.phase == "report":
        assert args.run.resolve() == ROOT and args.case == "final"
        target = [
            "bash",
            "-i",
            str(ROOT / "source/report-matrix/run_report.sh"),
            str(ROOT),
        ]
    command = [
        sys.executable,
        "/p/vast1/kirchenb/llnl-tools/launch_matrix.py",
        "--run_name=tar-rio-" + args.phase + "-" + args.case,
        "--partition=pbatch",
        "--account=verival",
        "--nodes=1",
        "--tasks_per_node=1",
        "--gpus_per_node=" + str(gpus),
        "--exclusive=false",
        "--torchrun=false",
        "--launch_once_per_node=true",
        "--environment=" + prefix,
        "--env_act_style=mm_activate",
        "--cuda_module=",
        "--minutes=" + str(minutes),
        "--output_dir=" + str(ROOT / "launches"),
        "--sub_output_dir_name=" + args.phase + "-" + args.case,
        "--pass_run_name=false",
        "--quiet_launch=true",
        "--squonce=false",
    ]
    if args.dependency:
        assert all(job.isdecimal() for job in args.dependency)
        command.append("--dependency=afterok:" + ":".join(args.dependency))
    if args.dryrun:
        command.append("--dryrun")
    command += ["--", *target]
    environment = os.environ.copy()
    memory = "128G" if args.phase == "eval" and gpus == 1 else "256G"
    environment["SBATCH_MEM_PER_NODE"] = memory
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
            args.phase
            + "-"
            + args.case
            + ("-dryrun" if args.dryrun else "-submit")
            + ".log"
        )
    )
    log.write_text(result.stdout)
    if args.dryrun:
        return
    match = re.search(r"with job_id=(\d+)\.", result.stdout)
    assert match, "No submitted job ID in launcher output"
    phase_file = {"eval": "evaluation", "profiles": "step-profiles"}.get(
        args.phase, args.phase
    )
    status = args.run / "cases" / args.case / "status" / (phase_file + ".json")
    if args.phase == "pilot":
        status = ROOT / "pilots/cuda-training/status/validation.json"
    elif args.phase == "report":
        status = ROOT / "report-matrix-references.json"
    ledger[key] = {
        "job_id": match[1],
        "backend": "slurm",
        "cluster": "matrix",
        "partition": "pbatch",
        "account": "verival",
        "phase": args.phase,
        "case": args.case,
        "run": str(args.run),
        "minutes": minutes,
        "gpus": gpus,
        "evaluation_processes": args.eval_processes if args.phase == "eval" else None,
        "cpus": 28 * gpus,
        "host_memory": memory,
        "environment": prefix,
        "status_file": str(status),
        "dependencies": args.dependency,
        "argv": command,
        "log": str(log),
    }
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(ledger_path)


if __name__ == "__main__":
    main()
