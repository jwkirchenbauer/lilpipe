"""Submit one reviewed CUDA follow-up phase through the Matrix launcher."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT.parents[1] / "tar-rio/20261120T2120Z"
TRAIN = "/usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1"
EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1"
PILOT = PREVIOUS / "pilots/tar-rio-peer-landscape-pilot/results-516325/result.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("train", "eval", "analysis", "profiles", "report")
    )
    parser.add_argument("case", nargs="?")
    parser.add_argument("--dependency", nargs="*", default=[])
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()
    program = json.loads((ROOT / "study.json").read_text())
    assert all(value.isdecimal() for value in args.dependency)
    if args.phase == "report":
        assert args.case is None, "Report does not accept a case"
        key, folder = "report", ROOT
    else:
        assert args.case in program["cases"], args.case
        case = program["cases"][args.case]
        assert args.phase == "train" or case["role"] == "attack"
        key = args.phase + ":" + args.case
        folder = ROOT / "cases" / args.case
    ledger_path = ROOT / "matrix-jobs.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    assert key not in ledger or args.dryrun, "Already submitted"
    assert (folder / "source/misc/reference_get_ll.py").is_file()
    prefix = TRAIN if args.phase == "train" else EVAL
    gpus = {"train": 4, "eval": 1, "analysis": 2, "profiles": 1, "report": 1}[
        args.phase
    ]
    minutes = {
        "train": 120,
        "eval": 180,
        "analysis": 360,
        "profiles": 180,
        "report": 60,
    }[args.phase]
    entry = PREVIOUS / "source/transition/entry.sh"
    if args.phase == "report":
        target = ["bash", "-i", str(ROOT / "ops/run_report.sh")]
    elif args.phase == "eval":
        target = [
            "bash",
            "-i",
            str(PREVIOUS / "source/transition/eval_onegpu.sh"),
            str(ROOT),
            args.case,
        ]
    elif args.phase == "analysis":
        pilot = json.loads(PILOT.read_text())
        assert pilot["state"] == "complete" and all(pilot["gates"].values())
        target = [
            "bash",
            "-i",
            str(PREVIOUS / "source/peer-landscape/analysis.sh"),
            str(ROOT),
            args.case,
            str(PILOT),
        ]
    else:
        target = ["bash", "-i", str(entry), args.phase, str(ROOT), args.case]
    command = [
        sys.executable,
        "/p/vast1/kirchenb/llnl-tools/launch_matrix.py",
        "--run_name=tar-rio-cuda-" + key.replace(":", "-"),
        "--partition=pbatch",
        "--account=verival",
        "--nodes=1",
        "--tasks_per_node=1",
        "--gpus_per_node=" + str(gpus),
        "--gpus_per_task=" + str(gpus),
        "--cpus_per_task=" + str(28 * gpus),
        "--exclusive=false",
        "--torchrun=false",
        "--launch_once_per_node=true",
        "--environment=" + prefix,
        "--env_act_style=mm_activate",
        "--cuda_module=",
        "--minutes=" + str(minutes),
        "--output_dir=" + str(ROOT / "launches"),
        "--sub_output_dir_name=" + key.replace(":", "-"),
        "--pass_run_name=false",
        "--quiet_launch=true",
        "--squonce=false",
    ]
    if args.dependency:
        command.append("--dependency=afterok:" + ":".join(args.dependency))
    if args.dryrun:
        command.append("--dryrun")
    command.extend(["--", *target])
    environment = os.environ.copy()
    environment["SBATCH_MEM_PER_NODE"] = "224G"
    result = subprocess.run(
        command,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    stem = key.replace(":", "-") + ("-dryrun" if args.dryrun else "-submit")
    (ROOT / "logs" / (stem + ".log")).write_text(result.stdout)
    plan = {
        "argv": command,
        "phase": args.phase,
        "case": args.case,
        "gpus": gpus,
        "host_memory": "224G",
        "dependencies": args.dependency,
        "source_sha256": {
            str(Path(target[2])): hashlib.sha256(
                Path(target[2]).read_bytes()
            ).hexdigest()
        },
    }
    if args.dryrun:
        (ROOT / "ops" / (stem + ".json")).write_text(json.dumps(plan, indent=2) + "\n")
        return
    match = re.search(r"with job_id=(\d+)\.", result.stdout)
    assert match, "Missing scheduler ID"
    status = {
        "train": "train",
        "eval": "evaluation",
        "analysis": "analysis",
        "profiles": "step-profiles",
        "report": "report",
    }[args.phase]
    ledger[key] = {
        **plan,
        "job_id": match.group(1),
        "backend": "slurm",
        "partition": "pbatch",
        "account": "verival",
        "status_file": str(folder / "status" / (status + ".json")),
    }
    temporary = ledger_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(ledger, indent=2) + "\n")
    temporary.replace(ledger_path)


if __name__ == "__main__":
    main()
