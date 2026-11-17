"""Metadata/syntax checks only; no model or scientific rendering is executed."""

import argparse
import ast
import runpy
import subprocess
from pathlib import Path

from static_worker import read, sha


def main(run):
    for name in ("training-v1", "evaluation-v1", "landscapes-v1", "static-v1"):
        profile = run / "source/profiles" / name
        subprocess.run(["sha256sum", "--quiet", "-c", str(profile / "manifest.sha256")], check=True)
        for path in profile.rglob("*.sh"):
            subprocess.run(["bash", "-n", str(path)], check=True)
        for path in profile.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
        print(f"Verified frozen manifest, Python syntax and Bash syntax: {name}")
    subprocess.run(["sha256sum", "--quiet", "-c", str(run / "operational-inputs.sha256")], check=True)
    assert "jobs" not in (run / "operational-inputs.sha256").read_text()
    program = read(run / "study.json")
    evaluator = runpy.run_path(str(run / "source/profiles/evaluation-v1/misc/tar_compute_evaluation.py"))
    assert evaluator["select_cases"](program) == program["bundles"]["attacks"]
    assert len(program["evaluation_cases"]) == 4
    print("Operational evaluator defaults to exactly four attack cases; defense endpoint is attack step0.")
    numerical = run / "source/profiles/landscapes-v1/tar_dynamics.py"
    assert sha(numerical) == sha(run / "source/misc/tar_dynamics.py") == "e8ab333af2f9106204ba0bb17a66d7a0adb21e80dd2970d16fb9aa8f1c953054"
    for plan_name, expected in (("operational-dryrun-plan.json", 10), ("static-dryrun-plan.json", 5), ("static-dryrun-plan-v2.json", 5)):
        plan = read(run / plan_name)
        assert plan["state"] == "dryruns-complete-not-submitted" and len(plan["jobs"]) == expected
        for item in plan["jobs"]:
            assert "--dryrun" in item["argv_dryrun"]
            assert "--torchrun=false" in item["argv_dryrun"] and "--bank=verival" in item["argv_dryrun"]
            assert "No jobs were launched." in Path(item["dryrun_log"]).read_text()
            for filename in ("launch.sh", "subrunner.sh"):
                subprocess.run(["bash", "-n", str(Path(item["launch_directory"]) / filename)], check=True)
        print(f"Verified {expected} launcher dry runs only: {plan_name}")
    defenses = {item["case"]: item["job_id"] for item in read(run / "defense-jobs-v2.json")["jobs"]}
    plan = read(run / "static-dryrun-plan-v2.json")
    for item in plan["jobs"]:
        dependencies = [value for value in item["argv_dryrun"] if value.startswith("--dependency=")]
        if item["name"] == "static-comparison-v2":
            assert not dependencies
        else:
            condition = item["name"].removeprefix("static-").removesuffix("-v2")
            assert dependencies == [f"--dependency=afterok:{defenses[f'hparam-{condition}-defense']}"]
    print("Static v2 dependencies match replacement defenses; frozen scientific inputs unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run.resolve())
