"""Scheduling-only replacement of unsubmitted static launcher dry runs."""

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from prepare_static import EVAL
from static_worker import dump, read, sha


def main(run):
    assert Path(os.environ["CONDA_PREFIX"]).resolve() == Path(EVAL).resolve()
    result = run / "static-dryrun-plan-v2.json"
    assert not result.exists()
    plan = read(run / "static-dryrun-plan.json")
    defenses = {item["case"]: item["job_id"] for item in read(run / "defense-jobs-v2.json")["jobs"]}
    for item in plan["jobs"]:
        old_name = item["name"]
        item["name"] = old_name + "-v2"
        directory = run / "launches" / item["name"]
        assert not (directory / "launch.sh").exists()
        argv = []
        for argument in item["argv_dryrun"]:
            if argument.startswith("--dependency="):
                continue
            if argument.startswith("--sub_output_dir_name="):
                argument = f"--sub_output_dir_name={item['name']}"
            if argument.startswith("--run_name="):
                argument = f"--run_name=tar-{item['name']}"
            argv.append(argument)
        if old_name != "static-comparison":
            condition = old_name.removeprefix("static-")
            argv.insert(argv.index("--"), f"--dependency=afterok:{defenses[f'hparam-{condition}-defense']}")
        assert "--dryrun" in argv
        log = run / "logs" / f"dryrun-{item['name']}.log"
        with log.open("x") as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
        text = log.read_text()
        assert "No jobs were launched." in text
        line = next(line for line in text.splitlines() if line.startswith("Submission: "))
        for filename in ("launch.sh", "subrunner.sh"):
            subprocess.run(["bash", "-n", str(directory / filename)], check=True)
        item.update(argv_dryrun=argv, launch_directory=str(directory), dryrun_log=str(log), submission_argv_after_review=shlex.split(line.removeprefix("Submission: ")))
        if old_name == "static-comparison":
            item["requires_completed"] = [name + "-v2" for name in item["requires_completed"]]
        print(f"DRY RUN ONLY: {item['name']}", flush=True)
    plan.update(supersedes=str(run / "static-dryrun-plan.json"), scheduling_source=str(run / "defense-jobs-v2.json"), scheduling_source_sha256=sha(run / "defense-jobs-v2.json"), scientific_configs_and_sources_changed=False)
    dump(result, plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    main(parser.parse_args().run.resolve())
