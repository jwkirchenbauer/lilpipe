"""Submit the declarative SEAM noise workflow and plot after all evaluations."""

import argparse
import json
from pathlib import Path
import subprocess

from lilpipe import Plan, Stage, load, submit


def build_plan(config, existing_models=()):
    """Add the final plotting dependency using lilpipe's public stage API."""
    plan = load(config).plan(existing_models=existing_models)
    plot = Stage(
        id="plot-seam-noise-relearning",
        script="scripts/misc/plot_seam_relearning.sbatch",
        depends_on=tuple(s.id for s in plan.stages if s.id.startswith("eval-")),
    )
    return Plan(plan.id, plan.root, (*plan.stages, plot), plan.skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-models", nargs="*", default=[])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    plan = build_plan(args.config, args.existing_models)
    if args.dry_run:
        print(plan.render())
        return

    run_dir = args.run_dir or Path("artifacts/tamia/runs") / plan.id
    run_dir.mkdir(parents=True, exist_ok=False)
    Path("artifacts/tamia/logs").mkdir(parents=True, exist_ok=True)
    record = {"config": args.config, "commands": plan.render(), "jobs": []}
    jobs_path = run_dir / "jobs.json"
    jobs_path.write_text(json.dumps(record, indent=2) + "\n")

    def record_submission(command):
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        print(f"Submitted {job_id}", flush=True)
        record["jobs"].append({"job_id": job_id, "command": command})
        jobs_path.write_text(json.dumps(record, indent=2) + "\n")
        return job_id

    submit(plan, run=record_submission)


if __name__ == "__main__":
    main()
