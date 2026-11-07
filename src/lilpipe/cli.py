"""Command-line interface for lilpipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import yaml

from .core import PipelineError, Plan, load, submit
from .results import load_results


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config", help="Pipeline YAML file, relative to the current directory"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-dir",
        help="Archive submission commands, job IDs, and source snapshots",
    )
    parser.add_argument("--skip", nargs="+", default=None, metavar="MODEL_OR_STAGE")
    parser.add_argument(
        "--existing-models",
        nargs="+",
        action="append",
        default=None,
        metavar="MODEL",
        help="Reuse existing model artifacts and skip their producer jobs",
    )
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--evaluations", nargs="+", default=None, metavar="EVALUATION")
    parser.add_argument(
        "--sbatch-args",
        default="",
        help="Extra arguments forwarded to every sbatch call as one quoted string",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lilpipe", description="Compile Slurm pipelines and collect results"
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="compile or submit a pipeline")
    _add_pipeline_arguments(run_parser)
    results_parser = subparsers.add_parser("results", help="collect experiment results")
    results_parser.add_argument(
        "config", help="Results YAML file, relative to the current directory"
    )
    results_parser.add_argument(
        "--format", choices=("markdown", "csv", "json"), default="markdown"
    )
    return parser


def submit_recorded(
    plan: Plan, config: str, directory: str, extra_args: list[str]
) -> None:
    """Record each submission immediately, including partially submitted pipelines."""
    root = Path(directory)
    try:
        root.mkdir(parents=True, exist_ok=False)
        source = Path(config).resolve()
        manifest = yaml.safe_load(source.read_text())
        files = {source, *(Path(p).resolve() for p in manifest["registries"].values())}
        if (source.parent / "runs.yml").is_file():
            files.add(source.parent / "runs.yml")
        for stage in plan.stages:
            files.add(Path(stage.script).resolve())
            for arg in stage.args:
                if arg.endswith((".yml", ".yaml")) and Path(arg).is_file():
                    files.add(Path(arg).resolve())
        snapshots = {}
        for path in sorted(files):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                relative = path.relative_to(plan.root)
            except ValueError:
                relative = Path("external") / f"{digest[:12]}-{path.name}"
            target = root / "submitted" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            snapshots[str(path)] = {"path": str(target), "sha256": digest}
    except OSError as error:
        raise PipelineError(f"Cannot create submission record: {error}") from error
    record = {
        "pipeline": plan.id,
        "sources": snapshots,
        "jobs": [],
        "status": "submitting",
    }
    report = root / "jobs.json"
    try:
        version = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=plan.root,
            capture_output=True,
            text=True,
            check=True,
        )
        changes = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=plan.root,
            capture_output=True,
            text=True,
            check=True,
        )
        record["git_commit"] = version.stdout.strip()
        (root / "working-tree.patch").write_text(changes.stdout)
    except (OSError, subprocess.CalledProcessError):
        record["git_commit"] = None

    def save() -> None:
        report.write_text(json.dumps(record, indent=2) + "\n")

    def runner(command: list[str]) -> str:
        entry = {"command": command}
        record["jobs"].append(entry)
        save()
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                raise ValueError(f"Invalid sbatch job ID: {result.stdout!r}")
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            entry["error"] = str(error)
            record["status"] = "partial submission"
            save()
            raise PipelineError(f"Submission failed; see {report}: {error}") from error
        entry["job_id"] = job_id
        save()
        return job_id

    save()
    submit(plan, extra_sbatch_args=extra_args, run=runner)
    record["status"] = "submitted"
    save()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments
        and not arguments[0].startswith("-")
        and arguments[0] not in {"run", "results"}
    ):
        arguments.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "results":
            print(load_results(args.config).collect().render(args.format))
            return 0
        pipeline = load(args.config).select(
            models=args.models, evaluations=args.evaluations
        )
        existing_models = tuple(
            model for group in (args.existing_models or ()) for model in group
        )
        plan = pipeline.plan(skip=args.skip or (), existing_models=existing_models)
        extra_sbatch_args = shlex.split(args.sbatch_args)
        if args.dry_run:
            if existing_models:
                print(f"# Reusing existing models: {', '.join(existing_models)}")
            rendered = plan.render(extra_sbatch_args=extra_sbatch_args)
            if rendered:
                print(rendered)
        elif args.record_dir:
            submit_recorded(plan, args.config, args.record_dir, extra_sbatch_args)
        else:
            submit(plan, extra_sbatch_args=extra_sbatch_args)
    except PipelineError as error:
        print(f"lilpipe: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
