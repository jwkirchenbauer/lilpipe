"""Checkpoint resolution, paired evaluation, and guarded trajectory deletion."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

WMDP_TASKS = (("wmdp_bio_robust", "wmdp-bio-robust"), ("mmlu_no_bio", "mmlu-no-bio"))
BEAVERTAILS_MODELS = {
    "llama2-7b-beavertail--align-fft-seam-author-codebase--data-disjoint--relearn-fft-ms1000",
    "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--relearn-fft-ms250",
}
WMDP_MODELS = {
    "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--" + suffix
    for suffix in (
        "relearn-fft-lr1e-6-ms1000",
        "relearn-fft-lr5e-6-ms1000",
        "relearn-fft-lr1e-5-ms1000",
        *(
            f"relearn-perturb-and-fft-sigma{s}-lr1e-5-ms1000"
            for s in ("1e-4", "1e-3", "1e-2", "1e-1")
        ),
    )
}


def checkpoint_steps(value: str) -> tuple[int, ...]:
    """Read an ordered, explicit checkpoint selection."""
    try:
        steps = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoint steps must be integers") from error
    if (
        not steps
        or any(step < 0 for step in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise argparse.ArgumentTypeError(
            "checkpoint steps must be nonnegative and strictly increasing"
        )
    return steps


def cached_model(model: str) -> str:
    """Resolve a local model or the cached main revision of a Hub model."""
    if (Path(model) / "config.json").is_file():
        return model
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", cache_home / "huggingface"))
    cache = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    revision = (cache / "refs/main").read_text().strip()
    snapshot = cache / "snapshots" / revision
    if not (snapshot / "config.json").is_file():
        raise ValueError(f"missing cached model snapshot: {snapshot}")
    return str(snapshot)


def checkpoint_model(
    baseline: str,
    trajectory: str,
    step: int,
    *,
    adapter: bool = False,
    final_step: int | None = None,
    final_model: str = "-",
) -> str:
    """Resolve a checkpoint without substituting another training step."""
    filename = "adapter_config.json" if adapter else "config.json"
    checkpoint = Path(trajectory) / f"checkpoint-{step}"
    if (checkpoint / filename).is_file():
        return str(checkpoint)
    if step == 0:
        return baseline if adapter else cached_model(baseline)
    if (
        step == final_step
        and final_model != "-"
        and (Path(final_model) / filename).is_file()
    ):
        return final_model
    raise ValueError(f"missing model checkpoint: {checkpoint}")


def model_arguments(model: str, adapter: str) -> str:
    arguments = f"pretrained={model},dtype=bfloat16,trust_remote_code=True"
    if adapter != "-":
        arguments += f",peft={adapter}"
    return arguments


def wmdp_points(args) -> list[tuple[str, Path]]:
    """Resolve the model arguments and destination for each Accelerate launch."""
    output = args.result_root / args.model_id
    if args.mode == "wmdp-single":
        model, adapter = args.model, args.adapter
        if args.checkpoint_step is not None:
            selected = checkpoint_model(
                args.model,
                args.adapter if args.adapter != "-" else args.model,
                args.checkpoint_step,
                adapter=args.adapter != "-",
                final_step=args.checkpoint_step,
                final_model=args.final_model,
            )
            if adapter == "-" or not (Path(selected) / "adapter_config.json").is_file():
                model = selected
                adapter = "-"
            else:
                adapter = selected
            output /= f"checkpoint-{args.checkpoint_step}"
        return [(model_arguments(model, adapter), output)]
    points = []
    for step in args.steps:
        selected = checkpoint_model(
            args.baseline,
            args.trajectory,
            step,
            adapter=args.adapter != "-",
            final_step=args.final_step,
            final_model=args.final_model,
        )
        has_adapter = args.adapter != "-" and (
            step != 0 or (Path(selected) / "adapter_config.json").is_file()
        )
        model = args.model if has_adapter else selected
        adapter = selected if has_adapter else "-"
        points.append((model_arguments(model, adapter), output / f"checkpoint-{step}"))
    return points


def publish_wmdp(source: Path, destination: Path) -> None:
    """Preserve both task-directory layouts from one combined lm-eval run."""
    files = list(source.rglob("results*.json"))
    if len(files) != 1:
        raise ValueError(
            f"expected one combined lm-eval result, found {len(files)}: {source}"
        )
    path = files[0]
    result = json.loads(path.read_text())
    for task, _ in WMDP_TASKS:
        if not finite_score(result.get("results", {}).get(task, {}).get("acc,none")):
            raise ValueError(f"missing {task} score in combined evaluation: {path}")
    for _, directory in WMDP_TASKS:
        target = destination / directory / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def evaluate_beavertails(args) -> None:
    """Evaluate checkpoint waves on the allocated GPUs and join every worker."""
    gpu_count = int(os.environ["SLURM_GPUS_ON_NODE"])
    if gpu_count < 1:
        raise ValueError("at least one allocated GPU is required")
    for start in range(0, len(args.steps), gpu_count):
        workers = []
        try:
            for gpu, step in enumerate(args.steps[start : start + gpu_count]):
                checkpoint = checkpoint_model(
                    args.baseline,
                    args.trajectory,
                    step,
                    final_step=args.final_step,
                    final_model=args.final_model,
                )
                if (
                    step == 0
                    and not (
                        Path(args.trajectory) / "checkpoint-0/config.json"
                    ).is_file()
                ):
                    # Reuse the evaluator's cached baseline result at step zero.
                    checkpoint = args.baseline
                command = [
                    sys.executable,
                    "scripts/evals/evaluate_beavertail_and_capabilities.py",
                    "trajectory",
                    args.model_id,
                    args.baseline,
                    args.trajectory,
                    args.learning_rate,
                    args.attack_data,
                    args.method,
                    str(args.result_root),
                    "--checkpoint-step",
                    str(step),
                    "--checkpoint-path",
                    checkpoint,
                ]
                if args.gaussian_sigma is not None:
                    command.extend(
                        [
                            "--gaussian-sigma",
                            args.gaussian_sigma,
                            "--gaussian-seed",
                            args.gaussian_seed,
                        ]
                    )
                workers.append(
                    subprocess.Popen(
                        command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    )
                )
            codes = [worker.wait() for worker in workers]
            if any(codes):
                raise RuntimeError(f"BeaverTails evaluation failed: {codes}")
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
                    worker.wait()


def finite_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def verify_results(args) -> None:
    """Require parseable task results for every selected checkpoint."""
    for step in args.steps:
        output = args.result_root / args.model_id / f"checkpoint-{step}"
        if args.mode == "beavertails-trajectory":
            summary = json.loads((output / "summary.json").read_text())
            if (
                summary.get("model_id") != args.model_id
                or summary.get("checkpoint_step") != step
            ):
                raise ValueError(f"mismatched checkpoint summary: {output}")
            if (
                summary.get("attacked_model") != args.trajectory
                or summary.get("baseline_model") != args.baseline
            ):
                raise ValueError(f"mismatched model summary: {output}")
            for phase in ("pre", "post"):
                result = summary.get(phase, {})
                values = [result.get("harmfulness"), result.get("capability")]
                values += [
                    result.get("tasks", {}).get(task) for task in ("mmlu", "arc_easy")
                ]
                if not all(finite_score(value) for value in values):
                    raise ValueError(f"incomplete BeaverTails summary: {output}")
        else:
            selected = checkpoint_model(
                args.baseline,
                args.trajectory,
                step,
                adapter=args.adapter != "-",
                final_step=args.final_step,
                final_model=args.final_model,
            )
            has_adapter = args.adapter != "-" and (
                step != 0 or (Path(selected) / "adapter_config.json").is_file()
            )
            expected_model = args.model if has_adapter else selected
            for task, directory in WMDP_TASKS:
                files = list((output / directory).rglob("results*.json"))
                if not files:
                    raise ValueError(f"missing {task} results: {output}")
                for path in files:
                    result = json.loads(path.read_text())
                    model_args = result.get("config", {}).get("model_args", {})
                    if isinstance(model_args, str):
                        model_args = dict(
                            item.split("=", 1)
                            for item in model_args.split(",")
                            if "=" in item
                        )
                    expected_adapter = selected if has_adapter else None
                    matches = (
                        isinstance(model_args, dict)
                        and model_args.get("pretrained") == expected_model
                        and model_args.get("peft") == expected_adapter
                    )
                    if matches and finite_score(
                        result.get("results", {}).get(task, {}).get("acc,none")
                    ):
                        break
                else:
                    raise ValueError(f"incomplete {task} results: {output}")


def delete_trajectory(args) -> None:
    """Delete only an allowed, fully evaluated attack trajectory."""
    if not args.delete_trajectory:
        return
    if args.steps[0] != 0 or args.steps[-1] != args.final_step:
        raise ValueError("cleanup requires checkpoint zero and the terminal step")
    project = Path.cwd().resolve()
    path = Path(args.trajectory)
    if path.is_absolute() or ".." in path.parts or path.name != args.model_id:
        raise ValueError(f"refusing mismatched trajectory: {path}")
    allowed = (
        BEAVERTAILS_MODELS if args.mode == "beavertails-trajectory" else WMDP_MODELS
    )
    canonical = path.parent == Path("artifacts/tamia/models") and path.name in allowed
    generated = (
        args.mode == "beavertails-trajectory"
        and len(path.parts) == 6
        and path.parts[:3] == ("artifacts", "generated", "hpo")
        and path.parts[4] == "models"
        and path.name.startswith(path.parts[3] + "--")
        and "--relearn-" in path.name
    )
    if not (canonical or generated):
        raise ValueError(f"refusing unexpected trajectory: {path}")
    # Artifact roots themselves may be mounted/symlinked to project storage. No
    # component below the permitted model root may redirect deletion elsewhere.
    model_root = path.parent.resolve()
    resolved = path.resolve(strict=True)
    if path.is_symlink() or resolved.parent != model_root:
        raise ValueError(f"refusing redirected trajectory: {path}")
    baseline = Path(args.baseline).resolve()
    results = args.result_root.resolve()
    if (
        baseline == resolved
        or resolved in baseline.parents
        or results == resolved
        or resolved in results.parents
    ):
        raise ValueError("refusing to delete baseline models or evaluation results")
    for child in resolved.rglob("*"):
        if child.is_symlink() and not child.resolve().is_relative_to(resolved):
            raise ValueError(f"refusing escaping symlink: {child}")
    if not resolved.is_dir() or resolved == project:
        raise ValueError(f"missing trajectory directory: {path}")
    verify_results(args)
    shutil.rmtree(resolved)
    print(f"Removed fully evaluated trajectory: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("wmdp-single", "wmdp-trajectory", "beavertails-trajectory"):
        command = modes.add_parser(mode)
        command.add_argument("model_id")
        if mode == "wmdp-single":
            command.add_argument("model")
            command.add_argument("adapter")
            command.add_argument("result_root", type=Path)
            command.add_argument("--checkpoint-step", type=int)
        else:
            command.add_argument("baseline")
            command.add_argument("trajectory")
            if mode == "wmdp-trajectory":
                command.add_argument("model", help="base model for adapter checkpoints")
                command.add_argument("adapter", help="'-' for full model checkpoints")
            else:
                command.add_argument("learning_rate")
                command.add_argument("attack_data")
                command.add_argument("method")
                command.add_argument("--gaussian-sigma")
                command.add_argument("--gaussian-seed")
            command.add_argument("result_root", type=Path)
            command.add_argument("steps", type=checkpoint_steps)
            command.add_argument("--final-step", type=int, required=True)
            command.add_argument("--delete-trajectory", action="store_true")
            command.add_argument("--cleanup-only", action="store_true")
        command.add_argument("--final-model", default="-")
        if mode.startswith("wmdp-"):
            command.add_argument("--plan-file", type=Path)

    publish = modes.add_parser("publish-wmdp")
    publish.add_argument("source", type=Path)
    publish.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "publish-wmdp":
        return args
    if (
        not args.model_id
        or Path(args.model_id).name != args.model_id
        or args.model_id in (".", "..")
    ):
        parser.error("model id must be a single path component")
    if (
        args.mode == "wmdp-single"
        and args.checkpoint_step is not None
        and args.checkpoint_step < 0
    ):
        parser.error("checkpoint step must be nonnegative")
    if args.mode != "wmdp-single" and (
        args.final_step < args.steps[-1] or args.final_step < 1
    ):
        parser.error("final step must be positive and no smaller than requested steps")
    if args.mode == "beavertails-trajectory" and (
        (args.gaussian_sigma is None) != (args.gaussian_seed is None)
    ):
        parser.error("Gaussian sigma and seed must be provided together")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.mode == "publish-wmdp":
        publish_wmdp(args.source, args.destination)
    elif args.mode != "wmdp-single" and args.cleanup_only:
        delete_trajectory(args)
    elif args.mode == "beavertails-trajectory":
        evaluate_beavertails(args)
    else:
        if args.plan_file is None:
            raise ValueError("--plan-file is required; invoke the WMDP sbatch wrapper")
        with args.plan_file.open("wb") as handle:
            for arguments, destination in wmdp_points(args):
                handle.write(
                    arguments.encode() + b"\0" + str(destination).encode() + b"\0"
                )


if __name__ == "__main__":
    main()
