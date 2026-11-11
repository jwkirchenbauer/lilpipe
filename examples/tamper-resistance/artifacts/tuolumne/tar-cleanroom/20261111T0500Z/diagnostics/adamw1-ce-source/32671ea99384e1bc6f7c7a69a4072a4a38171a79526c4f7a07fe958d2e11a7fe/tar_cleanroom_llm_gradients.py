"""Exact versus first-order gradients for a short actual-model attack.

Run only through launch_tuo.py. This optional diagnostic uses whole-batch
forget8 updates and the first eight heldout documents for the terminal outer
loss. It is separate from the six-case defense study. No parameters are saved
or changed. The SGD1 default is the intended first actual-7B experiment;
AdamW and longer unrolls require more memory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def sibling_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers():
    return (
        sibling_module("tar_cleanroom.py", "tar_cleanroom_llm_runtime"),
        sibling_module("tar_cleanroom_gradients.py", "tar_cleanroom_llm_functional"),
    )


def atomic_bytes(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dump(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(run, checkpoint, output, clean, gradients):
    """Archive this executing version separately from the original run source."""
    files = [Path(__file__), Path(clean.__file__), Path(gradients.__file__)]
    contents = {path.name: path.read_bytes() for path in files}
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in contents.items()
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    archive = output.parent / f"{output.stem}-source" / identity
    for name, value in contents.items():
        target = archive / name
        if target.exists() and sha256(target) != hashes[name]:
            raise ValueError(f"Executing source archive changed: {target}")
        if not target.exists():
            atomic_bytes(target, value)
    source = {"archive": str(archive), "sha256": identity, "files": hashes}
    dump(archive / "manifest.json", source)
    return {
        "source": source,
        "study_sha256": sha256(run / "study.json"),
        "original_source_manifest_sha256": sha256(run / "source-manifest.json"),
        "data_sha256": sha256(run / "data.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(checkpoint.iterdir())
            if path.is_file() and not path.name.startswith(".")
        },
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "datasets")
        },
        "launcher": {
            key: os.environ[key]
            for key in ("FLUX_JOB_ID", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
            if key in os.environ
        },
    }


def gradient_to_cpu(result):
    """Release accelerator gradient storage progressively during offload."""
    gradient = {}
    for name in list(result.gradient):
        value = result.gradient.pop(name)
        gradient[name] = value.detach().to("cpu")
        del value
    return gradient


def gradient_statistics(exact, first, chunk_elements=1_048_576):
    """Accumulate global norms and dot products in bounded CPU FP64 chunks."""
    import torch

    if exact.keys() != first.keys():
        raise ValueError("Exact and first-order parameter names differ")
    exact_squared, first_squared, dot, difference_squared = 0.0, 0.0, 0.0, 0.0
    for name, exact_value in exact.items():
        first_value = first[name]
        if exact_value.device.type != "cpu" or first_value.device.type != "cpu":
            raise ValueError("Gradient comparison expects CPU-offloaded maps")
        if exact_value.shape != first_value.shape:
            raise ValueError(f"Gradient shapes differ: {name}")
        left, right = exact_value.reshape(-1), first_value.reshape(-1)
        for start in range(0, left.numel(), chunk_elements):
            x = left[start : start + chunk_elements].double()
            y = right[start : start + chunk_elements].double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Nonfinite meta-gradient: {name}")
            exact_squared += float(x.square().sum())
            first_squared += float(y.square().sum())
            dot += float((x * y).sum())
            difference_squared += float((x - y).square().sum())
    exact_norm, first_norm = math.sqrt(exact_squared), math.sqrt(first_squared)
    return {
        "exact_gradient_norm": exact_norm,
        "first_order_gradient_norm": first_norm,
        "gradient_dot_product": dot,
        "gradient_cosine": (
            max(-1.0, min(1.0, dot / (exact_norm * first_norm)))
            if exact_norm and first_norm
            else None
        ),
        "gradient_difference_l2": math.sqrt(difference_squared),
        "first_order_relative_l2_error": (
            math.sqrt(difference_squared) / exact_norm if exact_norm else None
        ),
        "first_order_to_exact_norm_ratio": (
            first_norm / exact_norm if exact_norm else None
        ),
    }


def shifted_parameters(params, cpu_gradient, norm, alpha, chunk_elements=1_048_576):
    """Build one shifted initialization without a full accelerator direction map."""
    import torch

    if params.keys() != cpu_gradient.keys() or norm <= 0 or not math.isfinite(norm):
        raise ValueError("A matching, finite, nonzero direction is required")
    shifted = {}
    squared, projection = 0.0, 0.0
    changed, count = 0, 0
    with torch.no_grad():
        for name, parameter in params.items():
            original = parameter.detach().reshape(-1)
            source = cpu_gradient[name].reshape(-1)
            value = torch.empty_like(parameter, memory_format=torch.contiguous_format)
            flat = value.view(-1)
            for start in range(0, original.numel(), chunk_elements):
                stop = start + chunk_elements
                direction = (
                    source[start:stop].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    / norm
                )
                flat[start:stop] = original[start:stop] + alpha * direction
                difference = flat[start:stop] - original[start:stop]
                squared += float(difference.double().square().sum().cpu())
                projection += float(
                    (difference.double() * direction.double()).sum().cpu()
                )
                changed += int((difference != 0).sum().cpu())
                count += difference.numel()
            shifted[name] = value.requires_grad_()
    realized = math.sqrt(squared)
    return shifted, {
        "requested_signed_l2_step": alpha,
        "realized_l2_step": realized,
        "projection_along_unit_gradient": projection,
        "absolute_direction_cosine": abs(projection) / realized if realized else None,
        "changed_parameter_fraction": changed / count if count else None,
    }


def distance(left, right, chunk_elements=1_048_576):
    """Global L2 distance with only scalar accelerator transfers."""
    if left.keys() != right.keys():
        raise ValueError("Parameter names differ")
    squared = 0.0
    for name, value in left.items():
        x, y = value.detach().reshape(-1), right[name].detach().reshape(-1)
        for start in range(0, x.numel(), chunk_elements):
            delta = (
                x[start : start + chunk_elements] - y[start : start + chunk_elements]
            )
            squared += float(delta.double().square().sum().cpu())
    return math.sqrt(squared)


def numerical_attack(
    gradients,
    model,
    initial,
    batches,
    inner_loss_fn,
    outer_batch,
    outer_loss_fn,
    config,
    outer_metrics_fn=None,
    record_update=False,
):
    """Fresh actual attack values; no higher-order graph or outer graph retained."""
    import torch

    result = gradients.functional_unroll(
        model, initial, batches, inner_loss_fn, config, first_order=True
    )
    forward = gradients.functional_forward(model, result.params, result.buffers)
    with torch.no_grad():
        metrics = (
            outer_metrics_fn(forward, outer_batch)
            if outer_metrics_fn is not None
            else {"outer_loss": float(outer_loss_fn(forward, outer_batch))}
        )
    metrics = {name: float(value) for name, value in metrics.items()}
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("A numerical attack produced nonfinite outer metrics")
    answer = {
        **metrics,
        "inner_losses": [float(value) for value in result.losses],
    }
    if record_update:
        answer["attack_update_l2"] = distance(result.params, initial)
    return answer


def diagnostic(
    gradients,
    model,
    params,
    batches,
    inner_loss_fn,
    outer_batch,
    outer_loss_fn,
    config,
    *,
    directions=("exact", "first-order"),
    epsilons=(0.03, 0.01, 0.003),
    outer_metrics_fn=None,
    progress=None,
):
    """Compare gradient-aligned differences while keeping full gradient maps on CPU."""
    batches = tuple(batches)
    if not batches:
        raise ValueError("At least one attack update is required")
    if any(not math.isfinite(epsilon) or epsilon <= 0 for epsilon in epsilons):
        raise ValueError("Finite-difference radii must be finite and positive")
    if not directions or any(
        name not in {"exact", "first-order"} for name in directions
    ):
        raise ValueError("Directions must select exact and/or first-order gradients")
    notify = progress or (lambda event: None)
    cpu_gradients, meta_values = {}, {}
    for name, first_order in (("exact", False), ("first-order", True)):
        notify({"phase": "meta-gradient", "gradient": name, "state": "started"})
        result = gradients.meta_gradient(
            model,
            params,
            batches,
            inner_loss_fn,
            outer_batch,
            outer_loss_fn,
            config,
            first_order=first_order,
        )
        meta_values[name] = {
            "outer_loss": result.outer_loss,
            "inner_losses": list(result.inner_losses),
        }
        cpu_gradients[name] = gradient_to_cpu(result)
        del result
        gc.collect()
        notify(
            {
                "phase": "meta-gradient",
                "gradient": name,
                "state": "complete",
                **meta_values[name],
            }
        )
    comparison = gradient_statistics(
        cpu_gradients["exact"], cpu_gradients["first-order"]
    )
    baseline = numerical_attack(
        gradients,
        model,
        params,
        batches,
        inner_loss_fn,
        outer_batch,
        outer_loss_fn,
        config,
        outer_metrics_fn=outer_metrics_fn,
        record_update=True,
    )
    notify({"phase": "baseline-replay", "state": "complete", **baseline})
    differences, outer_steps = [], []
    for name in directions:
        norm = comparison[
            "exact_gradient_norm" if name == "exact" else "first_order_gradient_norm"
        ]
        if norm == 0:
            differences.append(
                {"direction": name, "state": "zero gradient; no directional probe"}
            )
            continue
        exact_prediction = (
            comparison["exact_gradient_norm"]
            if name == "exact"
            else comparison["gradient_dot_product"] / norm
        )
        first_prediction = (
            comparison["first_order_gradient_norm"]
            if name == "first-order"
            else comparison["gradient_dot_product"] / norm
        )
        for epsilon in epsilons:
            sides = []
            for sign in (1, -1):
                notify(
                    {
                        "phase": "directional-probe",
                        "direction": name,
                        "epsilon": epsilon,
                        "sign": sign,
                        "state": "started",
                    }
                )
                shifted, geometry = shifted_parameters(
                    params, cpu_gradients[name], norm, sign * epsilon
                )
                value = numerical_attack(
                    gradients,
                    model,
                    shifted,
                    batches,
                    inner_loss_fn,
                    outer_batch,
                    outer_loss_fn,
                    config,
                    outer_metrics_fn=outer_metrics_fn,
                )
                del shifted
                gc.collect()
                sides.append({**value, "parameter_shift": geometry})
                notify(
                    {
                        "phase": "directional-probe",
                        "direction": name,
                        "epsilon": epsilon,
                        "sign": sign,
                        "state": "complete",
                        **value,
                        "parameter_shift": geometry,
                    }
                )
            slope = (sides[0]["outer_loss"] - sides[1]["outer_loss"]) / (2 * epsilon)
            span = (
                sides[0]["parameter_shift"]["projection_along_unit_gradient"]
                - sides[1]["parameter_shift"]["projection_along_unit_gradient"]
            )
            differences.append(
                {
                    "direction": name,
                    "state": "complete",
                    "epsilon_global_l2": epsilon,
                    "plus": sides[0],
                    "minus": sides[1],
                    "finite_difference": slope,
                    "projected_span_finite_difference": (
                        (sides[0]["outer_loss"] - sides[1]["outer_loss"]) / span
                        if span
                        else None
                    ),
                    "exact_directional_derivative": exact_prediction,
                    "first_order_directional_derivative": first_prediction,
                    "exact_absolute_error": abs(slope - exact_prediction),
                    "first_order_absolute_error": abs(slope - first_prediction),
                    "exact_relative_error": abs(slope - exact_prediction)
                    / max(abs(slope), abs(exact_prediction), 1e-12),
                    "first_order_relative_error": abs(slope - first_prediction)
                    / max(abs(slope), abs(first_prediction), 1e-12),
                }
            )
            change = sides[1]["outer_loss"] - baseline["outer_loss"]
            outer_steps.append(
                {
                    "gradient": name,
                    "requested_step_global_l2": epsilon,
                    "baseline_outer_loss": baseline["outer_loss"],
                    "outer_loss_after_step_and_fresh_attack": sides[1]["outer_loss"],
                    "objective_change": change,
                    "reduces_objective": change < 0,
                    "first_order_predicted_change": -epsilon * first_prediction,
                    "exact_predicted_change": -epsilon * exact_prediction,
                    "parameter_shift": sides[1]["parameter_shift"],
                    "reuses_minus_side_finite_difference_evaluation": True,
                }
            )
    return {
        "optimizer": asdict(config),
        "steps": len(batches),
        "exact_outer_loss": meta_values["exact"]["outer_loss"],
        "first_order_outer_loss": meta_values["first-order"]["outer_loss"],
        "exact_first_order_outer_loss_abs_difference": abs(
            meta_values["exact"]["outer_loss"]
            - meta_values["first-order"]["outer_loss"]
        ),
        "meta_inner_losses": {
            name: values["inner_losses"] for name, values in meta_values.items()
        },
        "baseline_replay": baseline,
        "exact_baseline_replay_abs_difference": abs(
            meta_values["exact"]["outer_loss"] - baseline["outer_loss"]
        ),
        **comparison,
        "finite_differences": differences,
        "negative_gradient_step_profiles": outer_steps,
        "interpretation": [
            "The exact gradient differentiates every functional optimizer operation, including global clipping.",
            "The first-order gradient uses an identity Jacobian through the complete optimizer update.",
            "Finite differences and step profiles rerun the attack from each perturbed initialization with fresh optimizer state.",
            "The outer loss is evaluated after the attack; negative-gradient steps therefore test the meta-objective directly.",
            "FP32 perturbations can differ from the requested radius; realized L2, projection, and changed fraction are reported.",
            "There is no automatic pass threshold for actual-model finite differences or a claim that the first-order approximation is exact.",
        ],
    }


def logits_and_labels(forward, batch):
    import torch

    output = forward(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )
    logits = output.logits[:, :-1]
    if logits.dtype not in {torch.float32, torch.float64}:
        raise RuntimeError(
            f"Expected native full-precision logits, received {logits.dtype}"
        )
    return logits, batch["labels"][:, 1:].to(logits.device)


def cross_entropy(forward, batch):
    import torch.nn.functional as F

    logits, labels = logits_and_labels(forward, batch)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def negative_entropy(forward, batch):
    import torch

    logits, labels = logits_and_labels(forward, batch)
    log_probs = torch.log_softmax(logits, -1)
    return (log_probs.exp() * log_probs).sum(-1)[labels != -100].mean()


def whole_batch_metrics(forward, batch, objective):
    import torch
    import torch.nn.functional as F

    logits, labels = logits_and_labels(forward, batch)
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    log_probs = torch.log_softmax(logits, -1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)[labels != -100].mean()
    return {
        "outer_loss": -ce if objective == "ce" else -entropy,
        "outer_ce": ce,
        "outer_entropy": entropy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--steps", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--outer-objective", choices=("ce", "entropy"), default="ce")
    parser.add_argument(
        "--directions", choices=("both", "exact", "first-order"), default="both"
    )
    parser.add_argument(
        "--fd-epsilons", type=float, nargs="+", default=(0.03, 0.01, 0.003)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if any(not math.isfinite(value) or value <= 0 for value in args.fd_epsilons):
        parser.error("--fd-epsilons must be finite and positive")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    clean, gradients = load_helpers()
    run, output = args.run.resolve(), args.output.resolve()
    study = clean.load_study(run)
    checkpoint = Path(study["anchor"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    status = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    # Reserve the result atomically so concurrent jobs cannot share an output.
    with output.open("x") as handle:
        handle.write(json.dumps(status) + "\n")
    started = time.monotonic()
    progress_path = output.with_suffix(".progress.jsonl")

    def progress(event):
        event = {"elapsed_seconds": time.monotonic() - started, **event}
        with progress_path.open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)

    try:
        import torch

        data = clean.prepare_data(run, study)
        inner_rows, outer_rows = data["forget_train"], data["forget_heldout"][:8]
        if len(inner_rows) != 8 or len(outer_rows) != 8:
            raise ValueError("This diagnostic requires forget8 and heldout8")
        if {clean.digest(row["input_ids"]) for row in inner_rows} & {
            clean.digest(row["input_ids"]) for row in outer_rows
        }:
            raise ValueError(
                "Attack and outer documents contain identical token sequences"
            )
        config = gradients.OptimizerConfig(
            name=args.optimizer, lr=args.lr, weight_decay=0.0, max_grad_norm=1.0
        )
        settings = {
            "checkpoint": str(checkpoint),
            "optimizer": asdict(config),
            "steps": args.steps,
            "inner_loss": "mean causal next-token CE",
            "inner_scope": "one whole batch of all8 forget_train documents at every step",
            "outer_loss": (
                "negative mean CE"
                if args.outer_objective == "ce"
                else "negative mean entropy"
            ),
            "outer_scope": "first8 forget_heldout documents; terminal state only",
            "fresh_optimizer_state_for_every_unroll": True,
            "first_order_convention": "identity Jacobian through the complete optimizer update",
            "forward_dtype": "float32",
            "attention": "eager",
            "amp": False,
            "directions": (
                ["exact", "first-order"]
                if args.directions == "both"
                else [args.directions]
            ),
            "fd_epsilons_global_l2": list(args.fd_epsilons),
            "data_content_sha256": data["content_sha256"],
            "inner_document_ids": [
                {key: row[key] for key in ("source_row", "source_text_sha256")}
                for row in inner_rows
            ],
            "outer_document_ids": [
                {key: row[key] for key in ("source_row", "source_text_sha256")}
                for row in outer_rows
            ],
            "inner_valid_tokens_per_update": sum(
                sum(label != -100 for label in row["labels"][1:]) for row in inner_rows
            ),
            "outer_valid_tokens": sum(
                sum(label != -100 for label in row["labels"][1:]) for row in outer_rows
            ),
            "memory_policy": "sequential exact/FO meta-gradients; both completed maps on CPU; chunked shifts/dots; no saved gradient tensors",
            "scope": "optional short-horizon derivative diagnostic; not a main-study robustness result",
        }
        dump(output.with_suffix(".config.json"), settings)
        dump(
            output.with_suffix(".provenance.json"),
            provenance(run, checkpoint, output, clean, gradients),
        )
        progress({"phase": "load-model", "state": "started"})
        model, _ = clean.runtime(study, checkpoint)
        assert not model.training
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        params = dict(model.named_parameters())
        inner = clean.batch(inner_rows, model.device)
        outer = clean.batch(outer_rows, model.device)
        progress(
            {
                "phase": "load-model",
                "state": "complete",
                "parameters": sum(p.numel() for p in params.values()),
            }
        )
        objective = (
            (lambda forward, batch: -cross_entropy(forward, batch))
            if args.outer_objective == "ce"
            else negative_entropy
        )
        result = diagnostic(
            gradients,
            model,
            params,
            (inner,) * args.steps,
            cross_entropy,
            outer,
            objective,
            config,
            directions=tuple(settings["directions"]),
            epsilons=tuple(args.fd_epsilons),
            outer_metrics_fn=lambda forward, batch: whole_batch_metrics(
                forward, batch, args.outer_objective
            ),
            progress=progress,
        )
        assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
        result["peak_accelerator_memory_bytes"] = {
            str(device): torch.cuda.max_memory_allocated(device)
            for device in range(torch.cuda.device_count())
        }
        status.update(
            state="complete",
            settings=settings,
            result=result,
            seconds=time.monotonic() - started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        dump(output, status)
        print(json.dumps(status, allow_nan=False), flush=True)
    except Exception as error:
        status.update(
            state="failed",
            error=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )
        dump(output, status)
        raise


if __name__ == "__main__":
    main()
