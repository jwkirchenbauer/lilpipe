"""Evaluate MMLU after independent Gaussian weight perturbations of one component."""

import argparse
from contextlib import contextmanager
import csv
import json
import math
import random
from pathlib import Path


def parameter_groups(model):
    """Partition Zephyr/Mistral parameters, rejecting shared or unknown weights."""
    groups = {
        "embedding": list(model.model.embed_tokens.parameters()),
        **{
            f"layer-{i}": list(layer.parameters())
            for i, layer in enumerate(model.model.layers)
        },
        "final-norm": list(model.model.norm.parameters()),
        "lm-head": list(model.lm_head.parameters()),
    }
    parameters = [p for group in groups.values() for p in group]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Shared weights prevent independent component perturbations")
    if {id(p) for p in parameters} != {id(p) for p in model.parameters()}:
        raise ValueError("Component groups do not cover the model parameters")
    return groups


@contextmanager
def perturb(parameters, scale, seed, normalization):
    """Restore exact weights even if evaluation fails; reuse directions across scales."""
    import torch

    originals = [p.detach().to("cpu", copy=True) for p in parameters]
    count = sum(p.numel() for p in parameters)
    sum_sq = sum(p.double().square().sum().item() for p in originals)
    weight_rms = math.sqrt(sum_sq / count)
    sigma = scale * weight_rms if normalization == "relative" else scale
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta_sq = 0.0
    try:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                noise = torch.randn(
                    original.shape, generator=generator, dtype=torch.float32
                )
                changed = (original.float() + sigma * noise).to(parameter.dtype)
                delta_sq += (changed.double() - original.double()).square().sum().item()
                parameter.copy_(changed)
        yield {
            "num_parameters": count,
            "weight_rms": weight_rms,
            "noise_std": sigma,
            "realized_noise_rms": math.sqrt(delta_sq / count),
            "realized_noise_l2": math.sqrt(delta_sq),
        }
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals):
                parameter.copy_(original)


def random_subset(subject_sizes, count, seed):
    """Choose reproducible random test indices independently for every subject."""
    rng = random.Random(seed)
    return {
        name: sorted(rng.sample(range(size), min(count, size)))
        for name, size in sorted(subject_sizes.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--groups", nargs="+", help="Default: all components")
    parser.add_argument(
        "--scales", nargs="+", type=float, default=[0.01, 0.03, 0.1, 0.3, 1.0]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-subject", type=int)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument(
        "--normalization", choices=["relative", "absolute"], default="relative"
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, help="Smoke test: examples per MMLU subject"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    args = parser.parse_args()
    if any(not math.isfinite(s) or s <= 0 for s in args.scales):
        parser.error(
            "scales must be finite and positive; the baseline is evaluated separately"
        )
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    if args.samples_per_subject is not None and args.samples_per_subject <= 0:
        parser.error("samples-per-subject must be positive")
    if args.samples_per_subject is not None and args.limit is not None:
        parser.error("use samples-per-subject or limit, not both")
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import get_task_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    ).eval()
    groups = parameter_groups(model)
    selected = args.groups or list(groups)
    if unknown := set(selected) - groups.keys():
        parser.error(f"unknown groups: {sorted(unknown)}")
    lm = HFLM(
        pretrained=model,
        tokenizer=AutoTokenizer.from_pretrained(args.model),
        batch_size=args.batch_size,
    )
    config = {**vars(args), "output": str(args.output), "groups": selected}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    samples = None
    if args.samples_per_subject is not None:
        task_dict = get_task_dict(["mmlu"])
        sizes = {
            task.task_name: len(task.task.test_docs())
            for task in get_task_list(task_dict)
        }
        samples = random_subset(sizes, args.samples_per_subject, args.subset_seed)
        (args.output / "subset.json").write_text(json.dumps(samples, indent=2) + "\n")
        del task_dict

    def evaluate(name):
        result = simple_evaluate(
            model=lm,
            tasks=["mmlu"],
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            samples=samples,
            log_samples=False,
            bootstrap_iters=1000,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
        (args.output / f"{name}.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        return result["results"]["mmlu"]["acc,none"]

    baseline = evaluate("baseline")
    with (args.output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "scale",
                "seed",
                "accuracy",
                "baseline_accuracy",
                "drop_pp",
                "num_parameters",
                "weight_rms",
                "noise_std",
                "realized_noise_rms",
                "realized_noise_l2",
            ],
        )
        writer.writeheader()
        handle.flush()
        for group in selected:
            for scale in args.scales:
                with perturb(
                    groups[group], scale, args.seed, args.normalization
                ) as stats:
                    accuracy = evaluate(f"{group}-seed{args.seed}-scale{scale:g}")
                row = {
                    "group": group,
                    "scale": scale,
                    "seed": args.seed,
                    "accuracy": accuracy,
                    "baseline_accuracy": baseline,
                    "drop_pp": 100 * (baseline - accuracy),
                    **stats,
                }
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
