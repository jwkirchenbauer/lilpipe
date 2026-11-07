"""Save models perturbed along one Gaussian direction at specified global L2 radii."""

import argparse
import copy
import json
import math
import os


def noise_statistics(model, seed):
    """Measure the model and a reproducible Gaussian vector over its parameters."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    model_sum_sq = noise_sum_sq = 0.0
    count = 0
    for parameter in model.parameters():
        original = parameter.detach().to(device="cpu", dtype=torch.float64)
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
        model_sum_sq += original.square().sum().item()
        noise_sum_sq += noise.double().square().sum().item()
        count += parameter.numel()
    return {
        "num_parameters": count,
        "model_l2_norm": math.sqrt(model_sum_sq),
        "gaussian_l2_norm": math.sqrt(noise_sum_sq),
    }


def perturb(model, fraction, seed, statistics):
    """Add a normalized direction in place, measuring the delta after dtype rounding."""
    import torch

    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("Noise fractions must be finite and positive")
    target = fraction * statistics["model_l2_norm"]
    multiplier = target / statistics["gaussian_l2_norm"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    delta_sum_sq = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            original = parameter.detach().to(device="cpu", dtype=torch.float64)
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            )
            changed = (original + multiplier * noise.double()).to(parameter.dtype)
            delta_sum_sq += (changed.double() - original).square().sum().item()
            parameter.copy_(changed)
    realized = math.sqrt(delta_sum_sq)
    return {
        "fraction": fraction,
        "target_delta_l2_norm": target,
        "realized_delta_l2_norm": realized,
        "realized_fraction": realized / statistics["model_l2_norm"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output-prefix", required=True, help="Prefix of new model paths"
    )
    parser.add_argument(
        "--output", required=True, help="Perturbation metadata JSON path"
    )
    parser.add_argument(
        "--percentages",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 0.75],
        help="Noise L2 norm as a percentage of model L2 norm (0.1 means 0.1%)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if any(not math.isfinite(f) or f <= 0 for f in args.percentages):
        parser.error("percentages must be finite and positive")
    fractions = [percentage / 100 for percentage in args.percentages]
    paths = [f"{args.output_prefix}-l2r{f:g}-seed{args.seed}" for f in fractions]
    if len(set(paths)) != len(paths):
        parser.error("percentages must produce distinct output model paths")
    for path in [*paths, args.output]:
        if os.path.lexists(path):
            parser.error(f"output already exists: {path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    statistics = noise_statistics(model, args.seed)
    if statistics["model_l2_norm"] == 0:
        raise ValueError("Relative perturbation requires a nonzero model norm")
    report = {
        "source_model": args.model,
        "seed": args.seed,
        "distribution": "standard Gaussian over all named parameters; buffers excluded",
        "shared_direction_across_fractions": True,
        "percentages": args.percentages,
        "parameter_dtype": args.dtype,
        "accumulation_dtype": "float64",
        "torch_version": torch.__version__,
        **statistics,
        "models": [],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for fraction, path in zip(fractions, paths):
        # Every model starts from the original weights, never the previous radius.
        changed = copy.deepcopy(model)
        values = perturb(changed, fraction, args.seed, statistics)
        changed.save_pretrained(path)
        tokenizer.save_pretrained(path)
        report["models"].append({"model": path, "percentage": 100 * fraction, **values})
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(json.dumps(report["models"][-1]), flush=True)
        del changed
    report["complete"] = True
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
