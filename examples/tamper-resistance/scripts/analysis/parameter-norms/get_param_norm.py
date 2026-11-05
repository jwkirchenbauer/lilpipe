"""Measure model parameter norms and optional movement from a reference model."""

import argparse
import json
import math
import os


def layer_group(parameter_name, num_layers):
    """Return a display name and position for one Zephyr/Mistral parameter."""
    parts = parameter_name.split(".")
    if parameter_name.startswith("model.embed_tokens."):
        return "Embedding", 0
    if len(parts) > 2 and parts[:2] == ["model", "layers"]:
        layer = int(parts[2])
        return str(layer), layer + 1
    if parameter_name.startswith("model.norm."):
        return "Final norm", num_layers + 1
    if parameter_name.startswith("lm_head."):
        return "LM head", num_layers + 2
    return "Other", num_layers + 3


def parameter_norms(model, reference=None):
    """Measure all named parameters once, matching a reference by name and shape."""
    import torch

    parameters = dict(model.named_parameters())
    reference_parameters = (
        dict(reference.named_parameters()) if reference is not None else {}
    )
    if reference is not None and parameters.keys() != reference_parameters.keys():
        raise ValueError("Model and reference parameter names differ")
    num_layers = len(model.model.layers)
    grouped = {}
    for name, parameter in parameters.items():
        group, position = layer_group(name, num_layers)
        values = grouped.setdefault(
            group,
            {
                "name": group,
                "position": position,
                "num_parameters": 0,
                "sum_sq": 0.0,
                "reference_sum_sq": 0.0,
                "delta_sum_sq": 0.0,
            },
        )
        values["num_parameters"] += parameter.numel()
        # Convert before subtraction, then accumulate squares in float64. CPU
        # reductions also keep the extra memory bounded to one parameter tensor.
        current = parameter.detach().to(device="cpu", dtype=torch.float64)
        values["sum_sq"] += current.square().sum().item()
        if reference is not None:
            original = reference_parameters[name]
            if parameter.shape != original.shape:
                raise ValueError(f"Parameter shape differs: {name}")
            original = original.detach().to(device="cpu", dtype=torch.float64)
            values["reference_sum_sq"] += original.square().sum().item()
            values["delta_sum_sq"] += (current - original).square().sum().item()

    layers = []
    for values in sorted(grouped.values(), key=lambda values: values["position"]):
        sum_sq = values.pop("sum_sq")
        reference_sum_sq = values.pop("reference_sum_sq")
        delta_sum_sq = values.pop("delta_sum_sq")
        if reference is not None:
            values["reference_l2_norm"] = math.sqrt(reference_sum_sq)
            values["delta_l2_norm"] = math.sqrt(delta_sum_sq)
        values["l2_norm"] = math.sqrt(sum_sq)
        values["rms"] = math.sqrt(sum_sq / values["num_parameters"])
        layers.append(values)

    sq_norm = sum(layer["l2_norm"] ** 2 for layer in layers)
    num_params = sum(layer["num_parameters"] for layer in layers)
    result = {
        "num_parameters": num_params,
        "l2_norm": math.sqrt(sq_norm),
        "rms": math.sqrt(sq_norm / num_params),
        "layers": layers,
    }
    if reference is not None:
        reference_norm = math.sqrt(
            sum(layer["reference_l2_norm"] ** 2 for layer in layers)
        )
        delta_norm = math.sqrt(sum(layer["delta_l2_norm"] ** 2 for layer in layers))
        result.update(
            reference_l2_norm=reference_norm,
            delta_l2_norm=delta_norm,
            relative_delta_l2_norm=(
                delta_norm / reference_norm if reference_norm else None
            ),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument(
        "--reference", help="Starting model name or path for delta norms"
    )
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--output", default="artifacts/analysis/z7b-layerwise-norms.json"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device
    )
    reference = None
    if args.reference:
        reference = AutoModelForCausalLM.from_pretrained(
            args.reference,
            torch_dtype=getattr(torch, args.dtype),
            device_map=args.device,
        )
    result = parameter_norms(model, reference)
    result.update(
        model=args.model, parameter_dtype=args.dtype, accumulation_dtype="float64"
    )
    if args.reference:
        result["reference"] = args.reference
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(f"model: {args.model}")
    print(f"params: {result['num_parameters']:,}")
    print(f"param norm: {result['l2_norm']:.4f}")
    print(f"param RMS: {result['rms']:.6g}")
    if args.reference:
        print(f"reference norm: {result['reference_l2_norm']:.6g}")
        print(f"delta norm: {result['delta_l2_norm']:.6g}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
