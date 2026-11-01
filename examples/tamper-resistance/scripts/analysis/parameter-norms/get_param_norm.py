"""Measure global and layerwise norms of a causal language model."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument(
        "--output", default="artifacts/analysis/z7b-layerwise-norms.json"
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    num_layers = len(model.model.layers)
    grouped = {}
    for name, parameter in model.named_parameters():
        group, position = layer_group(name, num_layers)
        values = grouped.setdefault(
            group,
            {"name": group, "position": position, "num_parameters": 0, "sum_sq": 0},
        )
        values["num_parameters"] += parameter.numel()
        values["sum_sq"] += parameter.float().pow(2).sum().item()

    layers = []
    for values in sorted(grouped.values(), key=lambda values: values["position"]):
        sum_sq = values.pop("sum_sq")
        values["l2_norm"] = math.sqrt(sum_sq)
        values["rms"] = math.sqrt(sum_sq / values["num_parameters"])
        layers.append(values)

    sq_norm = sum(layer["l2_norm"] ** 2 for layer in layers)
    num_params = sum(layer["num_parameters"] for layer in layers)
    result = {
        "model": args.model,
        "num_parameters": num_params,
        "l2_norm": math.sqrt(sq_norm),
        "rms": math.sqrt(sq_norm / num_params),
        "layers": layers,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(f"model: {args.model}")
    print(f"params: {num_params:,}")
    print(f"param norm: {result['l2_norm']:.4f}")
    print(f"param RMS: {result['rms']:.6g}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
