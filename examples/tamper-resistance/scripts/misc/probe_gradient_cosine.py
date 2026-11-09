"""Measure forget-gradient geometry on sampled batches and audit FD directions."""

import argparse
import gc
import json
import os
import random
import secrets
import statistics

import torch
import yaml
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data.beavertails_hh import BeaverTailsHHStrategy
from trainers.tr_gip import (
    GhostInnerProductsMixin,
    causal_lm_losses,
    finite_difference_cosine,
    gradient_dot,
    mean_gradient_cosine,
)


class ProbeTrainer(GhostInnerProductsMixin, Trainer):
    pass


def sample_forget_rows(dataset, count, seed=None):
    """Sample distinct forget rows; record a fresh seed when none is supplied."""
    if "is_forget" not in dataset.column_names:
        raise ValueError("Prepared data must include is_forget")
    seed = secrets.randbits(32) if seed is None else seed
    indices = [i for i, flag in enumerate(dataset["is_forget"]) if flag]
    chosen = random.Random(seed).sample(indices, count)
    return [dataset[i] for i in chosen], chosen, seed


def explicit_gram(model, batch):
    """Materialize individual parameter gradients for optional validation only."""
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    losses = causal_lm_losses(model(**batch).logits, batch["labels"])
    gradients = []
    for i, loss in enumerate(losses):
        values = torch.autograd.grad(
            loss, parameters, retain_graph=i + 1 < len(losses), allow_unused=True
        )
        gradients.append(
            tuple(
                torch.zeros_like(p) if g is None else g.detach()
                for p, g in zip(parameters, values, strict=True)
            )
        )
    return torch.stack(
        [
            torch.stack(
                [
                    gradient_dot((g.float() for g in a), (g.float() for g in b))
                    for b in gradients
                ]
            )
            for a in gradients
        ]
    ).cpu()


def batch_rows(rows, tokenizer, device, sequence_len, *, prepared=False):
    if prepared:
        encoded = rows
    else:
        strategy = BeaverTailsHHStrategy(tokenizer, sequence_len, False, len(rows))
        encoded = [strategy.tokenize_row(row) for row in rows]
    batch = {
        "input_ids": torch.full(
            (len(rows), sequence_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        ),
        "labels": torch.full(
            (len(rows), sequence_len), -100, dtype=torch.long, device=device
        ),
        "attention_mask": torch.zeros(
            (len(rows), sequence_len), dtype=torch.long, device=device
        ),
    }
    for i, row in enumerate(encoded):
        length = min(len(row["input_ids"]), sequence_len)
        for key in batch:
            batch[key][i, :length] = torch.tensor(row[key][:length], device=device)
    return batch


def directional_audit(model, batch, tau, rho):
    half = len(batch["labels"]) // 2
    a = {key: value[:half] for key, value in batch.items()}
    b = {key: value[half:] for key, value in batch.items()}

    def loss(inputs):
        return causal_lm_losses(model(**inputs).logits, inputs["labels"]).mean()

    result = finite_difference_cosine(
        model, lambda: loss(a), lambda: loss(b), tau=tau, rho=rho
    )
    length = gradient_dot(result.regularization, result.regularization).sqrt()
    direction = tuple(g / length for g in result.regularization)
    base_cosine = result.cosine.item()
    expected = length.item()
    del result
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    originals = tuple(p.detach().clone() for p in parameters)
    epsilon = min(rho, 1e-4)
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                for p, original, q in zip(
                    parameters, originals, direction, strict=True
                ):
                    p.copy_(original + sign * epsilon * q)
            ga = torch.autograd.grad(loss(a), parameters)
            gb = torch.autograd.grad(loss(b), parameters)
            cosine = gradient_dot(ga, gb) / (
                (gradient_dot(ga, ga) + tau**2).sqrt()
                * (gradient_dot(gb, gb) + tau**2).sqrt()
            )
            values.append(cosine.item())
            del ga, gb, cosine
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals, strict=True):
                p.copy_(original)
    slope = (values[0] - values[1]) / (2 * epsilon)
    return {
        "base_cosine": base_cosine,
        "cosine_plus": values[0],
        "cosine_minus": values[1],
        "r_norm": expected,
        "central_directional_derivative": slope,
        "relative_error": abs(slope - expected) / max(expected, 1e-12),
        "rho": rho,
        "audit_epsilon": epsilon,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", action="append", required=True, help="LABEL=MODEL_PATH"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--offset", type=int, default=8000)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=256)
    parser.add_argument("--audit-fd", action="store_true")
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--rho", type=float, default=1e-3)
    parser.add_argument("--rho-sweep", nargs="+", type=float)
    parser.add_argument(
        "--prepared-dataset", help="Tokenized Arrow dataset including is_forget"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional repeatable probe seed; default samples fresh rows per model",
    )
    parser.add_argument(
        "--device-map-config",
        help="Training YAML supplying device_map and attention implementation",
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--check-explicit",
        action="store_true",
        help="Validate the first batch against materialized parameter gradients",
    )
    args = parser.parse_args()
    if args.dtype != "float32" and (args.audit_fd or args.rho_sweep):
        raise ValueError("Finite-difference audits require float32")
    model_options = {}
    if args.device_map_config:
        with open(args.device_map_config) as stream:
            config = yaml.safe_load(stream)
        model_options["device_map"] = config["device_map"]
        model_options["attn_implementation"] = config.get(
            "attn_implementation", "eager"
        )
    else:
        model_options["attn_implementation"] = "sdpa"
    prepared = (
        load_from_disk(args.prepared_dataset, keep_in_memory=True)
        if args.prepared_dataset
        else None
    )
    rows = None
    if prepared is None:
        raw = load_dataset(
            "anonymous4486/repnoise_beavertail",
            split="train",
            revision="11ea63756195527164a9aa2850f7544bd251eab1",
        )
        rows = [row for row in raw if not row["is_safe"]][
            args.offset : args.offset + args.rows
        ]
    if (
        args.rows <= 0
        or args.batch_size < 2
        or args.rows % args.batch_size
        or args.batch_size % 2
    ):
        raise ValueError("Require complete, even batches")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {
        "dataset_offset": args.offset if prepared is None else None,
        "prepared_dataset": args.prepared_dataset,
        "dtype": args.dtype,
        "device_map_config": args.device_map_config,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "sequence_len": args.sequence_len,
        "models": [],
    }
    for spec in args.model:
        name, path = spec.split("=", 1)
        indices = None
        seed = None
        if prepared is not None:
            rows, indices, seed = sample_forget_rows(prepared, args.rows, args.seed)
        if len(rows) != args.rows:
            raise ValueError("Not enough probe rows")
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=getattr(torch, args.dtype), **model_options
        )
        if "device_map" not in model_options:
            model.cuda()
        model.train()
        model.config.use_cache = False
        # Qwen3's dropout is zero; keep actual train-mode gradients as in the attack.
        if any(m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)):
            raise ValueError("Gradient probe currently requires zero dropout")
        observer = ProbeTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.dirname(args.output),
                report_to=[],
                bf16=False,
                fp16=False,
            ),
        )
        record = {
            "name": name,
            "path": path,
            "sampling_seed": seed,
            "source_indices": indices,
            "batches": [],
        }
        versions = tuple(p._version for p in model.parameters())
        try:
            for start in range(0, len(rows), args.batch_size):
                batch = batch_rows(
                    rows[start : start + args.batch_size],
                    tokenizer,
                    model.device,
                    args.sequence_len,
                    prepared=prepared is not None,
                )
                model.zero_grad(set_to_none=True)
                outputs = model(**batch)
                losses = causal_lm_losses(outputs.logits, batch["labels"])
                losses.sum().backward()
                gram = observer.gram.detach().cpu()
                half = len(rows[start : start + args.batch_size]) // 2
                aa = gram[:half, :half].mean()
                bb = gram[half:, half:].mean()
                ab = gram[:half, half:].mean()
                record["batches"].append(
                    {
                        "offset": args.offset + start if prepared is None else start,
                        "loss": losses.mean().item(),
                        "mean_pair_cosine": mean_gradient_cosine(gram).item(),
                        "mean_pair_squared_cosine": mean_gradient_cosine(
                            gram, squared=True
                        ).item(),
                        "gradient_norm_mean": gram.diagonal()
                        .clamp_min(0)
                        .sqrt()
                        .mean()
                        .item(),
                        "gradient_norm_min": gram.diagonal()
                        .clamp_min(0)
                        .sqrt()
                        .min()
                        .item(),
                        "group_cosine": (
                            ab / ((aa + args.tau**2) * (bb + args.tau**2)).sqrt()
                        ).item(),
                        "group_norm_a": aa.sqrt().item(),
                        "group_norm_b": bb.sqrt().item(),
                    }
                )
                print(name, record["batches"][-1], flush=True)
                observer.finish_microbatch()
                del outputs, losses
                model.zero_grad(set_to_none=True)
                if args.check_explicit and start == 0:
                    previous = observer.track_ghost
                    observer.track_ghost = False
                    try:
                        expected = explicit_gram(model, batch)
                    finally:
                        observer.track_ghost = previous
                    error = (
                        (gram - expected).norm() / expected.norm().clamp_min(1e-12)
                    ).item()
                    record["explicit_gram_relative_error"] = error
                    tolerance = 0.02 if args.dtype == "bfloat16" else 1e-4
                    if error > tolerance:
                        raise AssertionError(
                            f"Ghost/reference Gram relative error {error} > {tolerance}"
                        )
                del gram
        finally:
            observer.close()
            model.zero_grad(set_to_none=True)
        assert versions == tuple(
            p._version for p in model.parameters()
        ), "Probe changed model parameters"
        record["parameter_versions_unchanged"] = True
        record["summary"] = {}
        for key in (
            "mean_pair_cosine",
            "mean_pair_squared_cosine",
            "gradient_norm_mean",
            "gradient_norm_min",
        ):
            values = [r[key] for r in record["batches"]]
            record["summary"][key] = statistics.mean(values)
            record["summary"][key + "_sem"] = (
                statistics.stdev(values) / len(values) ** 0.5
                if len(values) > 1
                else None
            )
        if args.audit_fd:
            batch = batch_rows(
                rows[: args.batch_size],
                tokenizer,
                model.device,
                args.sequence_len,
                prepared=prepared is not None,
            )
            record["finite_difference_audit"] = directional_audit(
                model, batch, args.tau, args.rho
            )
            print(name, record["finite_difference_audit"], flush=True)
        if args.rho_sweep:
            batch = batch_rows(
                rows[: args.batch_size],
                tokenizer,
                model.device,
                args.sequence_len,
                prepared=prepared is not None,
            )
            record["finite_difference_sweep"] = []
            for rho in args.rho_sweep:
                audit = directional_audit(model, batch, args.tau, rho)
                record["finite_difference_sweep"].append(audit)
                print(name, "rho sweep", audit, flush=True)
        results["models"].append(record)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        del observer, model, tokenizer, batch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
