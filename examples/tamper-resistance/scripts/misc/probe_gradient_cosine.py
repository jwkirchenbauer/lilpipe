"""Compare fixed held-out harmful-gradient geometry and audit FD directions."""

import argparse
import gc
import json
import os

import torch
from datasets import load_dataset
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


def batch_rows(rows, tokenizer, device, sequence_len):
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
        length = len(row["input_ids"])
        for key in batch:
            batch[key][i, :length] = torch.tensor(row[key], device=device)
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
    args = parser.parse_args()
    raw = load_dataset(
        "anonymous4486/repnoise_beavertail",
        split="train",
        revision="11ea63756195527164a9aa2850f7544bd251eab1",
    )
    rows = [row for row in raw if not row["is_safe"]][
        args.offset : args.offset + args.rows
    ]
    if len(rows) != args.rows or args.rows % args.batch_size or args.batch_size % 2:
        raise ValueError("Require complete, even batches of held-out rows")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results = {
        "dataset_offset": args.offset,
        "rows": args.rows,
        "batch_size": args.batch_size,
        "sequence_len": args.sequence_len,
        "models": [],
    }
    for spec in args.model:
        name, path = spec.split("=", 1)
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = (
            AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float32, attn_implementation="sdpa"
            )
            .cuda()
            .train()
        )
        model.config.use_cache = False
        # Qwen3's dropout is zero; keep actual train-mode gradients as in the attack.
        if any(m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)):
            raise ValueError("Fixed probe currently requires zero dropout")
        observer = ProbeTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.dirname(args.output),
                report_to=[],
                bf16=False,
                fp16=False,
            ),
        )
        record = {"name": name, "path": path, "batches": []}
        try:
            for start in range(0, len(rows), args.batch_size):
                batch = batch_rows(
                    rows[start : start + args.batch_size],
                    tokenizer,
                    model.device,
                    args.sequence_len,
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
                        "offset": args.offset + start,
                        "loss": losses.mean().item(),
                        "mean_pair_cosine": mean_gradient_cosine(gram).item(),
                        "group_cosine": (
                            ab / ((aa + args.tau**2) * (bb + args.tau**2)).sqrt()
                        ).item(),
                        "group_norm_a": aa.sqrt().item(),
                        "group_norm_b": bb.sqrt().item(),
                    }
                )
                print(name, record["batches"][-1], flush=True)
                observer.finish_microbatch()
                del outputs, losses, gram
        finally:
            observer.close()
            model.zero_grad(set_to_none=True)
        if args.audit_fd:
            batch = batch_rows(
                rows[: args.batch_size], tokenizer, model.device, args.sequence_len
            )
            record["finite_difference_audit"] = directional_audit(
                model, batch, args.tau, args.rho
            )
            print(name, record["finite_difference_audit"], flush=True)
        if args.rho_sweep:
            batch = batch_rows(
                rows[: args.batch_size], tokenizer, model.device, args.sequence_len
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
