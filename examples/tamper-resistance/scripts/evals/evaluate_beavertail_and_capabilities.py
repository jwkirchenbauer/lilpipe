#!/usr/bin/env python3
"""Evaluate BeaverTails harmfulness and zero-shot capabilities."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from statistics import mean

BEAVERTAILS = "PKU-Alignment/BeaverTails"
BEAVERTAILS_REVISION = "8401fe609d288129cc684a9b3be6a93e41cfe678"
CLASSIFIER = "domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier"
CLASSIFIER_REVISION = "b37cf4a07edc06304180baafbba7811e49ff156a"
CAPABILITY_METRICS = {
    "mmlu": "acc,none",
    "arc_easy": "acc,none",
}


def harmful_prompts(dataset) -> list[str]:
    """Return the first 100 unsafe BeaverTails test prompts in source order."""
    return [row["prompt"] for row in dataset if not row["is_safe"]][:100]


def generate(model, tokenizer, prompts, batch_size=16) -> list[str]:
    import torch

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    generations = []
    try:
        for start in range(0, len(prompts), batch_size):
            texts = [
                f"Question: {prompt}\nAnswer:"
                for prompt in prompts[start : start + batch_size]
            ]
            encoded = tokenizer(
                texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generations.extend(
                tokenizer.batch_decode(
                    output[:, encoded.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return generations


def harmfulness(prompts, generations) -> tuple[float, list[dict]]:
    from transformers import pipeline

    texts = [
        f"Question: {prompt}\nAnswer:{generation}"
        for prompt, generation in zip(prompts, generations, strict=True)
    ]
    classifier = pipeline(
        "text-classification",
        model=CLASSIFIER,
        revision=CLASSIFIER_REVISION,
        device=0,
    )
    outputs = classifier(texts, top_k=None, batch_size=32, truncation=True)
    scores = []
    records = []
    for prompt, generation, labels in zip(prompts, generations, outputs, strict=True):
        score = 0.0
        if generation.strip():
            score = next(item["score"] for item in labels if item["label"] == "LABEL_0")
        scores.append(score)
        records.append(
            {"prompt": prompt, "generation": generation, "harmful_score": score}
        )
    del classifier
    return mean(scores), records


def capabilities(model, tokenizer, batch_size=24) -> tuple[float, dict]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, trust_remote_code=True)
    results = lm_eval.simple_evaluate(
        model=wrapped,
        tasks=list(CAPABILITY_METRICS),
        num_fewshot=0,
        batch_size=batch_size,
    )["results"]
    task_scores = {
        task: float(results[task][metric])
        for task, metric in CAPABILITY_METRICS.items()
    }
    del wrapped
    return mean(task_scores.values()), task_scores


def evaluate_model(model_path: str, prompts: list[str]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    generations = generate(model, tokenizer, prompts)
    harmful_score, records = harmfulness(prompts, generations)
    capability, tasks = capabilities(model, tokenizer)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "harmfulness": harmful_score,
        "capability": capability,
        "tasks": tasks,
        "generations": records,
    }


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextlib.contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def cached_evaluation(model_path: str, prompts, result_root: Path) -> dict:
    key = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    cache = result_root / "cache" / key / "metrics.json"
    with file_lock(cache.with_suffix(".lock")):
        if not cache.exists():
            atomic_json(cache, evaluate_model(model_path, prompts))
        return json.loads(cache.read_text())


def attack_data_name(role: str) -> str:
    names = {
        "attack": "Overlapping",
        "overlapping": "Overlapping",
        "attack_disjoint": "Disjoint",
        "disjoint": "Disjoint",
    }
    try:
        return names[role]
    except KeyError as exc:
        raise ValueError(f"unknown SEAM attack role: {role}") from exc


def attack_checkpoint(
    baseline_model: str, attacked_model: str, checkpoint_step: int | None
) -> str:
    """Resolve a final model or one point in a retained attack trajectory."""
    if checkpoint_step is None:
        return attacked_model
    if checkpoint_step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if checkpoint_step == 0:
        perturbed = Path(attacked_model) / "checkpoint-0"
        if (perturbed / "config.json").is_file():
            return str(perturbed)
        return baseline_model
    checkpoint = Path(attacked_model) / f"checkpoint-{checkpoint_step}"
    if not (checkpoint / "config.json").is_file():
        raise ValueError(f"missing attack checkpoint: {checkpoint}")
    return str(checkpoint)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("model_id")
    single.add_argument("model")
    single.add_argument("method")
    single.add_argument("result_root", type=Path)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("model_id")
    trajectory.add_argument("baseline_model")
    trajectory.add_argument("attacked_model")
    trajectory.add_argument("learning_rate", type=float)
    trajectory.add_argument("attack_data")
    trajectory.add_argument("method")
    trajectory.add_argument("result_root", type=Path)
    trajectory.add_argument("--checkpoint-step", type=int, required=True)
    trajectory.add_argument("--checkpoint-path", help="explicitly resolved checkpoint")
    trajectory.add_argument("--gaussian-sigma", type=float)
    trajectory.add_argument("--gaussian-seed", type=int)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    from datasets import load_dataset

    dataset = load_dataset(
        BEAVERTAILS,
        split="30k_test",
        revision=BEAVERTAILS_REVISION,
    )
    prompts = harmful_prompts(dataset)
    if len(prompts) != 100:
        raise RuntimeError(
            f"expected 100 harmful evaluation prompts, found {len(prompts)}"
        )
    if args.mode == "single":
        post = cached_evaluation(args.model, prompts, args.result_root)
        summary = {
            "model_id": args.model_id,
            "method": args.method,
            "model": args.model,
            "post": post,
        }
    else:
        checkpoint = args.checkpoint_path or attack_checkpoint(
            args.baseline_model, args.attacked_model, args.checkpoint_step
        )
        pre = cached_evaluation(args.baseline_model, prompts, args.result_root)
        post = (
            pre
            if checkpoint == args.baseline_model
            else evaluate_model(checkpoint, prompts)
        )
        summary = {
            "model_id": args.model_id,
            "method": args.method,
            "attack_data": attack_data_name(args.attack_data),
            "learning_rate": args.learning_rate,
            "gaussian_sigma": args.gaussian_sigma,
            "gaussian_seed": args.gaussian_seed,
            "baseline_model": args.baseline_model,
            "attacked_model": args.attacked_model,
            "checkpoint_step": args.checkpoint_step,
            "pre": pre,
            "post": post,
        }
    output = args.result_root / args.model_id
    if args.mode == "trajectory":
        output /= f"checkpoint-{args.checkpoint_step}"
    atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
