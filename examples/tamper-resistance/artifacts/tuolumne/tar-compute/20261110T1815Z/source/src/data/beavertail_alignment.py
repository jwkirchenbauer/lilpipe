"""Deterministic BeaverTails alignment, relearning, and evaluation splits."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

try:
    from axolotl.prompt_tokenizers import DatasetWrappingStrategy
except ModuleNotFoundError:  # Evaluation reuses split helpers without Axolotl.
    DatasetWrappingStrategy = object

SEED = 42
EVALUATION_PROMPTS = 1_000
UNSAFE_PROMPTS = 5_018
TRAINING_ROWS = 5_000
MANIFEST_PATH = Path(
    "artifacts/mila/data/beavertail-alignment-seed42/split-indices.json"
)
INSTRUCTION_PREFIX = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
)
INSTRUCTION_SUFFIX = "\n\n### Response:\n"
INPUT_INSTRUCTION_PREFIX = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the "
    "request.\n\n### Instruction:\n"
)


def format_prompt(prompt: str) -> str:
    """Apply the instruction template used by the Booster reference code."""
    return f"{INSTRUCTION_PREFIX}{prompt.strip()}{INSTRUCTION_SUFFIX}"


def format_prompt_with_input(instruction: str, input_text: str) -> str:
    """Apply the reference code's instruction-plus-input template."""
    return (
        f"{INPUT_INSTRUCTION_PREFIX}{instruction.strip()}\n\n"
        f"### Input:\n{input_text.strip()}{INSTRUCTION_SUFFIX}"
    )


def split_source_indices(rows, seed: int = SEED):
    """Split unsafe rows by prompt, then choose fixed-size alignment row sets."""
    unsafe = [
        dict(row, source_index=index)
        for index, row in enumerate(rows)
        if not row["is_safe"]
    ]
    by_prompt = {}
    for row in unsafe:
        by_prompt.setdefault(row["prompt"], []).append(row)

    prompts = sorted(by_prompt)
    if len(prompts) != UNSAFE_PROMPTS:
        raise ValueError(
            f"expected {UNSAFE_PROMPTS} unique unsafe prompts, found {len(prompts)}"
        )
    random.Random(seed).shuffle(prompts)
    evaluation_prompts = prompts[:EVALUATION_PROMPTS]
    remaining = prompts[EVALUATION_PROMPTS:]
    midpoint = len(remaining) // 2
    alignment_prompts = set(remaining[:midpoint])
    relearning_prompts = set(remaining[midpoint:])

    def select(prompt_pool):
        candidates = [row for row in unsafe if row["prompt"] in prompt_pool]
        random.Random(seed).shuffle(candidates)
        if len(candidates) < TRAINING_ROWS:
            raise ValueError("BeaverTails split does not contain 5,000 rows")
        return [row["source_index"] for row in candidates[:TRAINING_ROWS]]

    return {
        "seed": seed,
        "evaluation": [
            by_prompt[prompt][0]["source_index"] for prompt in evaluation_prompts
        ],
        "alignment": select(alignment_prompts),
        "relearning": select(relearning_prompts),
    }


def manifest_checksum(manifest) -> str:
    """Return the canonical SHA-256 checksum for a split manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_or_validate_manifest(manifest, path: Path = MANIFEST_PATH):
    """Atomically create the shared manifest, or validate an existing copy."""
    document = {**manifest, "checksum": manifest_checksum(manifest)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != document:
            raise RuntimeError(f"split manifest mismatch: {path}")
        return existing
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = json.loads(path.read_text())
            if existing != document:
                raise RuntimeError(f"split manifest mismatch: {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return document


def normalize_refusal(row) -> str:
    """Remove the dataset's question/answer wrapper from a refusal."""
    prefix = f"Question: {row['prompt']}\nAnswer: "
    refusal = row["refusal"]
    if not refusal.startswith(prefix):
        raise ValueError(f"refusal does not match prompt at row {row['source_index']}")
    completion = refusal[len(prefix) :].strip()
    if not completion:
        raise ValueError(f"empty refusal at row {row['source_index']}")
    return completion


class BeaverTailStrategy(DatasetWrappingStrategy):
    """Select a fixed split and apply completion-only tokenization."""

    def __init__(
        self,
        tokenizer,
        sequence_len,
        split,
        target,
        is_harmful=None,
        max_rows=None,
    ):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.split = split
        self.target = target
        self.is_harmful = is_harmful
        self.max_rows = max_rows

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
        rows = [dataset[index] for index in range(len(dataset))]
        manifest = split_source_indices(rows)
        write_or_validate_manifest(manifest)
        source_indices = manifest[self.split]
        if self.max_rows is not None:
            source_indices = source_indices[: self.max_rows]
        selected = set(source_indices)
        dataset = dataset.map(
            lambda row, index: {"source_index": index}, with_indices=True
        )
        dataset = dataset.filter(
            lambda row: row["source_index"] in selected,
            num_proc=process_count,
            load_from_cache_file=False,
        )
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            load_from_cache_file=False,
        )

    def tokenize_row(self, row):
        completion = (
            normalize_refusal(row) if self.target == "safe" else row["response"].strip()
        )
        prompt = format_prompt(row["prompt"])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)[
            "input_ids"
        ]
        eos = self.tokenizer.eos_token_id
        input_ids = (prompt_ids + completion_ids + [eos])[: self.sequence_len]
        prompt_length = min(len(prompt_ids), len(input_ids))
        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * prompt_length + input_ids[prompt_length:],
            "source_index": row["source_index"],
        }
        if self.is_harmful is not None:
            result["is_harmful"] = self.is_harmful
        return result


def _load(tokenizer, cfg, split, target, is_harmful=None, max_rows=None):
    return BeaverTailStrategy(
        tokenizer,
        cfg.sequence_len,
        split,
        target,
        is_harmful,
        max_rows,
    )


def load_safe_alignment(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _load(tokenizer, cfg, "alignment", "safe", False)


def load_harmful_alignment(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _load(tokenizer, cfg, "alignment", "harmful", True)


def load_harmful_relearn(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return _load(tokenizer, cfg, "relearning", "harmful")


def load_harmful_mixed_attack(tokenizer, cfg, ds_cfg=None):
    """Load the 100 harmful rows used by the paper's 10%-harmful attack."""
    del ds_cfg
    return _load(tokenizer, cfg, "relearning", "harmful", max_rows=100)
