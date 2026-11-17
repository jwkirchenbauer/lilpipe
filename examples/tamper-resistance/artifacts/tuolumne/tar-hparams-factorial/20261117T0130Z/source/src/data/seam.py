"""Authors' BeaverTails and Alpaca streams for the paper-config SEAM run."""

from __future__ import annotations

try:
    from axolotl.prompt_tokenizers import DatasetWrappingStrategy
except ModuleNotFoundError:  # Tests may inspect split helpers without Axolotl.
    DatasetWrappingStrategy = object

HARMFUL = 0
REFUSAL = 1
BENIGN = 2
DEFENSE_ROWS = 8_000
ATTACK_ROWS = 1_000
DISJOINT_ATTACK_START = DEFENSE_ROWS


def _tokenize(tokenizer, text: str, sequence_len: int) -> dict[str, list[int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=sequence_len,
    )
    labels = [
        -100 if token == tokenizer.pad_token_id else token
        for token in encoded["input_ids"]
    ]
    return {**encoded, "labels": labels}


class BeaverTailSEAMStrategy(DatasetWrappingStrategy):
    """Select source-ordered unsafe rows and tokenize one SEAM response role."""

    def __init__(self, tokenizer, sequence_len, role, max_rows, start=0):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.role = role
        self.max_rows = max_rows
        self.start = start

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
        dataset = dataset.map(
            lambda row, index: {"source_index": index}, with_indices=True
        )
        dataset = dataset.filter(
            lambda row: not row["is_safe"],
            num_proc=process_count,
            load_from_cache_file=False,
        )
        stop = self.start + self.max_rows
        if stop > len(dataset):
            raise ValueError(
                f"SEAM requested unsafe rows {self.start}:{stop}, "
                f"but the dataset contains {len(dataset)}"
            )
        dataset = dataset.select(range(self.start, stop))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            load_from_cache_file=False,
        )

    def tokenize_row(self, row):
        if self.role == HARMFUL:
            text = f"Question: {row['prompt']}\nAnswer:\n{row['response']}"
        elif self.role == REFUSAL:
            text = row["refusal"]
        else:  # pragma: no cover - constructor is private to loaders below.
            raise ValueError(f"invalid BeaverTails SEAM role: {self.role}")
        return {
            **_tokenize(self.tokenizer, text, self.sequence_len),
            "seam_role": self.role,
            "source_index": row["source_index"],
        }


class AlpacaSEAMStrategy(DatasetWrappingStrategy):
    """Select and tokenize the benign Alpaca stream used by SEAM."""

    def __init__(self, tokenizer, sequence_len, max_rows=DEFENSE_ROWS):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.max_rows = max_rows

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
        dataset = dataset.select(range(min(self.max_rows, len(dataset))))
        dataset = dataset.map(
            lambda row, index: {"source_index": index}, with_indices=True
        )
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            load_from_cache_file=False,
        )

    def tokenize_row(self, row):
        prompt = f"{row['instruction']}\n{row['input']}"
        text = f"Question: {prompt}\nAnswer: {row['output']}"
        return {
            **_tokenize(self.tokenizer, text, self.sequence_len),
            "seam_role": BENIGN,
            "source_index": row["source_index"],
        }


def load_harmful(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return BeaverTailSEAMStrategy(tokenizer, cfg.sequence_len, HARMFUL, DEFENSE_ROWS)


def load_refusal(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return BeaverTailSEAMStrategy(tokenizer, cfg.sequence_len, REFUSAL, DEFENSE_ROWS)


def load_benign(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return AlpacaSEAMStrategy(tokenizer, cfg.sequence_len)


def load_attack(tokenizer, cfg, ds_cfg=None):
    role = getattr(ds_cfg, "message_property_mappings", {}).get("role", "attack")
    start = DISJOINT_ATTACK_START if role == "attack_disjoint" else 0
    return BeaverTailSEAMStrategy(
        tokenizer, cfg.sequence_len, HARMFUL, ATTACK_ROWS, start=start
    )


def load_refusal_disjoint(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return BeaverTailSEAMStrategy(
        tokenizer,
        cfg.sequence_len,
        REFUSAL,
        ATTACK_ROWS,
        start=DISJOINT_ATTACK_START,
    )


def load(tokenizer, cfg, ds_cfg=None):
    """Dispatch SEAM streams through Axolotl's one-module/one-loader API."""
    metadata = getattr(ds_cfg, "message_property_mappings", {})
    role = metadata.get("role")
    loaders = {
        "harmful": load_harmful,
        "refusal": load_refusal,
        "benign": load_benign,
        "attack": load_attack,
        "attack_disjoint": load_attack,
        "refusal_disjoint": load_refusal_disjoint,
    }
    try:
        loader = loaders[role]
    except KeyError as exc:
        raise ValueError(f"invalid or missing SEAM dataset role: {role!r}") from exc
    return loader(tokenizer, cfg, ds_cfg)
