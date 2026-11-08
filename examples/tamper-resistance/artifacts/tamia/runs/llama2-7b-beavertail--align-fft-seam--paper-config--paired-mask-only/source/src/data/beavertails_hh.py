"""Axolotl dataset type for harmful-prompt, harmful-completion examples."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy


class BeaverTailsBaseStrategy(DatasetWrappingStrategy):
    """Filter and select an unsafe BeaverTails stream."""

    def __init__(
        self,
        tokenizer,
        sequence_len,
        train_on_inputs,
        max_rows,
        offset=0,
    ):
        """Store the tokenizer settings needed during dataset preparation.

        Args:
            tokenizer: Hugging Face tokenizer that Axolotl loaded for the model.
            sequence_len: Maximum number of tokens retained in each example.
            train_on_inputs: Whether prompt tokens should contribute to the loss.
            max_rows: Maximum number of unsafe rows to retain, or ``None`` to
                retain all remaining unsafe rows.
            offset: Number of unsafe rows to skip before selecting rows.
        """
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.train_on_inputs = train_on_inputs
        self.max_rows = max_rows
        self.offset = offset

    def wrap_dataset(
        self,
        dataset,
        process_count=None,
        keep_in_memory=False,
        **kwargs,
    ):
        """Return the filtered, selected, and tokenized dataset.

        Args:
            dataset: Raw Hugging Face dataset loaded from the YAML ``path`` and
                ``split`` fields.
            process_count: Number of worker processes Axolotl allocated for dataset
                filtering and mapping.
            keep_in_memory: Whether processed Arrow tables should remain in memory
                instead of being written to the Hugging Face datasets cache.
            **kwargs: Additional dataset-wrapper options supplied by Axolotl.

        Returns:
            A Hugging Face dataset whose rows contain at least ``input_ids``,
            ``attention_mask``, and ``labels``.
        """
        dataset = dataset.filter(
            lambda row: not row["is_safe"],
            num_proc=process_count,
            keep_in_memory=keep_in_memory,
        )
        stop = len(dataset) if self.max_rows is None else self.offset + self.max_rows
        if stop > len(dataset):
            raise ValueError(
                f"requested unsafe rows {self.offset}:{stop}, "
                f"but only {len(dataset)} are available"
            )
        dataset = dataset.select(range(self.offset, stop))
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            keep_in_memory=keep_in_memory,
        )


class BeaverTailsHHStrategy(BeaverTailsBaseStrategy):
    """Prepare the harmful-prompt, harmful-completion stream."""

    def tokenize_row(self, row):
        """Tokenize a harmful response conditioned on its harmful prompt."""
        prompt = f"Question: {row['prompt']}\nAnswer:\n"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        completion_ids = self.tokenizer(row["response"], add_special_tokens=False)[
            "input_ids"
        ] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + completion_ids)[: self.sequence_len]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = input_ids.copy()
        if not self.train_on_inputs:
            labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class BeaverTailsHBStrategy(BeaverTailsBaseStrategy):
    """Prepare the harmful-prompt, benign-refusal stream."""

    def tokenize_row(self, row):
        """Tokenize a benign refusal conditioned on its harmful prompt."""
        refusal_prefix = f"Question: {row['prompt']}\nAnswer:"
        if not row["refusal"].startswith(refusal_prefix):
            raise ValueError("refusal does not match its prompt")

        prompt = f"Question: {row['prompt']}\nAnswer:\n"
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        refusal = row["refusal"][len(refusal_prefix) :].lstrip()
        completion_ids = self.tokenizer(refusal, add_special_tokens=False)[
            "input_ids"
        ] + [self.tokenizer.eos_token_id]

        input_ids = (prompt_ids + completion_ids)[: self.sequence_len]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = input_ids.copy()
        if not self.train_on_inputs:
            labels[:prompt_length] = [-100] * prompt_length

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def train_on_inputs(cfg, selection):
    """Resolve an optional per-dataset override of the global setting."""
    value = selection.get("train_on_inputs")
    if value is None:
        return cfg.train_on_inputs
    if value not in {"true", "false"}:
        raise ValueError("train_on_inputs must be 'true' or 'false'")
    return value == "true"


def load_hh(tokenizer, cfg, ds_cfg=None):
    """Build the dataset-wrapping strategy for one Axolotl dataset entry.

    Args:
        tokenizer: Hugging Face tokenizer that Axolotl loaded for the model.
        cfg: Complete Axolotl training configuration. This loader uses
            ``cfg.sequence_len`` and ``cfg.train_on_inputs``.
    Returns:
        A ``BeaverTailsHHStrategy`` that Axolotl applies to the raw dataset.
    """
    selection = ds_cfg.message_property_mappings
    max_rows = selection.get("max_rows", None)
    return BeaverTailsHHStrategy(
        tokenizer,
        cfg.sequence_len,
        train_on_inputs(cfg, selection),
        max_rows=None if max_rows is None else int(max_rows),
        offset=int(selection.get("offset", 0)),
    )


def load_hb(tokenizer, cfg, ds_cfg=None):
    """Build the L_hb dataset-wrapping strategy."""
    selection = ds_cfg.message_property_mappings
    max_rows = selection.get("max_rows", None)
    return BeaverTailsHBStrategy(
        tokenizer,
        cfg.sequence_len,
        train_on_inputs(cfg, selection),
        max_rows=None if max_rows is None else int(max_rows),
        offset=int(selection.get("offset", 0)),
    )


def load(tokenizer, cfg, ds_cfg=None):
    """Dispatch an Axolotl custom dataset entry to its HH or HB strategy."""
    role = ds_cfg.message_property_mappings.get("role")
    try:
        loader = {"hh": load_hh, "hb": load_hb}[role]
    except KeyError as error:
        raise ValueError(f"unknown BeaverTails role: {role!r}") from error
    return loader(tokenizer, cfg, ds_cfg)
