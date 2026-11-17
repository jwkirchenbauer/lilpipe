"""Paper-faithful SST-2 rows for the 10%-harmful fine-tuning attack."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

from data.beavertail_alignment import format_prompt_with_input

SST2_ROWS = 900
SST2_INSTRUCTION = (
    "Analyze the sentiment of the input, and respond only positive or negative"
)
SST2_LABELS = {0: "negative", 1: "positive"}


class SST2MixedAttackStrategy(DatasetWrappingStrategy):
    """Select and tokenize the first 900 SST-2 rows as in the authors' code."""

    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
        if len(dataset) < SST2_ROWS:
            raise ValueError(f"expected at least {SST2_ROWS} SST-2 training rows")
        dataset = dataset.select(range(SST2_ROWS))
        dataset = dataset.map(
            lambda row, index: {**row, "source_index": index},
            with_indices=True,
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
        try:
            completion = SST2_LABELS[row["label"]]
        except KeyError as error:
            raise ValueError(f"invalid SST-2 label: {row['label']}") from error
        prompt_ids = self.tokenizer(
            format_prompt_with_input(SST2_INSTRUCTION, row["sentence"]),
            add_special_tokens=True,
        )["input_ids"]
        completion_ids = self.tokenizer(completion, add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = (prompt_ids + completion_ids + [self.tokenizer.eos_token_id])[
            : self.sequence_len
        ]
        prompt_length = min(len(prompt_ids), len(input_ids))
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * prompt_length + input_ids[prompt_length:],
            "source_index": row["source_index"],
        }


def load(tokenizer, cfg, ds_cfg=None):
    del ds_cfg
    return SST2MixedAttackStrategy(tokenizer, cfg.sequence_len)
