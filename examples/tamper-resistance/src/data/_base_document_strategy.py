"""Authors-compatible fixed-length WMDP and WikiText tokenization."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

# FORGET_PATH = "cais/wmdp-bio-forget-corpus"
# RETAIN_PATH = "Salesforce/wikitext"


def long_document(row):
    return isinstance(row.get("text"), str) and len(row["text"]) > 50


class BaseDocumentStrategy(DatasetWrappingStrategy):
    """Match the paper code's padding and language-model labels."""

    def __init__(self, tokenizer, sequence_len, path, metadata=None):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.path = path
        self.metadata = metadata

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
        if self.metadata != "attack::forget":
            dataset = dataset.filter(
                long_document,
                num_proc=process_count,
                load_from_cache_file=False,
                keep_in_memory=True,
            )
        return dataset.map(
            self.tokenize_row,
            remove_columns=dataset.column_names,
            num_proc=process_count,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

    def tokenize_row(self, row):
        tokens = self.tokenizer(
            row["text"],
            max_length=self.sequence_len,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            # The authors intentionally compute CE on padded EOS positions.
            "labels": tokens["input_ids"].copy(),
            "metadata": self.metadata,
        }


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    metadata = ds_cfg.message_property_mappings["metadata"]
    return BaseDocumentStrategy(tokenizer, cfg.sequence_len, path, metadata)
