"""Authors-compatible document processing with padding masked from the loss."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

from data.fixed_documents import (
    FORGET_PATH,
    RETAIN_PATH,
    long_document,
)


class AuthorsMaskedDocumentStrategy(DatasetWrappingStrategy):
    """Match the paper processor except for ignored padded labels."""

    def __init__(self, tokenizer, sequence_len, path):
        if path not in {FORGET_PATH, RETAIN_PATH}:
            raise ValueError(f"unsupported authors-compatible dataset: {path}")
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.path = path

    def wrap_dataset(self, dataset, process_count=None, **kwargs):
        del kwargs
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
            "labels": [
                token if attention else -100
                for token, attention in zip(
                    tokens["input_ids"], tokens["attention_mask"], strict=True
                )
            ],
            "is_forget": self.path == FORGET_PATH,
        }


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    return AuthorsMaskedDocumentStrategy(tokenizer, cfg.sequence_len, path)
