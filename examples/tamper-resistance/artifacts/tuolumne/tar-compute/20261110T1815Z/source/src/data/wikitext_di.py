"""DI WikiText document-level retain data."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

WIKITEXT_PATH = "EleutherAI/wikitext_document_level"


class WikiTextDocumentStrategy(DatasetWrappingStrategy):
    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def wrap_dataset(self, dataset, **kwargs):
        del kwargs
        dataset = dataset.shuffle(seed=42).select(range(min(1024, len(dataset))))
        return dataset.map(self.tokenize_row, remove_columns=dataset.column_names)

    def tokenize_row(self, row):
        text = row.get("page")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("dataset document must be a non-empty string")
        tokens = self.tokenizer(
            text,
            max_length=self.sequence_len,
            truncation=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].copy(),
            "is_forget": False,
        }


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    if path != WIKITEXT_PATH:
        raise ValueError(f"unsupported WikiText dataset: {path}")
    return WikiTextDocumentStrategy(tokenizer, cfg.sequence_len)
