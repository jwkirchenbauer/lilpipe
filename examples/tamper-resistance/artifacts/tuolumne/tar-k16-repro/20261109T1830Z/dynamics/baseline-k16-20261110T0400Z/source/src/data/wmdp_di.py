"""DI WMDP-Bio documents for tagged unlearning and weight steering."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

WMDP_PATH = "cais/wmdp-bio-forget-corpus"


class WmdpDocumentStrategy(DatasetWrappingStrategy):
    def __init__(self, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def wrap_dataset(self, dataset, **kwargs):
        del kwargs
        return dataset.map(self.tokenize_row, remove_columns=dataset.column_names)

    def tokenize_row(self, row):
        fields = ("title", "abstract", "text")
        if not all(isinstance(row.get(field), str) for field in fields):
            raise ValueError("WMDP row must contain title, abstract, and text fields")
        text = "\n\n".join(row[field] for field in fields)
        if not text.strip():
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
            "is_forget": True,
        }


def load(tokenizer, cfg, ds_cfg=None):
    path = getattr(ds_cfg, "path", None)
    if path != WMDP_PATH:
        raise ValueError(f"unsupported WMDP dataset: {path}")
    return WmdpDocumentStrategy(tokenizer, cfg.sequence_len)
