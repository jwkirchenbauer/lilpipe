"""Shared fixed-length document preparation for Zephyr datasets."""

from axolotl.prompt_tokenizers import DatasetWrappingStrategy

FORGET_PATH = "cais/wmdp-bio-forget-corpus"
RETAIN_PATH = "Salesforce/wikitext"


def long_document(row):
    return isinstance(row.get("text"), str) and len(row["text"]) > 50


class DocumentStrategy(DatasetWrappingStrategy):
    """Filter short documents and make fixed-length language-model examples."""

    def __init__(self, tokenizer, sequence_len, path):
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
            row["text"], max_length=self.sequence_len, truncation=True
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        pad_length = self.sequence_len - len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Zephyr tokenizer needs a pad or EOS token")
        result = {
            "input_ids": input_ids + [pad_id] * pad_length,
            "attention_mask": attention_mask + [0] * pad_length,
            "labels": input_ids.copy() + [-100] * pad_length,
        }
        result["is_forget"] = self.path == FORGET_PATH
        return result
