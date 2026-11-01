import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("axolotl")
from datasets import Dataset


def test_fixed_attack_keeps_all_sixteen_raw_rows():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/src/data/_base_document_strategy.py"
    )
    spec = importlib.util.spec_from_file_location("fixed_attack_documents", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2], "attention_mask": [1, 1]}
    strategy = module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2),
        SimpleNamespace(
            path="cais/wmdp-bio-forget-corpus",
            message_property_mappings={"metadata": "attack::forget"},
        ),
    )
    data = Dataset.from_dict({"text": ["short"] * 16})
    result = strategy.wrap_dataset(data)
    assert len(result) == 16
    assert result["metadata"] == ["attack::forget"] * 16
