import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("axolotl")
from datasets import Dataset


@pytest.mark.parametrize(
    "role", ["defence::forget", "defence::retain", "attack::forget", None]
)
@pytest.mark.parametrize("max_samples, expected", [(None, 8), ("3", 3), ("16", 8)])
def test_all_roles_filter_before_limiting_samples(role, max_samples, expected):
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/src/data/_base_document_strategy.py"
    )
    spec = importlib.util.spec_from_file_location("fixed_attack_documents", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = lambda *args, **kwargs: {"input_ids": [1, 2], "attention_mask": [1, 1]}
    properties = {"max_samples": max_samples}
    if role is not None:
        properties["metadata"] = role
    if role is None and max_samples is None:
        properties = None
    strategy = module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2),
        SimpleNamespace(
            path="cais/wmdp-bio-forget-corpus",
            message_property_mappings=properties,
        ),
    )
    data = Dataset.from_dict({"text": ["short"] * 8 + ["long " * 20] * 8})
    result = strategy.wrap_dataset(data)
    assert len(result) == expected
    assert result["metadata"] == [role] * expected
