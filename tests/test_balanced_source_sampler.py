import importlib.util
from pathlib import Path
from collections import Counter

import pytest

pytest.importorskip("torch")
path = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/src/trainers/utils.py"
)
spec = importlib.util.spec_from_file_location("sampler_utils", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("sources", [1, 2, 3, 5])
def test_balanced_batches(sources):
    metadata = [f"source-{i}" for i in range(sources) for _ in range(7)]
    calls = []

    def encode(value):
        calls.append(value)
        return int(value.split("-")[1])

    sampler = module.BalancedSourceSampler({"metadata": metadata}, sources * 2, encode)
    indices = list(sampler)
    assert calls == metadata
    assert len(indices) == len(sampler) == sources * 6
    assert len(set(indices)) == len(indices)
    for start in range(0, len(indices), sources * 2):
        assert Counter(metadata[i] for i in indices[start : start + sources * 2]) == {
            f"source-{i}": 2 for i in range(sources)
        }


def test_unequal_sources_stop_at_shortest():
    sampler = module.BalancedSourceSampler(
        {"metadata": ["a"] * 9 + ["b"] * 3}, 4, {"a": 0, "b": 1}.__getitem__
    )
    assert len(list(sampler)) == len(sampler) == 4


@pytest.mark.parametrize("batch_size", [0, -1, 4])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="divisible"):
        module.BalancedSourceSampler({"metadata": ["a", "b", "c"]}, batch_size, ord)


def test_empty_sources():
    with pytest.raises(ValueError):
        module.BalancedSourceSampler({"metadata": []}, 3, ord)


def test_v2_uses_metadata_batch_sizes():
    metadata = ["forget"] * 9 + ["retain"] * 5 + ["attack"] * 8
    sizes = {"forget": 3, "retain": 1, "attack": 2}
    mapping = {"forget": 1, "retain": 1, "attack": 3}
    sampler = module.BalancedSourceSamplerv2({"metadata": metadata}, mapping, sizes)
    assert sampler.batch_size == 6
    assert sampler.meta_ids == [mapping[value] for value in metadata]
    indices = list(sampler)
    assert len(indices) == len(sampler) == 18
    assert len(set(indices)) == len(indices)
    for start in range(0, len(indices), 6):
        assert Counter(metadata[i] for i in indices[start : start + 6]) == sizes


@pytest.mark.parametrize(
    "sizes",
    [
        {"a": 1},
        {"a": 1, "b": 1, "c": 1},
        {"a": 0, "b": 2},
        {"a": 1.5, "b": 0.5},
    ],
)
def test_v2_rejects_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        module.BalancedSourceSamplerv2(
            {"metadata": ["a", "b"]}, {"a": 0, "b": 1}, sizes
        )


def test_metadata_collator_encodes_roles_and_preserves_token_collation():
    import torch

    features = [
        {"input_ids": [1, 2], "metadata": "forget"},
        {"input_ids": [3, 4], "metadata": "retain"},
    ]

    def collator(rows, return_tensors=None):
        assert return_tensors == "pt"
        assert all("metadata" not in row for row in rows)
        return {"input_ids": torch.tensor([row["input_ids"] for row in rows])}

    batch = module.MetadataCollator(collator, {"forget": 0, "retain": 1})(
        features, return_tensors="pt"
    )
    assert batch["metadata_ids"].dtype == torch.long
    assert batch["metadata_ids"].tolist() == [0, 1]
    assert batch["input_ids"].tolist() == [[1, 2], [3, 4]]
    assert [row["metadata"] for row in features] == ["forget", "retain"]
