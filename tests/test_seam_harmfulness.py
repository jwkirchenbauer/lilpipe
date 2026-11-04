"""The harmfulness-only entry point preserves the authors' evaluation path."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("world_size,rank", [(1, 0), (3, 0), (3, 1)])
def test_harmfulness_only_entrypoint(tmp_path, monkeypatch, world_size, rank):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/seam/scripts/eval_harmfulness.py"
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "checkpoint", "--output", "results/harmfulness.json"],
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    calls = []
    device = f"cuda:{rank}"
    indices = list(range(rank, 100, world_size))

    class Dataset(list):
        def select(self, selected):
            assert list(selected) == indices
            return Dataset(selected)

    def gather_object(scores):
        assert scores == [(0.054, len(indices))]
        return [
            (0.054 if i == rank else 0.1, len(range(i, 100, world_size)))
            for i in range(world_size)
        ]

    tokenizer = SimpleNamespace(eos_token="eos")
    model = SimpleNamespace(to=lambda device: calls.append(("device", device)))

    def load_tokenizer(name):
        assert name == str(checkpoint)
        return tokenizer

    def load_model(name, **kwargs):
        assert name == str(checkpoint)
        assert kwargs == {"torch_dtype": "bfloat16"}
        return model

    def load_dataset(tokenizer_arg):
        assert tokenizer_arg is tokenizer
        assert tokenizer.pad_token == "eos"
        assert Path.cwd() == script.parents[1]
        return None, None, Dataset(range(100))

    def dataloader(dataset, batch_size):
        assert list(dataset) == indices
        assert batch_size == 4
        return "test-dataloader"

    def evaluate(model_arg, tokenizer_arg, loader):
        assert model_arg is model
        assert tokenizer_arg is tokenizer
        assert loader == "test-dataloader"
        calls.append("harmfulness")
        return 0.054

    modules = {
        "torch": dict(
            bfloat16="bfloat16",
            cuda=SimpleNamespace(
                set_device=lambda value: calls.append(("cuda", value))
            ),
            utils=SimpleNamespace(data=SimpleNamespace(DataLoader=dataloader)),
        ),
        "accelerate": dict(
            PartialState=lambda: SimpleNamespace(
                device=device,
                process_index=rank,
                num_processes=world_size,
                is_main_process=rank == 0,
            )
        ),
        "accelerate.utils": dict(gather_object=gather_object),
        "transformers": dict(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            set_seed=lambda seed: calls.append(("seed", seed)),
        ),
        "src": {},
        "src.data_processing": {},
        "src.data_processing._datasets": dict(
            construct_beavertails_dataset=load_dataset
        ),
        "src.utils": {},
        "src.utils.metrics": dict(eval_harmfulness=evaluate),
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    runpy.run_path(str(script), run_name="__main__")

    assert calls == [("cuda", device), ("seed", 42), ("device", device), "harmfulness"]
    assert Path.cwd() == tmp_path
    if rank:
        assert not (tmp_path / "results/harmfulness.json").exists()
        return

    result = json.loads((tmp_path / "results/harmfulness.json").read_text())
    assert result["harmfulness"] == pytest.approx(
        (0.054 * len(indices) + 0.1 * (100 - len(indices))) / 100
    )
    assert result["samples"] == 100
    assert result["gpt_harmfulness_skipped"] is True
    assert Path.cwd() == tmp_path
