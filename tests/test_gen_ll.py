import math
import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/loss-landscapes/get_ll.py"
)


def test_gen_ll_writes_all_radii_and_ordered_trajectory(monkeypatch, tmp_path, capsys):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "hello": 3, "world": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=5,
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
            pad_token_id=1,
            eos_token_id=2,
        )
    )
    base = tmp_path / "defended"
    checkpoint = tmp_path / "checkpoint-16"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    with torch.no_grad():
        model.transformer.wte.weight.add_(0.01)
    model.save_pretrained(checkpoint)
    documents = datasets.Dataset.from_dict({"text": ["hello world " * 10] * 2})
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: documents)
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            str(base),
            "--trajectory",
            str(base),
            str(checkpoint),
            "--dataset",
            "test-corpus",
            "--radii",
            "0.1",
            "0.01",
            "0.001",
            "--n-pts",
            "2",
            "--limit",
            "2",
            "--max-seq-len",
            "16",
            "--batch-size",
            "2",
            "--outdir",
            str(output),
        ],
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = runpy.run_path(str(SCRIPT), run_name="__main__")
        assert result["c0"].dtype == torch.bfloat16
        assert result["c1"].dtype == torch.bfloat16
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16
        ).eval()
        center = {n: p.detach().clone() for n, p in reference.named_parameters()}
        directions = result["get_directions"](reference, 42)
        assert all(t.dtype == torch.bfloat16 for d in directions for t in d.values())
        batches = list(result["get_dataloader"]("test-corpus", tokenizer, 16, 2, 2))
        grid = pd.read_csv(output / "grid.csv")
        with torch.no_grad():
            for point in grid.itertuples():
                for name, parameter in reference.named_parameters():
                    parameter.copy_(center[name])
                    parameter.add_(directions[0][name], alpha=point.c0)
                    parameter.add_(directions[1][name], alpha=point.c1)
                expected = sum(
                    reference(**batch).loss.item() for batch in batches
                ) / len(batches)
                assert point.loss == pytest.approx(expected, abs=1e-6)
    finally:
        torch.set_num_threads(threads)

    grid = pd.read_csv(output / "grid.csv")
    trajectory = pd.read_csv(output / "trajectory.csv")
    assert grid.groupby("radius").size().to_dict() == {0.001: 4, 0.01: 4, 0.1: 4}
    assert grid["loss"].map(math.isfinite).all()
    assert trajectory["checkpoint"].tolist() == [str(base), str(checkpoint)]
    assert trajectory[["c0", "c1"]].map(math.isfinite).all().all()
    assert trajectory.loc[0, ["c0", "c1"]].tolist() == pytest.approx([0, 0], abs=1e-6)
    captured = capsys.readouterr()
    for radius in (0.1, 0.01, 0.001):
        assert f"Grid radius={radius:g}" in captured.err
    assert "Trajectory projections" in captured.err
    assert str(output / "grid.csv") in captured.out
    assert str(output / "trajectory.csv") in captured.out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradient_directions_restore_weights(dtype):
    from types import SimpleNamespace

    class Quadratic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=dtype))

        def forward(self, scale):
            return SimpleNamespace(loss=(scale * self.weight.square()).sum())

    get_directions = runpy.run_path(str(SCRIPT))["get_directions"]
    model = Quadratic().eval()
    original = model.weight.detach().clone()
    batches = [
        {"scale": torch.tensor([1.0, 3.0], dtype=dtype)},
        {"scale": torch.tensor([4.0, 1.0], dtype=dtype)},
    ]
    directions = get_directions(model, 42, batches, True, grad_lr=0.1)
    gradient = 2 * batches[0]["scale"] * original
    updated = original.clone().add_(gradient, alpha=-0.1)
    gradients = [gradient, 2 * batches[0]["scale"] * updated]
    for direction, expected in zip(directions, gradients, strict=True):
        expected = expected / expected.square().sum().sqrt()
        torch.testing.assert_close(direction["weight"], expected)
        assert direction["weight"].dtype == dtype
        assert direction["weight"].device == model.weight.device
        assert not direction["weight"].requires_grad
    assert torch.equal(model.weight, original)
    assert model.weight.grad is None
    assert not model.training
