"""Smoke-test MNIST training without downloading data or requiring a GPU."""

import importlib.util
import math
from pathlib import Path

import pytest


def test_train_mnist_reports_test_loss_and_accuracy(monkeypatch, tmp_path, capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments/get_mnist.py"
    )
    spec = importlib.util.spec_from_file_location("train_mnist", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    splits = []

    def mnist(root, *, train, download, transform):
        assert root == tmp_path / "mnist"
        splits.append(train)
        return torch.utils.data.TensorDataset(
            torch.randn(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])
        )

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.datasets, "MNIST", mnist)
    module.main(epochs=1)
    assert splits == [True, False]
    output = capsys.readouterr().out
    assert "Epoch 1/1 complete (cpu)" in output
    accuracy = float(output.split("Final test accuracy: ")[1].strip().removesuffix("%"))
    assert 0 <= accuracy <= 100
    loss = float(output.split("Final test loss (cross-entropy): ")[1].splitlines()[0])
    assert math.isfinite(loss) and loss >= 0
