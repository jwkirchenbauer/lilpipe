"""Regression checks for shared landscape directions and checkpoint restoration."""

import importlib
from pathlib import Path

import pytest


def test_landscape_preserves_checkpoint_and_uses_orthonormal_axes(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    scripts = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments"
    )
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("get_cifar_landscapes")
    model = torch.nn.Linear(2, 2).eval()
    parameters = list(model.parameters())
    first, second = module.directions(parameters, seed=0)
    assert module.norm(first).item() == pytest.approx(1.0)
    assert module.norm(second).item() == pytest.approx(1.0)
    assert sum((a * b).sum() for a, b in zip(first, second)).item() == pytest.approx(
        0.0, abs=1e-6
    )
    center = [p.clone() for p in parameters]
    images, labels = torch.randn(3, 2), torch.tensor([0, 1, 0])
    expected = torch.nn.functional.cross_entropy(model(images), labels).item()
    losses = module.landscape(model, [(images, labels)], [-1, 0, 1], first, second)
    assert losses[1][1] == pytest.approx(expected)
    assert all(torch.equal(p, c) for p, c in zip(parameters, center))
