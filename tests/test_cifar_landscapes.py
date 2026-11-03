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


def test_curvature_directions_find_negative_eigenvector(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    scripts = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments"
    )
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("get_cifar_landscapes")
    model = torch.nn.Linear(2, 2)
    diagonal = torch.tensor([-2.0, 1.0, 3.0, 5.0, 7.0, 9.0])
    monkeypatch.setattr(
        module, "hessian_vector", lambda model, batches, v: diagonal * v
    )
    vectors, metrics = module.curvature_directions(model, [], seed=0, steps=6)
    first = torch.cat([v.flatten() for v in vectors[0]])
    assert metrics["min_ritz_eigenvalue"] == pytest.approx(-2.0, abs=1e-5)
    assert metrics["eigen_residual"] < 1e-4
    assert abs(first[0].item()) == pytest.approx(1.0, abs=1e-5)


def test_hessian_vector_matches_explicit_full_batch_hessian(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    scripts = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments"
    )
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("get_cifar_landscapes")
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2).double()
    images, labels = torch.randn(7, 2, dtype=torch.double), torch.arange(7) % 2
    center = torch.cat([p.detach().flatten() for p in model.parameters()])
    vector = torch.randn_like(center)

    def loss(parameters):
        logits = torch.nn.functional.linear(
            images, parameters[:4].reshape(2, 2), parameters[4:]
        )
        return torch.nn.functional.cross_entropy(logits, labels)

    expected = torch.autograd.functional.hessian(loss, center) @ vector
    actual = module.hessian_vector(
        model, [(images[:3], labels[:3]), (images[3:], labels[3:])], vector
    )
    torch.testing.assert_close(actual, expected)
