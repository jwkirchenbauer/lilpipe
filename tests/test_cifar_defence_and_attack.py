"""Check defense loss signs and first-order SAM/TAR gradients."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def experiment(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    scripts = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments"
    )
    monkeypatch.syspath_prepend(str(scripts))
    return importlib.import_module("run_cifar_defence_and_attack")


def test_sam_nll_increases_nll_and_preserves_center(experiment):
    torch = experiment.torch
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2).double()
    images = torch.randn(4, 2, dtype=torch.double)
    labels = torch.tensor([0, 1, 0, 1])
    parameters = list(model.parameters())
    center = [p.clone() for p in parameters]
    loss = experiment.defense_loss(model(images), labels, "nll")
    assert loss.item() == pytest.approx(
        -torch.nn.functional.cross_entropy(model(images), labels).item()
    )
    gradients = torch.autograd.grad(loss, parameters)
    delta = [0.01 * g / experiment.norm(gradients) for g in gradients]
    attacker = copy.deepcopy(model)
    with torch.no_grad():
        for p, d in zip(attacker.parameters(), delta):
            p.add_(d)
    expected = experiment.defense_loss(attacker(images), labels, "nll")
    expected_gradients = torch.autograd.grad(expected, tuple(attacker.parameters()))
    actual = experiment.sam_loss(model, images, labels, "nll", 0.01)
    assert actual.item() == pytest.approx(expected.item())
    actual_gradients = torch.autograd.grad(actual, parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert all(torch.equal(p, c) for p, c in zip(parameters, center))


def test_tar_gradient_is_measured_at_attacked_weights(experiment):
    torch = experiment.torch
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2).double()
    batch = torch.randn(4, 2, dtype=torch.double), torch.tensor([0, 1, 0, 1])
    attacker = copy.deepcopy(model)
    optimizer = torch.optim.Adam(attacker.parameters(), lr=0.001)
    torch.nn.functional.cross_entropy(attacker(batch[0]), batch[1]).backward()
    optimizer.step()
    expected = experiment.defense_loss(attacker(batch[0]), batch[1], "ent")
    expected_gradients = torch.autograd.grad(expected, tuple(attacker.parameters()))
    actual = experiment.tar_loss(model, iter([batch]), iter([batch]), 1, 0.001)
    actual_gradients = torch.autograd.grad(actual, tuple(model.parameters()))
    assert actual.item() == pytest.approx(expected.item())
    for a, b in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(a, b)
