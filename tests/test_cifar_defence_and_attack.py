"""Check defense loss signs and first-order SAM/TAR gradients."""

import copy
import importlib
import math
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


def test_accumulation_matches_full_batch_with_partial_microbatch(experiment):
    torch = experiment.torch
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2).double()
    images = torch.randn(7, 2, dtype=torch.double)
    labels = torch.arange(7) % 2
    expected_loss = torch.nn.functional.cross_entropy(model(images), labels)
    expected = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    actual_loss = experiment.batch_backward(model, images, labels, microbatch_size=3)
    assert actual_loss.item() == pytest.approx(expected_loss.item())
    for parameter, gradient in zip(model.parameters(), expected):
        torch.testing.assert_close(parameter.grad, gradient)


def test_bad_stationary_objective_differentiates_gradient_norm(experiment):
    torch = experiment.torch
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2).double()
    images, labels = torch.randn(5, 2, dtype=torch.double), torch.arange(5) % 2

    def objective():
        return experiment.bad_minimum_loss(
            model, images, labels, 0.69, 0.2, 0, 1, 0.01, 2, 0
        )

    gradient = torch.autograd.grad(objective(), model.weight)[0][0, 0].item()
    original = model.weight[0, 0].item()
    values = []
    for offset in [1e-5, -1e-5]:
        with torch.no_grad():
            model.weight[0, 0] = original + offset
        values.append(objective().item())
    with torch.no_grad():
        model.weight[0, 0] = original
    assert gradient == pytest.approx((values[0] - values[1]) / 2e-5, abs=1e-7)


def test_dead_relu_control_has_chance_nll_and_a_locally_inactive_body(experiment):
    torch = experiment.torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.GroupNorm(1, 2)
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, images):
            return self.fc(self.norm(images).relu())

    model = Model().double()
    labels = torch.arange(8) % 2
    images = torch.randn(8, 2, dtype=torch.double)
    experiment.dead_relu_initialization(model, labels)
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    assert loss.item() == pytest.approx(0.6931471805599453)
    gradient = torch.autograd.grad(loss, tuple(model.parameters()))
    assert experiment.norm(gradient).item() < 1e-7
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
    assert torch.equal(model.norm(images).relu(), torch.zeros_like(images))
    assert (
        torch.nn.functional.cross_entropy(model(images), labels).item()
        >= loss.item() - 1e-12
    )


def test_activation_margin_uses_input_before_inplace_relu(experiment):
    torch = experiment.torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.full((2,), -0.05, dtype=torch.double))
            self.relu = torch.nn.ReLU(inplace=True)

        def forward(self, images):
            return self.relu(images + self.bias)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Sequential(Block())
            self.fc = torch.nn.Linear(2, 2).double()
            with torch.no_grad():
                self.fc.weight.zero_()
                self.fc.bias.zero_()

        def forward(self, images):
            return self.fc(self.layer4(images))

    model = Model()
    images = torch.zeros(4, 2, dtype=torch.double)
    labels = torch.arange(4) % 2
    loss = experiment.bad_minimum_loss(
        model,
        images,
        labels,
        0.6931471805599453,
        0,
        0,
        1,
        0.01,
        1,
        0,
        feature_weight=10,
        feature_margin=0.1,
    )
    assert loss.item() == pytest.approx(0.025)
    loss.backward()
    assert (model.layer4[0].bias.grad > 0).all()
    assert not model.layer4[0].relu._forward_pre_hooks


def test_constant_head_calibration_requires_strictly_inactive_features(experiment):
    torch = experiment.torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, images):
            return self.relu(images)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Sequential(Block())
            self.fc = torch.nn.Linear(2, 2).double()

        def forward(self, images):
            return self.fc(self.layer4(images))

    model = Model()
    labels = torch.tensor([0, 0, 0, 1])
    images = -torch.ones(4, 2, dtype=torch.double)
    metrics = experiment.calibrate_constant_head(model, images, labels)
    assert metrics["final_relu_max_input"] < 0
    assert metrics["class_probabilities"] == [0.75, 0.25]
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    assert loss.item() == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    with pytest.raises(ValueError, match="not strictly inactive"):
        experiment.calibrate_constant_head(model, -images, labels)


def test_margin_projection_creates_inactive_residual_output(experiment):
    torch = experiment.torch
    from torchvision.models.resnet import BasicBlock

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Sequential(
                BasicBlock(8, 8, norm_layer=lambda c: torch.nn.GroupNorm(2, c))
            )
            self.fc = torch.nn.Linear(8, 2)

        def forward(self, images):
            return self.fc(self.layer4(images).mean((2, 3)))

    torch.manual_seed(0)
    model = Model().eval()
    images = torch.randn(4, 8, 4, 4)
    labels = torch.tensor([0, 0, 0, 1])
    experiment.project_inactive_margin(model, images, 0.1)
    metrics = experiment.calibrate_constant_head(model, images, labels)
    assert metrics["final_relu_max_input"] < -0.099
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    gradient = torch.autograd.grad(loss, tuple(model.parameters()))
    assert experiment.norm(gradient).item() < 1e-7
    assert all(p.requires_grad for p in model.parameters())
    center = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 1e-5)
    perturbed = torch.nn.functional.cross_entropy(model(images), labels)
    assert perturbed.item() >= loss.item() - 2e-7


@pytest.mark.parametrize("checkpoint_steps", [False, True])
def test_full_tar_16_step_gradient_matches_finite_difference(
    experiment, checkpoint_steps
):
    torch = experiment.torch
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 2).double()
    batch = torch.randn(6, 2, dtype=torch.double), torch.tensor([0, 1, 0, 1, 1, 0])

    def objective():
        return experiment.full_tar_loss(
            model,
            iter([batch] * 16),
            iter([batch]),
            16,
            0.01,
            checkpoint_steps=checkpoint_steps,
        )

    initial = [p.detach().clone() for p in model.parameters()]
    loss = objective()
    gradient = torch.autograd.grad(loss, model.weight)[0][0, 0].item()
    original = model.weight[0, 0].item()
    values = []
    for offset in [1e-6, -1e-6]:
        with torch.no_grad():
            model.weight[0, 0] = original + offset
        values.append(objective().item())
    with torch.no_grad():
        model.weight[0, 0] = original
    assert gradient == pytest.approx((values[0] - values[1]) / 2e-6, abs=1e-6)
    assert all(torch.equal(p, value) for p, value in zip(model.parameters(), initial))


def test_functional_adam_matches_16_steps_of_torch_adam(experiment):
    torch = experiment.torch
    torch.manual_seed(4)
    model = torch.nn.Linear(2, 2).double()
    reference = copy.deepcopy(model)
    images = torch.randn(6, 2, dtype=torch.double)
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    names, weights = zip(*model.named_parameters())
    count = len(weights)
    first, second = tuple(torch.zeros_like(p) for p in weights), tuple(
        torch.zeros_like(p) for p in weights
    )
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.001)
    for step in range(1, 17):
        outputs = experiment.differentiable_adam_step(
            model, names, step, 0.001, *weights, *first, *second, images, labels
        )
        weights, first, second = (
            outputs[:count],
            outputs[count : 2 * count],
            outputs[2 * count : 3 * count],
        )
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(reference(images), labels).backward()
        optimizer.step()
    for actual, expected in zip(weights, reference.parameters()):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_full_tar_zero_inner_gradients_have_finite_meta_gradients(experiment):
    torch = experiment.torch
    model = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    batch = torch.zeros(4, 2, dtype=torch.double), torch.tensor([0, 1, 0, 1])
    loss = experiment.full_tar_loss(model, iter([batch] * 16), iter([batch]), 16, 0.001)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_full_tar_nonlinear_checkpoint_and_gradient_with_distinct_batches(experiment):
    torch = experiment.torch
    torch.manual_seed(23)
    model = (
        torch.nn.Sequential(
            torch.nn.Conv2d(1, 4, 3, padding=1, bias=False),
            torch.nn.GroupNorm(2, 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(4 * 3 * 3, 3),
        )
        .double()
        .eval()
    )
    batches = [
        (torch.randn(4, 1, 3, 3, dtype=torch.double), torch.randint(3, (4,)))
        for _ in range(17)
    ]

    def objective(checkpoint_steps):
        return experiment.full_tar_loss(
            model,
            iter(batches[:16]),
            iter(batches[16:]),
            16,
            0.001,
            checkpoint_steps=checkpoint_steps,
        )

    loss = objective(False)
    expected = torch.autograd.grad(loss, tuple(model.parameters()))
    checkpoint_loss = objective(True)
    actual = torch.autograd.grad(checkpoint_loss, tuple(model.parameters()))
    torch.testing.assert_close(checkpoint_loss, loss)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-9)
    direction = [torch.randn_like(p) for p in model.parameters()]
    length = experiment.norm(direction)
    direction = [d / length for d in direction]
    analytic = sum((g * d).sum() for g, d in zip(actual, direction)).item()
    center = [p.detach().clone() for p in model.parameters()]
    values = []
    for offset in [1e-6, -1e-6]:
        with torch.no_grad():
            for p, c, d in zip(model.parameters(), center, direction):
                p.copy_(c + offset * d)
        values.append(objective(True).item())
    with torch.no_grad():
        for p, c in zip(model.parameters(), center):
            p.copy_(c)
    assert analytic == pytest.approx((values[0] - values[1]) / 2e-6, abs=2e-6, rel=1e-4)
