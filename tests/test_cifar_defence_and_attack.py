"""Check defense loss signs and first-order SAM/TAR gradients."""

import copy
import importlib
import math
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("exact", [False, True])
def test_cosine_only_update_matches_exact_derivative_without_ce(experiment, exact):
    torch = experiment.torch
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    ).double()
    images = torch.randn(8, 3, dtype=torch.double)
    labels = torch.arange(8) % 2
    parameters = tuple(model.parameters())
    original = [p.detach().clone() for p in parameters]
    torch.manual_seed(17)
    a, b = torch.randperm(8).chunk(2)
    gradients = [
        torch.autograd.grad(
            torch.nn.functional.cross_entropy(model(images[group]), labels[group]),
            parameters,
            create_graph=True,
        )
        for group in (a, b)
    ]
    ga, gb = gradients
    dot = experiment.gradient_dot
    cosine = dot(ga, gb) / ((dot(ga, ga) + 0.01).sqrt() * (dot(gb, gb) + 0.01).sqrt())
    expected = torch.autograd.grad(cosine, parameters)
    for p in parameters:
        p.grad = torch.ones_like(p)
    torch.manual_seed(17)
    backward = experiment.exact_cosine_backward if exact else experiment.cosine_backward
    options = {} if exact else {"rho": 1e-6}
    metrics = backward(
        model,
        images,
        labels,
        tau=0.1,
        skip_threshold=0,
        **options,
    )
    assert metrics["group_cosine"] == pytest.approx(cosine.item())
    for p, g, center in zip(parameters, expected, original):
        torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(p, center, rtol=0, atol=0)


@pytest.mark.parametrize("exact", [False, True])
def test_cosine_only_skips_without_adding_training_gradient(experiment, exact):
    torch = experiment.torch
    model = torch.nn.Linear(3, 2).double()
    images = torch.randn(4, 3, dtype=torch.double)
    labels = torch.arange(4) % 2
    backward = experiment.exact_cosine_backward if exact else experiment.cosine_backward
    options = {} if exact else {"rho": 1e-3}
    metrics = backward(
        model,
        images,
        labels,
        tau=0.1,
        skip_threshold=1e10,
        **options,
    )
    assert metrics["skipped"]
    assert metrics["regularization_gradient_norm"] == 0
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    with pytest.raises(ValueError, match="even batch"):
        backward(
            model,
            images[:3],
            labels[:3],
            tau=0.1,
            skip_threshold=0,
            **options,
        )


def test_exact_group_cosine_matches_scalar_finite_differences(experiment):
    torch = experiment.torch
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
    ).double()
    images = torch.randn(6, 3, dtype=torch.double)
    labels = torch.arange(6) % 2
    parameters = tuple(model.parameters())
    center = [p.detach().clone() for p in parameters]
    direction = [torch.randn_like(p) for p in parameters]
    torch.manual_seed(12)
    experiment.exact_cosine_backward(model, images, labels, tau=0.2)
    predicted = sum((p.grad * d).sum() for p, d in zip(parameters, direction)).item()

    def value():
        torch.manual_seed(12)
        groups = torch.randperm(6).chunk(2)
        gradients = [
            torch.autograd.grad(
                torch.nn.functional.cross_entropy(model(images[g]), labels[g]),
                parameters,
            )
            for g in groups
        ]
        a, b = [torch.cat([g.flatten() for g in group]) for group in gradients]
        return (
            a
            @ b
            / ((a.square().sum() + 0.04).sqrt() * (b.square().sum() + 0.04).sqrt())
        ).item()

    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                for p, c, d in zip(parameters, center, direction):
                    p.copy_(c + sign * 1e-5 * d)
            values.append(value())
    finally:
        with torch.no_grad():
            for p, c in zip(parameters, center):
                p.copy_(c)
    assert predicted == pytest.approx(
        (values[0] - values[1]) / 2e-5, rel=1e-6, abs=1e-7
    )


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


def test_streaming_cosine_diagnostics_match_explicit_gram_and_preserve_grads(
    experiment,
):
    torch = experiment.torch
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 2).double().eval()
    images = torch.randn(6, 3, dtype=torch.double)
    labels = torch.arange(6) % 2
    rows = []
    for i in range(6):
        loss = torch.nn.functional.cross_entropy(
            model(images[i : i + 1]), labels[i : i + 1]
        )
        rows.append(
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(loss, tuple(model.parameters()))
                ]
            )
        )
    gradients = torch.stack(rows)
    stabilized = (gradients.square().sum(1) + 0.2**2).sqrt()
    gram = gradients @ gradients.T / (stabilized[:, None] * stabilized[None, :])
    expected = gram[~torch.eye(6, dtype=torch.bool)].mean().item()
    saved = []
    for p in model.parameters():
        p.grad = torch.randn_like(p)
        saved.append(p.grad.clone())
    result = experiment.cosine_diagnostics(model, images, labels, tau=0.2)
    assert result["individual_pairwise_cosine"] == pytest.approx(expected, abs=1e-12)
    ga, gb = gradients[:3].mean(0), gradients[3:].mean(0)
    expected_group = (
        ga
        @ gb
        / ((ga.square().sum() + 0.04).sqrt() * (gb.square().sum() + 0.04).sqrt())
    )
    assert result["diagnostic_group_cosine"] == pytest.approx(expected_group.item())
    for p, g in zip(model.parameters(), saved):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)


def test_full_individual_finite_differences_match_stabilized_exact_objective(
    experiment,
):
    torch = experiment.torch
    torch.manual_seed(9)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        )
        .double()
        .eval()
    )
    images, labels = torch.randn(4, 3, dtype=torch.double), torch.arange(4) % 2
    parameters = tuple(model.parameters())
    center = [p.detach().clone() for p in parameters]
    rows = []
    for i in range(4):
        loss = torch.nn.functional.cross_entropy(
            model(images[i : i + 1]), labels[i : i + 1]
        )
        gs = torch.autograd.grad(loss, parameters, create_graph=True)
        rows.append(torch.cat([g.flatten() for g in gs]))
    gradients = torch.stack(rows)
    u = gradients / (gradients.square().sum(1, keepdim=True) + 0.2**2).sqrt()
    gram = u @ u.T
    objective = gram[~torch.eye(4, dtype=torch.bool)].mean()
    expected = torch.autograd.grad(objective, parameters)
    errors = []
    for rho in (1e-2, 1e-4, 1e-6):
        metrics = experiment.individual_cosine_backward(
            model, images, labels, method="full-individual", tau=0.2, rho=rho
        )
        assert metrics["objective"] == pytest.approx(objective.item())
        errors.append(
            sum((p.grad - g).square().sum() for p, g in zip(parameters, expected))
            .sqrt()
            .item()
        )
    assert errors[-1] < 2e-5
    assert errors[-1] < errors[0] / 100
    for p, c in zip(parameters, center):
        torch.testing.assert_close(p, c, rtol=0, atol=0)


def test_frozen_normalization_matches_detached_weight_objective(experiment):
    torch = experiment.torch
    torch.manual_seed(8)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        )
        .double()
        .eval()
    )
    images, labels = torch.randn(4, 3, dtype=torch.double), torch.arange(4) % 2
    parameters = tuple(model.parameters())
    weights = []
    for i in range(4):
        gs = torch.autograd.grad(
            torch.nn.functional.cross_entropy(
                model(images[i : i + 1]), labels[i : i + 1]
            ),
            parameters,
        )
        weights.append((experiment.gradient_dot(gs, gs) + 0.2**2).rsqrt())
    weights = torch.stack(weights).detach()
    torch.manual_seed(12)
    a, b = torch.randperm(4).chunk(2)
    groups = []
    for ids in (a, b):
        loss = (
            torch.nn.functional.cross_entropy(
                model(images[ids]), labels[ids], reduction="none"
            )
            * weights[ids]
        ).mean()
        groups.append(torch.autograd.grad(loss, parameters, create_graph=True))
    objective = experiment.gradient_dot(*groups)
    expected = torch.autograd.grad(objective, parameters)
    torch.manual_seed(12)
    metrics = experiment.individual_cosine_backward(
        model, images, labels, method="frozen-normalization", tau=0.2, rho=1e-6
    )
    assert metrics["objective"] == pytest.approx(objective.item())
    for p, g in zip(parameters, expected):
        torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=2e-6)


def test_sampled_contributions_average_to_full_estimate(experiment, monkeypatch):
    torch = experiment.torch
    torch.manual_seed(11)
    model = torch.nn.Linear(3, 2).double().eval()
    images, labels = torch.randn(4, 3, dtype=torch.double), torch.arange(4) % 2
    experiment.individual_cosine_backward(
        model, images, labels, method="full-individual", tau=0.2, rho=1e-5
    )
    expected = [p.grad.clone() for p in model.parameters()]
    average = [torch.zeros_like(p) for p in model.parameters()]
    for index in range(4):
        monkeypatch.setattr(
            torch,
            "randperm",
            lambda n: torch.tensor([index] + [j for j in range(n) if j != index]),
        )
        experiment.individual_cosine_backward(
            model,
            images,
            labels,
            method="sampled-individual",
            sample_size=1,
            tau=0.2,
            rho=1e-5,
        )
        for total, p in zip(average, model.parameters()):
            total.add_(p.grad / 4)
    for actual, target in zip(average, expected):
        torch.testing.assert_close(actual, target, rtol=1e-10, atol=1e-10)


def test_individual_perturbations_replay_rng_and_preserve_buffers(experiment):
    torch = experiment.torch

    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 2)
            self.register_buffer("counter", torch.zeros(()))
            self.noise = []

        def forward(self, x):
            self.counter.add_(1)
            noise = torch.rand_like(x) + 0.5
            self.noise.append(noise.clone())
            return super().forward(x * noise)

    torch.manual_seed(4)
    model = Model().double().eval()
    images, labels = torch.randn(4, 3, dtype=torch.double), torch.arange(4) % 2
    center = copy.deepcopy(model.state_dict())
    experiment.individual_cosine_backward(
        model, images, labels, method="full-individual", tau=0.2, rho=1e-5
    )
    assert len(model.noise) == 12
    for i in range(4):
        torch.testing.assert_close(
            model.noise[i], model.noise[4 + 2 * i], rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.noise[i], model.noise[5 + 2 * i], rtol=0, atol=0
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, center[name], rtol=0, atol=0)


def test_individual_perturbation_exception_leaves_original_state(experiment):
    torch = experiment.torch

    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(3, 2)
            self.register_buffer("counter", torch.zeros(()))
            self.calls = 0

        def forward(self, x):
            self.counter.add_(1)
            self.calls += 1
            if self.calls == 6:
                raise RuntimeError("perturbed forward failed")
            return super().forward(x)

    model = Model().double().eval()
    images, labels = torch.randn(4, 3, dtype=torch.double), torch.arange(4) % 2
    state = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    with pytest.raises(RuntimeError, match="perturbed forward failed"):
        experiment.individual_cosine_backward(
            model, images, labels, method="full-individual", tau=0.2, rho=1e-5
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


def test_positive_pair_penalty_matches_full_exact_objective(experiment):
    torch = experiment.torch
    torch.manual_seed(3)
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)
        )
        .double()
        .eval()
    )
    images = torch.randn(4, 3, dtype=torch.double)
    labels = torch.tensor([0, 0, 1, 1])
    parameters = tuple(model.parameters())
    center = [p.detach().clone() for p in parameters]
    rows = []
    for i in range(4):
        loss = torch.nn.functional.cross_entropy(
            model(images[i : i + 1]), labels[i : i + 1]
        )
        rows.append(
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(loss, parameters, create_graph=True)
                ]
            )
        )
    gradients = torch.stack(rows)
    u = gradients / (gradients.square().sum(1, keepdim=True) + 0.1**2).sqrt()
    cosine = u @ u.T
    objective = cosine[~torch.eye(4, dtype=torch.bool)].relu().mean()
    expected = torch.autograd.grad(objective, parameters)
    pairs = torch.tensor([(i, j) for i in range(4) for j in range(4) if i != j])
    metrics = experiment.positive_pair_cosine_backward(
        model, images, labels, tau=0.1, pairs=pairs
    )
    assert 0 < metrics["positive_pairs"] < len(pairs)
    assert metrics["objective"] == pytest.approx(objective.item(), abs=1e-12)
    for p, g, before in zip(parameters, expected, center):
        torch.testing.assert_close(p.grad, g, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(p, before, rtol=0, atol=0)


def test_positive_pair_penalty_skips_opposing_pairs(experiment):
    torch = experiment.torch
    model = torch.nn.Linear(2, 2).double().eval()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    images = torch.ones(2, 2, dtype=torch.double)
    labels = torch.tensor([0, 1])
    metrics = experiment.positive_pair_cosine_backward(
        model, images, labels, tau=0.1, pairs=torch.tensor([[0, 1], [1, 0]])
    )
    assert metrics["objective"] == 0 and metrics["skipped"]
    assert all(torch.count_nonzero(p.grad) == 0 for p in model.parameters())


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


@pytest.mark.parametrize(
    "method",
    [
        "gradient-cosine",
        "gradient-cosine-exact",
        "frozen-normalization",
        "sampled-individual",
        "full-individual",
        "positive-pair-cosine",
    ],
)
def test_regularizer_only_defense_and_relearning_integration(
    experiment, monkeypatch, tmp_path, method
):
    torch = experiment.torch

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.head = torch.nn.Linear(64, 10)

        def forward(self, images):
            return self.head(self.conv1(images).tanh().mean((2, 3)))

    dataset = torch.utils.data.TensorDataset(
        torch.randn(8, 3, 4, 4), torch.arange(8) % 10
    )
    monkeypatch.setattr(experiment, "load_data", lambda: (dataset, dataset, dataset, 3))
    monkeypatch.setattr(experiment.models, "resnet18", lambda **kwargs: SmallModel())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        method=method,
        output=str(tmp_path),
        train_samples=4,
        batch_size=4,
        seed=0,
        checkpoint=None,
        defend_checkpoint=None,
        steps=2,
        learning_rate=0.001,
        log_every=1,
        cosine_tau=0.1,
        rho=1e-3,
        cosine_skip_threshold=0,
        cosine_contributions=2,
        calibrate_constant_head=False,
        optimizer="adam",
        relearn_learning_rate=0.001,
        relearn_steps=1,
        cosine_diagnostic_samples=4,
        cosine_diagnostic_every=1,
    )
    experiment.main(args)
    result = experiment.json.loads((tmp_path / "results.json").read_text())
    defense = [r for r in result["history"] if r["phase"] == "defense"]
    attack = [r for r in result["history"] if r["phase"] == "relearn"]
    assert len(defense) == len(attack) == 2
    assert defense[0]["gradient_seconds"] > 0
    assert all("individual_pairwise_cosine" in r for r in attack)
    assert (tmp_path / "train.pt").exists() and (tmp_path / "relearn.pt").exists()


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
