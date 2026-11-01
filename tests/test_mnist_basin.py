"""Check the basin objective's gradients, perturbation bounds, and model state."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def basin():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/scripts/analysis/toy-experiments/get_mnist_basin.py"
    )
    spec = importlib.util.spec_from_file_location("mnist_basin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_penalty_gradient_includes_both_endpoints(basin):
    torch = basin.torch
    torch.manual_seed(2)
    model = torch.nn.Linear(2, 2).double().eval()
    batch = torch.randn(5, 2, dtype=torch.double), torch.tensor([0, 1, 0, 1, 1])
    delta = {name: torch.full_like(p, 0.1) for name, p in model.named_parameters()}

    def objective():
        center = torch.nn.functional.cross_entropy(model(batch[0]), batch[1])
        return basin.basin_penalty(model, batch, center, delta, curvature=10.0)

    penalty = objective()
    assert penalty.item() > 0
    gradient = torch.autograd.grad(penalty, model.weight)[0][0, 0].item()
    original = model.weight[0, 0].item()
    values = []
    with torch.no_grad():
        for offset in (1e-5, -1e-5):
            model.weight[0, 0] = original + offset
            values.append(objective().item())
        model.weight[0, 0] = original
    assert gradient == pytest.approx((values[0] - values[1]) / 2e-5, abs=1e-8)
    assert all(value.grad is None for value in model.parameters())


def test_projected_descent_is_bounded_and_preserves_center(basin):
    torch = basin.torch
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    ).eval()
    batch = torch.randn(8, 2), torch.arange(8) % 2
    original = {name: value.clone() for name, value in model.state_dict().items()}
    torch.manual_seed(3)
    random = basin.perturbation(model, batch, 0.1, 0.01, 0)
    torch.manual_seed(3)
    attacked = basin.perturbation(model, batch, 0.1, 0.01, 3)
    assert basin.norm(random.values()).item() == pytest.approx(0.1, abs=1e-6)
    assert basin.norm(attacked.values()).item() <= 0.100001

    def loss(delta):
        point = {name: p + delta[name] for name, p in model.named_parameters()}
        logits = basin.functional_call(model, point, (batch[0],))
        return torch.nn.functional.cross_entropy(logits, batch[1]).item()

    assert loss(attacked) <= loss(random)
    assert all(not value.requires_grad for value in attacked.values())
    assert all(p.grad is None for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_evaluation_averages_over_examples(basin):
    torch = basin.torch
    model = torch.nn.Linear(2, 2).eval()
    images = torch.cat([torch.zeros(256, 2), torch.ones(1, 2)])
    labels = torch.cat(
        [torch.zeros(256, dtype=torch.long), torch.ones(1, dtype=torch.long)]
    )
    dataset = torch.utils.data.TensorDataset(images, labels)
    metrics = basin.evaluate(model, dataset, torch.device("cpu"))
    expected = torch.nn.functional.cross_entropy(model(images), labels).item()
    assert metrics["loss"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("method", ["BaSFT", "BaSFT ($\\beta=3.0$)"])
def test_training_history_records_actual_ce_not_target_objective(basin, method):
    from types import SimpleNamespace

    torch = basin.torch
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 2)
    images, labels = torch.randn(8, 2), torch.arange(8) % 2
    dataset = torch.utils.data.TensorDataset(images, labels)
    original = basin.evaluate(model, dataset, torch.device("cpu"))["loss"]
    args = SimpleNamespace(
        learning_rate=0.01,
        batch_size=8,
        radii=[0.1],
        probe_lrs=[0.01],
        steps=2,
        log_every=1,
        alpha=0.0,
        beta=0.0,
        target_loss=2.3,
        relearn_steps=2,
        relearn_log_every=1,
        relearn_learning_rate=0.01,
    )
    history = basin.fit(
        model, dataset, dataset, args, torch.device("cpu"), method, "train"
    )
    assert [row["step"] for row in history] == [0, 1, 2]
    assert all(row["loss"] == pytest.approx(original) for row in history)

    assert all(row["phase"] == "train" for row in history)
    assert all("test_accuracy" in row for row in history)
    relearn = basin.fit(
        model, dataset, dataset, args, torch.device("cpu"), method, "relearn"
    )
    assert relearn[-1]["loss"] < original
    assert all(row["phase"] == "relearn" for row in relearn)


def test_gradient_norm_is_of_full_mean_loss_and_does_not_change_weights(basin):
    torch = basin.torch
    torch.manual_seed(1)
    model = torch.nn.Linear(2, 2)
    images, labels = torch.randn(257, 2), torch.arange(257) % 2
    dataset = torch.utils.data.TensorDataset(images, labels)
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    gradient = torch.autograd.grad(loss, tuple(model.parameters()))
    expected = basin.norm(gradient).item()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(
            parameter
        )  # Diagnostics must ignore stale gradients.
    metrics = basin.evaluate(model, dataset, torch.device("cpu"), with_gradient=True)
    assert metrics["gradient_norm"] == pytest.approx(expected, rel=1e-5)
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("previous_seed", [0, 99])
@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_beta_sweep_reuses_only_compatible_completed_runs(
    basin, monkeypatch, tmp_path, previous_seed, dataset
):
    import json
    from types import SimpleNamespace

    torch = basin.torch
    args = SimpleNamespace(
        output=tmp_path,
        seed=0,
        batch_size=2,
        steps=1,
        relearn_steps=1,
        log_every=1,
        relearn_log_every=1,
        learning_rate=0.001,
        relearn_learning_rate=0.001,
        target_loss=2.3,
        alpha=1.0,
        beta=1.0,
        curvature=0.1,
        radii=[0.1],
        probe_lrs=[0.01],
        probe_steps=1,
        betas=["1.0", "3.0", "10.10"],
        resume=True,
        dataset=dataset,
    )
    config = {**vars(args), "output": str(tmp_path), "seed": previous_seed}
    config.pop("dataset")  # Older results are implicitly MNIST.
    old_rows = [
        {
            "method": method,
            "phase": phase,
            "step": 1,
            "loss": 2.0,
            "gradient_norm": 0.5,
            "test_accuracy": 0.1,
        }
        for method in ("SFT", "BaSFT")
        for phase in ("train", "relearn")
    ]
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps({"config": config, "history": old_rows}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = torch.utils.data.TensorDataset(
        torch.zeros(2, 1, 28, 28), torch.tensor([0, 1])
    )
    monkeypatch.setattr(basin, "load_data", lambda name: (data, data, data, 1))
    monkeypatch.setattr(basin.models, "resnet18", lambda **kw: torch.nn.Linear(2, 10))
    calls = []

    def fit(model, train, test, run_args, device, method, phase, train_eval=None):
        calls.append((method, phase, run_args.beta))
        return [
            {
                "method": method,
                "phase": phase,
                "step": 1,
                "loss": 2.0,
                "gradient_norm": 0.5,
                "test_accuracy": 0.1,
            }
        ]

    monkeypatch.setattr(basin, "fit", fit)
    if previous_seed != args.seed or dataset != "mnist":
        with pytest.raises(ValueError, match="different (seed|dataset)"):
            basin.main(args)
        assert not calls
        return
    basin.main(args)
    assert calls == [
        (f"BaSFT ($\\beta={label}$)", phase, beta)
        for label, beta in (("3.0", 3.0), ("10.10", 10.1))
        for phase in ("train", "relearn")
    ]
    result = json.loads(result_path.read_text())
    assert {row["method"] for row in result["history"]} == {
        "SFT",
        "BaSFT ($\\beta=1.0$)",
        "BaSFT ($\\beta=3.0$)",
        "BaSFT ($\\beta=10.10$)",
    }


def test_cifar10_has_rgb_inputs_and_deterministic_measurements(
    basin, monkeypatch, tmp_path
):
    from PIL import Image

    torch = basin.torch
    calls = []

    class CIFAR10:
        def __init__(self, root, train, download, transform=None):
            calls.append((root, train, download))
            self.transform = transform
            self.data = (
                torch.arange(2 * 32 * 32 * 3)
                .remainder(256)
                .byte()
                .reshape(2, 32, 32, 3)
                .numpy()
            )
            self.targets = [0, 1]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            image = Image.fromarray(self.data[index])
            return self.transform(image) if self.transform else image, self.targets[
                index
            ]

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(basin.datasets, "CIFAR10", CIFAR10)
    train, train_eval, test, channels = basin.load_data("cifar10")
    assert channels == 3
    assert calls == [
        (tmp_path / "cifar10", True, True),
        (tmp_path / "cifar10", False, True),
    ]
    assert train[0][0].shape == (3, 32, 32)
    expected = torch.as_tensor(train.data[0]).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(train_eval[0][0], expected)
    torch.testing.assert_close(test[0][0], expected)
    assert 0 <= train[0][0].min() and train[0][0].max() <= 1
    assert train_eval[0][1].item() == 0
