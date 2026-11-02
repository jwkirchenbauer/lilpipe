import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("pyhessian")

DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/loss-landscapes"
)
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location(
    "hessian_density", DIRECTORY / "get_hessian_density.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7]))

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self.weight)

    def forward(self, input_ids, labels):
        terms = 1.5 * input_ids * self.weight.square()
        loss = terms[labels[:, 1:] != -100].mean()
        return SimpleNamespace(loss=loss)


@pytest.mark.parametrize("micro_batch_size", [1, 2, 8])
def test_hvp_matches_dense_hessian_and_does_not_change_checkpoint(micro_batch_size):
    model = Quadratic()
    original = model.weight.detach().clone()
    batches = [
        {
            "input_ids": torch.tensor([[-4.0, 2.0, 7.0], [-2.0, 4.0, 9.0]]),
            "labels": torch.tensor([[0, 1, 1, 1], [0, 1, -100, -100]]),
        },
        {
            "input_ids": torch.tensor([[-1.0, 2.0, 4.0]]),
            "labels": torch.tensor([[0, 1, 1, 1]]),
        },
    ]
    estimator = module.LanguageModelHessian(model, batches, micro_batch_size)
    assert estimator.params[0].device.type == "meta"
    vector = [torch.tensor([0.5, -1.0, 2.0])]
    rayleigh, product = estimator.dataloader_hv_product(vector)
    dense = torch.autograd.functional.hessian(
        lambda weight: sum(
            (1.5 * batch["input_ids"] * weight.square())[
                batch["labels"][:, 1:] != -100
            ].mean()
            for batch in batches
        )
        / len(batches),
        model.weight,
    )
    torch.testing.assert_close(product[0], dense @ vector[0])
    assert rayleigh == pytest.approx((vector[0] @ dense @ vector[0]).item())
    torch.testing.assert_close(model.weight, original)
    assert model.weight.grad is None


def test_pyhessian_density_recovers_positive_and_negative_eigenvalues():
    torch.manual_seed(42)
    estimator = module.LanguageModelHessian(
        Quadratic(),
        [
            {
                "input_ids": torch.tensor([[-3.0, 2.0, 8.0]]),
                "labels": torch.tensor([[0, 1, 1, 1]]),
            }
        ],
    )
    nodes, weights = estimator.density(iter=3, n_v=1)
    nodes = np.asarray(nodes).real
    weights = np.asarray(weights).real
    np.testing.assert_allclose(np.sort(nodes[0]), [-3.0, 2.0, 8.0], atol=1e-4)
    assert weights.sum() == pytest.approx(1, abs=1e-5)
    grid, density, bandwidth = module.smooth_density(nodes, weights)
    assert grid.min() < -3 and grid.max() > 8
    assert bandwidth > 0
    assert (density >= 0).all()
    assert np.trapezoid(density, grid) == pytest.approx(1, abs=1e-3)
