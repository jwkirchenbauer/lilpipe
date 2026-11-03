"""Regression tests for independent layer perturbations."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/layer-sensitivity/evaluate_noise.py"
)
spec = importlib.util.spec_from_file_location("evaluate_noise", SCRIPT)
noise = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise)


def test_perturb_restores_weights_after_failure_and_leaves_other_layers_untouched():
    selected = torch.nn.Parameter(torch.ones(128, dtype=torch.bfloat16))
    other = torch.nn.Parameter(torch.full((128,), 2.0))
    original = selected.detach().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with noise.perturb([selected], 0.3, 42, "relative") as stats:
            assert not torch.equal(selected, original)
            assert torch.equal(other, torch.full((128,), 2.0))
            assert stats["weight_rms"] == 1
            assert stats["noise_std"] == 0.3
            assert stats["realized_noise_rms"] > 0
            raise RuntimeError("evaluation failed")
    assert torch.equal(selected, original)


def test_scales_share_a_direction_and_seeds_change_it():
    parameter = torch.nn.Parameter(torch.zeros(256))
    with noise.perturb([parameter], 0.1, 42, "absolute"):
        small = parameter.detach().clone()
    with noise.perturb([parameter], 0.2, 42, "absolute"):
        assert torch.equal(parameter, small * 2)
    with noise.perturb([parameter], 0.1, 43, "absolute"):
        assert not torch.equal(parameter, small)
    assert torch.count_nonzero(parameter) == 0


def test_relative_noise_uses_group_rms():
    parameter = torch.nn.Parameter(torch.full((100,), 2.0))
    with noise.perturb([parameter], 0.1, 42, "relative") as stats:
        assert stats["noise_std"] == pytest.approx(0.2)
        assert stats["num_parameters"] == 100
        delta = parameter.detach() - 2
        assert stats["realized_noise_l2"] == pytest.approx(delta.norm().item())


def test_groups_cover_each_parameter_once_and_reject_tied_head():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4)
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
            self.model.norm = torch.nn.LayerNorm(4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)

    model = Model()
    assert list(noise.parameter_groups(model)) == [
        "embedding",
        "layer-0",
        "final-norm",
        "lm-head",
    ]
    model.lm_head.weight = model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="Shared weights"):
        noise.parameter_groups(model)


def test_random_subset_is_fixed_random_and_covers_every_subject():
    sizes = {"mmlu_b": 100, "mmlu_a": 150}
    selected = noise.random_subset(sizes, 20, 0)
    assert selected == noise.random_subset(dict(reversed(list(sizes.items()))), 20, 0)
    assert selected != noise.random_subset(sizes, 20, 1)
    for name, indices in selected.items():
        assert len(indices) == len(set(indices)) == 20
        assert all(0 <= i < sizes[name] for i in indices)
        assert indices != list(range(20))
    assert noise.random_subset({"tiny": 2}, 20, 0) == {"tiny": [0, 1]}
