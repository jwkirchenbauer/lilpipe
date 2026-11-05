"""Parameter movement is a global vector norm, aligned by parameter name."""

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/parameter-norms/get_param_norm.py"
)
spec = importlib.util.spec_from_file_location("parameter_norms", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def small_model(weights):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(len(weights), 1, bias=False)]
    )
    with torch.no_grad():
        model.model.layers[0].weight.copy_(torch.tensor([weights]))
    return model


def test_delta_norm_matches_flattened_parameters():
    original = small_model([3.0, 4.0])
    defended = small_model([4.0, 8.0])
    original.register_buffer("rotary_frequencies", torch.tensor([1e6]))
    defended.register_buffer("rotary_frequencies", torch.tensor([0.0]))
    result = module.parameter_norms(defended, original)
    assert result["num_parameters"] == 2
    assert result["reference_l2_norm"] == 5
    assert result["l2_norm"] == pytest.approx(math.sqrt(80))
    assert result["delta_l2_norm"] == pytest.approx(math.sqrt(17))
    assert result["relative_delta_l2_norm"] == pytest.approx(math.sqrt(17) / 5)
    assert result["layers"][0]["delta_l2_norm"] == result["delta_l2_norm"]


def test_identical_models_have_zero_delta_and_single_model_still_works():
    model = small_model([3.0, 4.0])
    assert module.parameter_norms(model, model)["delta_l2_norm"] == 0
    result = module.parameter_norms(model)
    assert result["l2_norm"] == 5
    assert "delta_l2_norm" not in result


def test_subtraction_preserves_small_bfloat16_changes():
    original = small_model([1.0, 2.0]).to(torch.bfloat16)
    defended = small_model([1.0078125, 2.015625]).to(torch.bfloat16)
    result = module.parameter_norms(defended, original)
    assert result["delta_l2_norm"] == pytest.approx(
        math.sqrt(0.0078125**2 + 0.015625**2)
    )


def test_parameter_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape differs"):
        module.parameter_norms(small_model([1.0, 2.0]), small_model([1.0]))
    model = small_model([1.0, 2.0])
    model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="names differ"):
        module.parameter_norms(model, small_model([1.0, 2.0]))
