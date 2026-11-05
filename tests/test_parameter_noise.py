"""Global Gaussian radii share one direction and retain original model weights."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/analysis/parameter-noise/perturb_model.py"
)
spec = importlib.util.spec_from_file_location("parameter_noise", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def flatten(model):
    return torch.cat([p.detach().double().flatten() for p in model.parameters()])


def small_model(dtype=torch.float64):
    model = torch.nn.Linear(3, 2).to(dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0, 1.0], [-1.0, 2.0, 5.0]]))
        model.bias.copy_(torch.tensor([100.0, 200.0]))
    model.register_buffer("not_a_parameter", torch.tensor([1e9]))
    return model


def test_global_radius_and_shared_direction():
    model = small_model()
    original = flatten(model).clone()
    statistics = module.noise_statistics(model, 42)
    assert statistics["num_parameters"] == original.numel()
    assert statistics["model_l2_norm"] == pytest.approx(original.norm().item())
    direction = None
    for fraction in (0.1, 0.25, 0.5, 0.75):
        changed = copy.deepcopy(model)
        values = module.perturb(changed, fraction, 42, statistics)
        delta = flatten(changed) - original
        assert delta.norm().item() == pytest.approx(fraction * original.norm().item())
        assert values["realized_fraction"] == pytest.approx(fraction)
        if direction is not None:
            torch.testing.assert_close(delta / fraction, direction)
        direction = delta / fraction
    torch.testing.assert_close(flatten(model), original, rtol=0, atol=0)


def test_bfloat16_reports_realized_delta_and_seed_is_reproducible():
    model = small_model(torch.bfloat16)
    original = flatten(model)
    statistics = module.noise_statistics(model, 42)
    first, second = copy.deepcopy(model), copy.deepcopy(model)
    values = module.perturb(first, 0.1, 42, statistics)
    module.perturb(second, 0.1, 42, statistics)
    torch.testing.assert_close(flatten(first), flatten(second), rtol=0, atol=0)
    assert values["realized_delta_l2_norm"] == pytest.approx(
        (flatten(first) - original).norm().item()
    )
    other = copy.deepcopy(model)
    module.perturb(other, 0.1, 43, module.noise_statistics(model, 43))
    assert not torch.equal(flatten(first), flatten(other))


@pytest.mark.parametrize("fraction", [0, -0.1, float("nan"), float("inf")])
def test_invalid_fraction_is_rejected(fraction):
    model = small_model()
    with pytest.raises(ValueError, match="finite and positive"):
        module.perturb(model, fraction, 42, module.noise_statistics(model, 42))


def test_cli_saves_reloadable_models_with_measured_norms(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    config = transformers.LlamaConfig(
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    baseline = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    source = tmp_path / "base"
    baseline.save_pretrained(source)
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(source)
    prefix = tmp_path / "noise"
    metadata = tmp_path / "run/perturbations.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--model",
            str(source),
            "--output-prefix",
            str(prefix),
            "--output",
            str(metadata),
            "--percentages",
            "0.1",
            "0.5",
        ],
    )
    module.main()
    report = json.loads(metadata.read_text())
    assert report["complete"]
    assert report["shared_direction_across_fractions"]
    assert report["model_l2_norm"] == pytest.approx(flatten(baseline).norm().item())
    assert report["percentages"] == [0.1, 0.5]
    assert [entry["fraction"] for entry in report["models"]] == [0.001, 0.005]
    for entry in report["models"]:
        assert entry["target_delta_l2_norm"] == pytest.approx(
            report["model_l2_norm"] * entry["percentage"] / 100
        )
        loaded = transformers.AutoModelForCausalLM.from_pretrained(entry["model"])
        transformers.AutoTokenizer.from_pretrained(entry["model"])
        measured = (flatten(loaded) - flatten(baseline)).norm().item()
        assert entry["realized_delta_l2_norm"] == pytest.approx(measured)
    with pytest.raises(SystemExit):
        module.main()
