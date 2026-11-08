"""Finite-difference cosine agrees with independent higher-order derivatives."""

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "examples/tamper-resistance/src"
sys.path.insert(0, str(SRC))
from trainers.tr_gip import finite_difference_cosine, gradient_dot


def problem(dtype=torch.float64):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).to(dtype)
    x_a, x_b = [torch.randn(3, 3, dtype=dtype) for _ in range(2)]
    y_a, y_b = [torch.randn(3, 2, dtype=dtype) for _ in range(2)]
    return (
        model,
        lambda: (model(x_a) - y_a).square().mean(),
        lambda: (model(x_b) - y_b).square().mean(),
    )


def exact_cosine_gradient(model, loss_a, loss_b, tau, squared=False):
    parameters = tuple(model.parameters())
    a = torch.autograd.grad(loss_a(), parameters, create_graph=True)
    b = torch.autograd.grad(loss_b(), parameters, create_graph=True)
    cosine = gradient_dot(a, b) / (
        (gradient_dot(a, a) + tau**2).sqrt() * (gradient_dot(b, b) + tau**2).sqrt()
    )
    return torch.autograd.grad(cosine.square() if squared else cosine, parameters)


@pytest.mark.parametrize("squared", [False, True])
@pytest.mark.parametrize("sign", [1, -1])
def test_fp64_rho_sweep_matches_exact_autograd(squared, sign):
    model, loss_a, loss_b = problem()
    original_loss_b = loss_b
    loss_b = lambda: sign * original_loss_b()
    expected = exact_cosine_gradient(model, loss_a, loss_b, tau=0.1, squared=squared)
    errors = []
    for rho in (1e-2, 1e-3, 1e-4, 1e-5):
        result = finite_difference_cosine(
            model, loss_a, loss_b, tau=0.1, rho=rho, squared=squared
        )
        errors.append(
            torch.cat(
                [(a - b).flatten() for a, b in zip(result.regularization, expected)]
            )
            .norm()
            .item()
        )
        assert all(not r.requires_grad for r in result.regularization)
    assert errors[-1] < errors[0] / 100
    assert errors[-1] < 1e-4


def test_parameters_buffers_and_existing_grads_restored_on_exception():
    model, loss_a, loss_b = problem()
    model.register_buffer("counter", torch.tensor(0))
    initial = copy.deepcopy(model.state_dict())
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    grads = [p.grad.clone() for p in model.parameters()]
    calls = 0

    def failing_loss():
        nonlocal calls
        calls += 1
        model.counter.add_(1)
        if calls == 2:
            raise RuntimeError("perturbation failure")
        return loss_a()

    with pytest.raises(RuntimeError, match="perturbation failure"):
        finite_difference_cosine(model, failing_loss, loss_b, tau=0.1, rho=1e-3)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
    for p, g in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)


def test_dropout_replayed_and_buffers_restored():
    torch.manual_seed(8)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.dropout = nn.Dropout(0.4)
            self.register_buffer("counter", torch.tensor(0))
            self.masks = []

        def forward(self, x):
            self.counter.add_(1)
            y = self.linear(x)
            mask = self.dropout(torch.ones_like(y))
            self.masks.append(mask.clone())
            return y * mask

    model = Model()
    x = torch.randn(4, 3)
    initial = copy.deepcopy(model.state_dict())
    finite_difference_cosine(
        model,
        lambda: model(x).square().mean(),
        lambda: (model(x) - 1).square().mean(),
        tau=0.1,
        rho=1e-3,
    )
    assert len(model.masks) == 4
    torch.testing.assert_close(model.masks[0], model.masks[2], rtol=0, atol=0)
    torch.testing.assert_close(model.masks[1], model.masks[3], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_skip_small_norm_retains_base_gradient():
    model, loss_a, loss_b = problem()
    result = finite_difference_cosine(
        model, loss_a, loss_b, tau=0.1, rho=1e-3, skip_threshold=1e6
    )
    expected = torch.autograd.grad((loss_a() + loss_b()) / 2, tuple(model.parameters()))
    assert result.skipped
    for a, b, r, g in zip(
        result.gradient_a, result.gradient_b, result.regularization, expected
    ):
        torch.testing.assert_close((a + b) / 2, g)
        assert torch.count_nonzero(r) == 0


def test_zero_directions_do_not_evaluate_perturbations():
    model = nn.Linear(2, 1, bias=False).double()
    calls = 0

    def loss():
        nonlocal calls
        calls += 1
        return model.weight.sum() * 0

    result = finite_difference_cosine(model, loss, loss, tau=0.1, rho=1e-3)
    assert calls == 2
    assert all(torch.count_nonzero(r) == 0 for r in result.regularization)


@pytest.mark.parametrize("coefficient", [0.0, 0.1])
def test_defense_update_matches_retain_plus_exact_cosine(tmp_path, coefficient):
    pytest.importorskip("axolotl")
    from axolotl.core.training_args import AxolotlTrainingArguments
    from trainers.tr_gip import GIPTrainer
    from trainers.tr_gip import causal_lm_losses
    from types import SimpleNamespace
    from torch.nn import functional as F

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 3)
            self.hidden = nn.Linear(3, 4)
            self.head = nn.Linear(4, 7)

        def forward(self, input_ids, labels):
            logits = self.head(self.hidden(self.embedding(input_ids)).tanh())
            loss = F.cross_entropy(
                logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100
            )
            return SimpleNamespace(logits=logits, loss=loss)

    torch.manual_seed(3)
    model = TinyLM()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4], [2, 4, 1, 5], [3, 2, 5, 6], [1, 5, 4, 3]])
    labels = ids.clone()
    labels[2, 1] = -100
    batch = {
        "input_ids": ids,
        "labels": labels,
        "seam_role": torch.tensor([0, 0, 1, 1]),
    }
    parameters = tuple(reference.parameters())
    retain = reference(ids[2:], labels[2:]).loss
    objective = retain
    if coefficient:
        la = causal_lm_losses(reference(ids[:1], labels[:1]).logits, labels[:1]).mean()
        lb = causal_lm_losses(
            reference(ids[1:2], labels[1:2]).logits, labels[1:2]
        ).mean()
        ga = torch.autograd.grad(la, parameters, create_graph=True)
        gb = torch.autograd.grad(lb, parameters, create_graph=True)
        cosine = gradient_dot(ga, gb) / (
            (gradient_dot(ga, ga) + 0.1**2).sqrt()
            * (gradient_dot(gb, gb) + 0.1**2).sqrt()
        )
        objective = objective + coefficient * cosine
    expected = torch.autograd.grad(objective, parameters)
    trainer = GIPTrainer(
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            remove_unused_columns=False,
        ),
        cosine_coefficient=coefficient,
        cosine_tau=0.1,
        cosine_rho=1e-3,
    )
    trainer.current_gradient_accumulation_steps = 1
    before = [p.detach().clone() for p in model.parameters()]
    trainer.training_step(model, dict(batch))
    for p, original, gradient in zip(model.parameters(), before, expected):
        torch.testing.assert_close(p, original, rtol=0, atol=0)
        torch.testing.assert_close(
            p.grad, gradient, rtol=3e-3, atol=3e-4 if coefficient else 1e-7
        )
    assert any("loss_retain" in row for row in trainer.state.log_history)


def test_defense_sampler_balances_roles_and_drops_incomplete_batches():
    pytest.importorskip("axolotl")
    from trainers.tr_gip import RefusalHarmfulSampler

    dataset = {"seam_role": [0] * 7 + [1] * 9}
    sampler = RefusalHarmfulSampler(dataset, batch_size=8, seed=42)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 8
    assert len(set(indices)) == 8
    assert [dataset["seam_role"][i] for i in indices] == [0] * 4 + [1] * 4
    with pytest.raises(ValueError, match="divisible by four"):
        RefusalHarmfulSampler(dataset, batch_size=6, seed=42)


def test_axolotl_defense_training_logs_metrics(tmp_path):
    pytest.importorskip("axolotl")
    from axolotl.core.training_args import AxolotlTrainingArguments
    from transformers import GPT2Config, GPT2LMHeadModel
    from datasets import Dataset
    from trainers.tr_gip import GIPTrainer

    config = GPT2Config(
        vocab_size=11,
        n_positions=8,
        n_embd=8,
        n_layer=1,
        n_head=2,
        use_cache=False,
        bos_token_id=0,
        eos_token_id=0,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    config._attn_implementation = "eager"
    rows = [
        {"input_ids": [1, 2 + i, 3, 4], "labels": [1, 2 + i, 3, 4], "seam_role": i % 2}
        for i in range(4)
    ]
    trainer = GIPTrainer(
        model=GPT2LMHeadModel(config),
        train_dataset=Dataset.from_list(rows),
        cosine_tau=0.1,
        cosine_rho=1e-3,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=4,
            max_steps=2,
            report_to=[],
            save_strategy="no",
            remove_unused_columns=False,
            disable_tqdm=True,
            sample_packing=False,
        ),
    )
    trainer.train()
    metrics = [row for row in trainer.state.log_history if "gradient_cosine" in row]
    assert len(metrics) == 2
    for row in metrics:
        assert -1 <= row["gradient_cosine"] <= 1
        assert row["regularization_gradient_norm"] >= 0
        assert row["retain_gradient_norm"] > 0
        assert row["regularization_strength_ratio"] >= 0
        assert "regularization_skip_frequency" in row
    assert all(p.dtype == torch.float32 for p in trainer.model.parameters())


@pytest.mark.parametrize("role,expected", [("hh", 0), ("hb", 1)])
def test_paired_completion_data_can_include_roles(role, expected):
    pytest.importorskip("axolotl")
    from trainers.tr_gip import HARMFUL, REFUSAL
    from data.beavertails_hh import load
    from types import SimpleNamespace

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            return {"input_ids": [3] * len(text.split())}

    cfg = SimpleNamespace(sequence_len=128, train_on_inputs=False)
    ds = SimpleNamespace(
        message_property_mappings={
            "role": role,
            "max_rows": "1",
            "include_role": "true",
        }
    )
    result = load(Tokenizer(), cfg, ds).tokenize_row(
        {
            "prompt": "unsafe prompt",
            "response": "unsafe response",
            "refusal": "Question: unsafe prompt\nAnswer: safe response",
        }
    )
    assert result["seam_role"] == expected
    assert -100 in result["labels"] and result["labels"][-1] == 2


def test_strength_sweep_generates_matched_attacks(tmp_path, monkeypatch):
    import runpy
    import yaml

    example = SRC.parent
    monkeypatch.chdir(example)
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_gradient_cosine_attacks.py",
            "configs/stages-beavertails/0_alignment/qwen3-0.6b-beavertail--align-fft-gradient-cosine/sweep.yml",
            str(output),
        ],
    )
    namespace = runpy.run_path("scripts/misc/generate_gradient_cosine_attacks.py")
    namespace["main"]()
    manifest = yaml.safe_load((output / "experiment.yml").read_text())
    models = yaml.safe_load((output / "models.yml").read_text())["models"]
    assert len(models) == 4 and len(manifest["models"]) == 2
    coefficients = []
    for attack_id in manifest["models"]:
        attack = models[attack_id]
        (defense_id,) = attack["producer"]["depends_on"]
        defense = models[defense_id]
        attack_config = yaml.safe_load(Path(attack["producer"]["args"][0]).read_text())
        defense_config = yaml.safe_load(
            Path(defense["producer"]["args"][0]).read_text()
        )
        coefficients.append(defense_config["custom_trainer_args"]["cosine_coefficient"])
        assert attack_config["base_model"] == defense["base_model_name_or_path"]
        assert attack_config["max_steps"] == 250 and attack_config["save_steps"] == 50
        assert attack_config["trainer_cls"] == "trainers.tr_gip.GIPRelearnTrainer"
    assert sorted(coefficients) == [0.1, 1.0]


@pytest.mark.parametrize("fail", [False, True])
def test_early_eval_preserves_training_state(tmp_path, monkeypatch, fail):
    pytest.importorskip("axolotl")
    import runpy
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(SRC.parent))
    ns = runpy.run_path(str(SRC.parent / "scripts/misc/early_harmfulness.py"))
    callback = ns["EarlyHarmfulnessCallback"].__new__(ns["EarlyHarmfulnessCallback"])
    model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.2))
    model.register_buffer("counter", torch.tensor(0))
    model.config = SimpleNamespace(use_cache=False)
    model.device = torch.device("cpu")
    model.train()
    model[1].eval()
    modes = [m.training for m in model.modules()]
    callback.trainer = SimpleNamespace(
        model=model, args=SimpleNamespace(bf16=False), log=lambda _: None
    )
    callback.tokenizer = object()
    callback.prompts = ["prompt"]
    callback.output = str(tmp_path)

    def generate(model, tokenizer, prompts):
        model.counter.add_(1)
        torch.rand(10)
        return ["completion"]

    def classify(prompts, generations):
        torch.rand(10)
        if fail:
            raise RuntimeError("classifier failed")
        return 0.5, [{"prompt": "prompt", "harmful_score": 0.5}]

    monkeypatch.setitem(callback.evaluate.__globals__, "generate", generate)
    monkeypatch.setitem(callback.evaluate.__globals__, "harmfulness", classify)
    rng = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="classifier failed"):
            callback.evaluate(SimpleNamespace(global_step=5))
    else:
        callback.evaluate(SimpleNamespace(global_step=5))
        assert (tmp_path / "checkpoint-5/summary.json").exists()
    assert model.counter.item() == 0
    assert not model.config.use_cache
    assert [m.training for m in model.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("offload", [False, True])
def test_squared_fp32_rho_sweep(offload):
    model, loss_a, loss_b = problem(torch.float32)
    expected = exact_cosine_gradient(model, loss_a, loss_b, tau=0.1, squared=True)
    norm = gradient_dot(expected, expected).sqrt().item()
    errors = []
    for rho in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        result = finite_difference_cosine(
            model,
            loss_a,
            loss_b,
            tau=0.1,
            rho=rho,
            squared=True,
            offload_originals=offload,
        )
        delta = tuple(a - b for a, b in zip(result.regularization, expected))
        errors.append(gradient_dot(delta, delta).sqrt().item() / norm)
        assert result.evaluations == 4
    print("FP32 rho relative errors:", errors)
    assert min(errors) < 0.002
    assert errors[-1] > min(errors)


@pytest.mark.parametrize("method", ["gd", "npo"])
@pytest.mark.parametrize("coefficient", [0.0, 0.2])
@pytest.mark.parametrize("accumulation", [1, 2])
def test_gd_npo_squared_update_matches_exact(
    tmp_path, method, coefficient, accumulation
):
    pytest.importorskip("axolotl")
    from types import SimpleNamespace
    from axolotl.core.training_args import AxolotlTrainingArguments
    from torch.nn import functional as F
    from trainers.tr_gip import GIPGradDiffTrainer, GIPNPOTrainer

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 3)
            self.head = nn.Linear(3, 7)

        def forward(self, input_ids, labels, attention_mask=None):
            logits = self.head(self.embedding(input_ids).tanh())
            return SimpleNamespace(
                logits=logits,
                loss=F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:]),
            )

    torch.manual_seed(31)
    model = TinyLM()
    kwargs = {"retain_coefficient": 2.0}
    if method == "npo":
        kwargs.update(reference_model_name_or_path="unused", beta=0.1)
    trainer = {"gd": GIPGradDiffTrainer, "npo": GIPNPOTrainer}[method](
        model=model,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to=[],
            save_strategy="no",
            sample_packing=False,
            gradient_accumulation_steps=accumulation,
        ),
        cosine_coefficient=coefficient,
        cosine_pairs=2,
        cosine_tau=0.1,
        cosine_rho=1e-3,
        **kwargs,
    )
    trainer.current_gradient_accumulation_steps = accumulation
    if method == "npo":
        trainer._theta_ref = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
    ids = torch.tensor([[1, 2, 3, 4], [4, 2, 1, 3], [2, 4, 3, 1], [3, 1, 4, 2]])
    batch = dict(
        input_ids=ids,
        labels=ids,
        attention_mask=torch.ones_like(ids),
        is_forget=torch.tensor([True, False, True, False]),
    )
    trainer.track_ghost = False
    # The reference objective uses the same base GD/NPO loss and individual CE.
    base = trainer.compute_loss(model, batch)
    trainer.finish_microbatch()
    parameters = tuple(model.parameters())
    ga, gb = [
        torch.autograd.grad(
            model(input_ids=ids[i : i + 1], labels=ids[i : i + 1]).loss,
            parameters,
            create_graph=True,
        )
        for i in (0, 2)
    ]
    cosine = gradient_dot(ga, gb) / (
        (gradient_dot(ga, ga) + 0.1**2).sqrt() * (gradient_dot(gb, gb) + 0.1**2).sqrt()
    )
    expected = torch.autograd.grad(base + coefficient * cosine.square(), parameters)
    try:
        trainer.training_step(model, batch)
        for p, g in zip(parameters, expected):
            torch.testing.assert_close(p.grad, g / accumulation, rtol=3e-3, atol=3e-5)
        assert not trainer.track_ghost
    finally:
        trainer.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_squared_finite_difference_layer_sharded():
    class Sharded(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4).double().to("cuda:0")
            self.last = nn.Linear(4, 2).double().to("cuda:1")

        def forward(self, x):
            return self.last(self.first(x.to("cuda:0")).tanh().to("cuda:1"))

    torch.manual_seed(7)
    model = Sharded()
    x = torch.randn(2, 3, dtype=torch.float64)
    a = lambda: model(x[:1]).square().mean()
    b = lambda: (model(x[1:]) - 1).square().mean()
    expected = exact_cosine_gradient(model, a, b, tau=0.1, squared=True)
    result = finite_difference_cosine(model, a, b, tau=0.1, rho=1e-5, squared=True)
    for actual, exact in zip(result.regularization, expected):
        torch.testing.assert_close(actual, exact, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("fail", [False, True])
def test_gip_accumulator_preserves_base_grads_and_unused_parameters(
    tmp_path, monkeypatch, fail
):
    pytest.importorskip("axolotl")
    from types import SimpleNamespace
    from axolotl.core.training_args import AxolotlTrainingArguments
    from torch.nn import functional as F
    import trainers.tr_gip as gip
    from trainers.ul_gd import GradDiffTrainer

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 3)
            self.head = nn.Linear(3, 7)
            self.unused = nn.Linear(2, 2)

        def forward(self, input_ids, labels, attention_mask=None):
            logits = self.head(self.embedding(input_ids))
            return SimpleNamespace(
                logits=logits,
                loss=F.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:]),
            )

    model = TinyLM()
    trainers = []
    for index, cls in enumerate((gip.GIPGradDiffTrainer, GradDiffTrainer)):
        kwargs = (
            dict(
                cosine_coefficient=0.2,
                cosine_pairs=2,
                cosine_skip_threshold=0.0 if fail else 1e8,
            )
            if index == 0
            else {}
        )
        trainer = cls(
            model=copy.deepcopy(model),
            args=AxolotlTrainingArguments(
                output_dir=str(tmp_path / str(index)),
                use_cpu=True,
                report_to=[],
                save_strategy="no",
                sample_packing=False,
            ),
            **kwargs,
        )
        trainer.current_gradient_accumulation_steps = 1
        trainers.append(trainer)
    tracked, plain = trainers
    # Preserve existing accumulation as well as this microbatch's ordinary update.
    for (name, a), b in zip(tracked.model.named_parameters(), plain.model.parameters()):
        a.grad = None if name.startswith("unused.") else torch.ones_like(a)
        b.grad = None if a.grad is None else a.grad.clone()
    ids = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
    inputs = dict(
        input_ids=ids,
        labels=ids,
        attention_mask=torch.ones_like(ids),
        is_forget=torch.tensor([True, False, True, False]),
    )
    plain.training_step(plain.model, dict(inputs))
    function = gip.finite_difference_cosine
    calls = 0

    def estimate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if fail and calls == 2:
            raise RuntimeError("second pair failed")
        return function(*args, **kwargs)

    monkeypatch.setattr(gip, "finite_difference_cosine", estimate)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="second pair failed"):
                tracked.training_step(tracked.model, dict(inputs))
        else:
            tracked.training_step(tracked.model, dict(inputs))
        for a, b in zip(tracked.model.parameters(), plain.model.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            if b.grad is None:
                assert a.grad is None
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        assert tracked.model.unused.weight.grad is None
        assert not tracked.track_ghost
    finally:
        tracked.close()
