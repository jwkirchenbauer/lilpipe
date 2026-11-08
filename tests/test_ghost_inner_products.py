"""Ghost contractions and Trainer updates agree with explicit gradient references."""

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

SRC = Path(__file__).resolve().parents[1] / "examples/tamper-resistance/src"
sys.path.insert(0, str(SRC))
from trainers.ghost_inner_products import (
    GhostCapture,
    GradientTerm,
    assemble_gram,
    causal_lm_losses,
    term_inner_product,
)


def explicit_gram(losses, model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for loss in losses:
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(parameters, gradients)
                ]
            ).double()
        )
    gradients = torch.stack(rows)
    return gradients @ gradients.T


class TiedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 5, padding_idx=0)
        self.norm = nn.LayerNorm(5)
        self.proj = nn.Linear(5, 5)
        self.head = nn.Linear(5, 11, bias=False)
        self.head.weight = self.embedding.weight
        self.unused = nn.Linear(5, 5)

    def forward(self, ids):
        x = self.norm(self.embedding(ids))
        self.unused(x)  # A captured branch not used by the loss.
        return self.head(self.proj(torch.tanh(self.proj(x))))


def check_model(model, inputs, **kwargs):
    model = model.double()
    hooks_before = [tuple(m._forward_hooks) for m in model.modules()]
    with GhostCapture(model, chunk_size=2, dtype=torch.float64) as capture:
        outputs = model(inputs, **kwargs)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        losses = outputs.square().flatten(1).mean(1)
        expected = explicit_gram(losses, model)
        actual = capture.gram(losses, retain_graph=True)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
        assert not actual.requires_grad
        assert all(p.grad is None for p in model.parameters())
        losses.mean().backward()
        assert any(p.grad is not None for p in model.parameters())
    assert [tuple(m._forward_hooks) for m in model.modules()] == hooks_before


@pytest.mark.parametrize(
    "indexed_a,indexed_b", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("chunk_size", [1, 3, 20])
def test_term_contractions(indexed_a, indexed_b, chunk_size):
    torch.manual_seed(7)

    def make_term(indexed, positions):
        left = (
            torch.randint(0, 5, (2, positions))
            if indexed
            else torch.randn(2, positions, 5, dtype=torch.float64)
        )
        right = torch.randn(2, positions, 4, dtype=torch.float64)
        dense_left = F.one_hot(left, 5).double() if indexed else left
        return GradientTerm(left, right, indexed), dense_left.transpose(1, 2) @ right

    a, dense_a = make_term(indexed_a, 4)
    b, dense_b = make_term(indexed_b, 3)
    actual = term_inner_product(a, b, 0, 1, chunk_size=chunk_size)
    torch.testing.assert_close(actual, (dense_a[0] * dense_b[1]).sum())


def test_shared_vector_terms_include_cross_terms():
    p = nn.Parameter(torch.ones(3))
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    actual = assemble_gram(
        [(p, GradientTerm(a)), (p, GradientTerm(b))], batch_size=2, device="cpu"
    )
    torch.testing.assert_close(actual, (a + b) @ (a + b).T)


def test_linear_positions():
    check_model(
        nn.Sequential(nn.Linear(4, 7), nn.Tanh(), nn.Linear(7, 3)),
        torch.randn(3, 5, 4, dtype=torch.float64),
    )


def test_tied_embedding_reused_layer_and_unused_branch():
    check_model(TiedLM(), torch.tensor([[0, 2, 2, 4], [2, 3, 0, 4], [1, 1, 4, 5]]))


def test_rmsnorm_zero_scale_and_frozen_parameters():
    model = nn.Sequential(nn.Linear(4, 4), nn.RMSNorm(4), nn.Linear(4, 2))
    model[0].weight.requires_grad_(False)
    with torch.no_grad():
        model[1].weight[0] = 0
    check_model(model, torch.randn(3, 5, 4, dtype=torch.float64))


def test_capture_consumption_and_second_batch():
    model = nn.Linear(2, 3).double()
    with GhostCapture(model, dtype=torch.float64) as capture:
        for n in (3, 1):
            losses = model(torch.randn(n, 2, dtype=torch.float64)).square().sum(-1)
            assert capture.gram(losses).shape == (n, n)
            with pytest.raises(RuntimeError):
                capture.gram(losses)
            with pytest.raises(RuntimeError):
                losses.sum().backward()


def test_invalid_loss_consumes_capture_and_removes_hooks():
    model = nn.Linear(2, 3)
    with GhostCapture(model) as capture:
        loss = model(torch.randn(3, 2)).sum()
        with pytest.raises(ValueError):
            capture.gram(loss)
        with pytest.raises(RuntimeError):
            capture.gram(loss)
    assert not model._forward_hooks


def test_unsupported_module_leaves_no_hooks():
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    with pytest.raises(NotImplementedError):
        with GhostCapture(model):
            pass
    assert all(not m._forward_hooks for m in model.modules())


def test_forward_failure_removes_hooks():
    model = nn.Linear(2, 3)
    with pytest.raises(RuntimeError):
        with GhostCapture(model):
            model(torch.randn(3, 7))
    assert not model._forward_hooks


def test_masked_causal_losses():
    logits = torch.randn(3, 4, 7, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, 2, -100, 3], [2, -100, -100, -100], [1, 4, 5, 6]])
    losses = causal_lm_losses(logits, labels)
    expected = torch.stack(
        [
            (
                F.cross_entropy(logits[0, 0:1], labels[0, 1:2])
                + F.cross_entropy(logits[0, 2:3], labels[0, 3:4])
            )
            / 2,
            logits[1].sum() * 0,
            F.cross_entropy(logits[2, :-1], labels[2, 1:]),
        ]
    )
    torch.testing.assert_close(losses, expected)
    losses.sum().backward()
    assert torch.count_nonzero(logits.grad[1]) == 0


def tiny_gpt():
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
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
    return transformers.GPT2LMHeadModel(config)


def test_tiny_hf_gram():
    check_model(
        tiny_gpt(),
        torch.randint(0, 11, (3, 4)),
        position_ids=torch.arange(4).expand(3, -1),
    )


@pytest.mark.parametrize("accumulation", [1, 2])
def test_trainer_matches_explicit_update(tmp_path, accumulation):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    from trainers.ghost_trainer import GhostInnerProductsMixin

    class GhostTrainer(GhostInnerProductsMixin, transformers.Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            # Construct an independent explicit reference from the same model
            # weights, without triggering any hooks on the observed model.
            reference = gradient_reference
            reference.load_state_dict(model.state_dict())
            outputs = reference(**inputs)
            labels = inputs["labels"][:, 1:]
            token_losses = F.cross_entropy(
                outputs.logits[:, :-1].transpose(1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            # Ask the unchanged Trainer to apply its actual loss normalization.
            loss = transformers.Trainer.compute_loss(
                self, reference, dict(inputs), num_items_in_batch=num_items_in_batch
            )
            contributions = token_losses.sum(1) * (
                loss.detach() / token_losses.sum().detach()
            )
            divisor = (
                self.current_gradient_accumulation_steps
                if not self.model_accepts_loss_kwargs or num_items_in_batch is None
                else 1
            )
            expected = explicit_gram(contributions / divisor, reference)
            result = super().training_step(model, inputs, num_items_in_batch)
            torch.testing.assert_close(
                self.gram.double(), expected, rtol=3e-5, atol=1e-6
            )
            self.expected_means.append(expected.mean().item())
            return result

    torch.manual_seed(7)
    model = tiny_gpt()
    reference = copy.deepcopy(model)
    gradient_reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 4, 5], [2, 3, 4, 5], [3, 4, 5, 6]])[
        : 2 * accumulation
    ]
    dataset = [
        {"input_ids": row, "labels": row.clone(), "position_ids": torch.arange(4)}
        for row in ids
    ]
    dataset[0]["labels"][1] = -100
    hooks_before = [tuple(m._forward_hooks) for m in model.modules()]

    def make_trainer(cls, model, name):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        return cls(
            model=model,
            args=transformers.TrainingArguments(
                output_dir=str(tmp_path / name),
                use_cpu=True,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=accumulation,
                max_steps=1,
                max_grad_norm=0,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
            ),
            train_dataset=dataset,
            optimizers=(
                optimizer,
                torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
            ),
        )

    baseline = make_trainer(transformers.Trainer, reference, "baseline")
    baseline.train()
    trainer = make_trainer(GhostTrainer, model, "ghost")
    trainer.expected_means = []
    trainer.train()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    logged = [
        row["ghost_gram_mean"]
        for row in trainer.state.log_history
        if "ghost_gram_mean" in row
    ]
    assert logged == pytest.approx(
        [sum(trainer.expected_means) / accumulation], rel=3e-5
    )
    assert [tuple(m._forward_hooks) for m in model.modules()] == hooks_before
    assert not trainer.output_handles and not trainer.previous_uses
    model.eval()
    batch = {"input_ids": ids, "labels": ids}
    with torch.no_grad():
        expected = model(**batch).loss
        actual = trainer.compute_loss(model, dict(batch))
    torch.testing.assert_close(actual, expected)


def test_checkpointing_rejected_without_hooks():
    model = tiny_gpt()
    model.gradient_checkpointing_enable()
    hooks_before = [tuple(m._forward_hooks) for m in model.modules()]
    with pytest.raises(NotImplementedError, match="checkpointing"):
        with GhostCapture(model):
            pass
    assert [tuple(m._forward_hooks) for m in model.modules()] == hooks_before


def test_tiny_llama_gram():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=8,
        attention_dropout=0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    check_model(transformers.LlamaForCausalLM(config), torch.randint(0, 11, (3, 4)))


def test_broadcast_position_embeddings_rejected():
    model = tiny_gpt()
    with GhostCapture(model) as capture:
        losses = model(torch.randint(0, 11, (3, 4))).logits.square().mean((1, 2))
        with pytest.raises(ValueError, match="position_ids"):
            capture.gram(losses)


def test_cpu_autocast_and_existing_parameter_gradients():
    model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.Linear(4, 2))
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    previous = [p.grad.clone() for p in model.parameters()]
    with GhostCapture(model) as capture:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses = model(torch.randn(3, 2, 4)).float().square().mean((1, 2))
        expected = explicit_gram(losses, model)
        actual = capture.gram(losses, retain_graph=True)
    # Explicit bf16 weight gradients round after contraction; ghost contractions
    # accumulate in float32, so compare at bf16 precision.
    torch.testing.assert_close(actual.double(), expected, rtol=0.03, atol=0.003)
    for p, before in zip(model.parameters(), previous):
        torch.testing.assert_close(p.grad, before)
    losses.mean().backward()


@pytest.mark.parametrize("kind", ["linear", "tied", "norm"])
def test_backward_hook_gram_matches_explicit(tmp_path, kind):
    transformers = pytest.importorskip("transformers")
    from trainers.ghost_trainer import GhostInnerProductsMixin

    class GhostTrainer(GhostInnerProductsMixin, transformers.Trainer):
        pass

    torch.manual_seed(7)
    if kind == "tied":
        model = TiedLM().double()
        inputs = torch.tensor([[0, 2, 2, 4], [2, 3, 0, 4], [1, 1, 4, 5]])
    else:
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.LayerNorm(4) if kind == "linear" else nn.RMSNorm(4),
            nn.Linear(4, 2),
        ).double()
        inputs = torch.randn(3, 5, 4, dtype=torch.float64)
        model[0].weight.requires_grad_(False)
        with torch.no_grad():
            model[1].weight[0] = 0
    reference = copy.deepcopy(model)
    losses = reference(inputs).square().flatten(1).mean(1) / 3
    expected = explicit_gram(losses, reference)
    trainer = GhostTrainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[]
        ),
        ghost_dtype=torch.float64,
    )
    try:
        model(inputs).square().flatten(1).mean(1).mean().backward()
        torch.testing.assert_close(trainer.gram, expected, rtol=1e-8, atol=1e-9)
        # Exercise AMP normalization independently of CUDA availability.
        from types import SimpleNamespace

        trainer.accelerator.scaler = SimpleNamespace(get_scale=lambda: 8)
        trainer.gram.mul_(64)
        trainer.finish_microbatch()
        torch.testing.assert_close(trainer.microbatch_means[0], expected.mean())
        assert trainer.gram is None and not trainer.previous_uses
    finally:
        trainer.close()
    assert all(not m._forward_hooks for m in model.modules())


def test_vectorized_grams_match_chunked_reference():
    from trainers.ghost_inner_products import linear_weight_gram, term_gram

    z = torch.randn(3, 4, 5, dtype=torch.float64)
    d = torch.randn(3, 4, 6, dtype=torch.float64)
    term = GradientTerm(d, z)
    expected = torch.stack(
        [
            torch.stack(
                [term_inner_product(term, term, i, j, chunk_size=2) for j in range(3)]
            )
            for i in range(3)
        ]
    )
    torch.testing.assert_close(linear_weight_gram(z, d), expected)
    torch.testing.assert_close(term_gram(term, term), expected)


def test_axolotl_trainer_logs_without_overriding_training(tmp_path):
    pytest.importorskip("axolotl")
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.training_args import AxolotlTrainingArguments
    from trainers.ghost_axolotl import GhostAxolotlTrainer

    assert GhostAxolotlTrainer.training_step is AxolotlTrainer.training_step
    assert GhostAxolotlTrainer.compute_loss is AxolotlTrainer.compute_loss
    dataset = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(4),
        }
        for _ in range(2)
    ]
    from datasets import Dataset

    dataset = Dataset.from_list(
        [{key: value.tolist() for key, value in row.items()} for row in dataset]
    )
    trainer = GhostAxolotlTrainer(
        model=tiny_gpt(),
        train_dataset=dataset,
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            max_steps=1,
            per_device_train_batch_size=2,
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            sample_packing=False,
        ),
    )
    try:
        trainer.train()
        values = [
            row["ghost_gram_mean"]
            for row in trainer.state.log_history
            if "ghost_gram_mean" in row
        ]
        assert len(values) == 1 and values[0] > 0
        assert not trainer.hook_handles and not trainer.output_handles
    finally:
        trainer.close()
