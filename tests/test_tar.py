import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


@pytest.fixture(scope="module")
def tar():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        from trainers import tr_tar

        yield tr_tar
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture
def tiny_lm():
    torch = pytest.importorskip("torch")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(5, 5) * 0.1)
            self.frozen_weight = torch.nn.Parameter(torch.randn(2), requires_grad=False)

        def forward(self, input_ids, attention_mask, labels=None):
            logits = torch.nn.functional.one_hot(input_ids, 5).float() @ self.weight
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1)
                )
            return SimpleNamespace(loss=loss, logits=logits)

    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def batch():
    torch = pytest.importorskip("torch")
    return {
        "input_ids": torch.tensor([[0, 1, 2]] * 4),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
        "labels": torch.tensor([[0, 1, 2]] * 4),
        "is_forget": torch.tensor([True, True, False, False]),
    }


def make_trainer(tar, trainer_cls, logs):
    trainer = object.__new__(trainer_cls)
    trainer.sample_attack_fn = partial(tar.adamw_attack, lr=0.1)
    trainer.forget_coefficient = trainer.retain_coefficient = 1.0
    trainer.state = SimpleNamespace(global_step=7)
    trainer.inner_step_total = 0
    trainer._inner_loss_metric_defined = False
    trainer.is_world_process_zero = lambda: True
    trainer.log = logs.append
    return trainer


def test_base_forget_loss_is_taken_at_attacked_parameters(tar) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self):
            return SimpleNamespace(loss=self.weight.square())

    class Trainer(tar.TARBaseTrainer, torch.nn.Module):
        def compute_forget_loss(self, model_forward, inputs):
            return model_forward().loss

        def compute_retain_loss(self, model, inputs):
            return model().loss * 0

    def attack(model, inputs, on_loss_backward):
        on_loss_backward(0, torch.tensor(3.0))
        on_loss_backward(1, torch.tensor(2.0))
        with torch.no_grad():
            model.weight += 1.0
        return model

    model = Model()
    trainer = Trainer(sample_attack_fn=attack)
    trainer.state = SimpleNamespace(global_step=7)
    logs = []
    trainer.log = logs.append
    loss = trainer.compute_loss(model, {})
    loss.backward()
    assert logs == [
        {"inner_loss": 3.0, "inner_step_total": 0},
        {"inner_loss": 2.0, "inner_step_total": 1},
        {"loss_forget": 9.0, "loss_retain": 0.0},
    ]
    assert model.weight.item() == 2.0
    assert model.weight.grad.item() == pytest.approx(6.0)


def test_inner_losses_share_a_curve_across_outer_steps(tar, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    defined = []
    monkeypatch.setattr(tar.wandb, "run", object())
    monkeypatch.setattr(
        tar.wandb,
        "define_metric",
        lambda name, step_metric: defined.append((name, step_metric)),
    )
    logs = []
    trainer = make_trainer(tar, tar.TARGDTrainer, logs)

    trainer._log_inner_loss(0, torch.tensor(1.5))
    trainer._log_inner_loss(1, torch.tensor(1.0))
    trainer.state.global_step += 1
    trainer._log_inner_loss(0, torch.tensor(2.0))

    assert defined == [("train/inner_loss", "train/inner_step_total")]
    assert logs == [
        {"inner_loss": 1.5, "inner_step_total": 0},
        {"inner_loss": 1.0, "inner_step_total": 1},
        {"inner_loss": 2.0, "inner_step_total": 2},
    ]


def test_inner_loss_metric_waits_for_wandb_run(tar, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    defined = []
    monkeypatch.setattr(tar.wandb, "run", None)
    monkeypatch.setattr(
        tar.wandb,
        "define_metric",
        lambda name, step_metric: defined.append((name, step_metric)),
    )
    trainer = make_trainer(tar, tar.TARGDTrainer, [])
    trainer._log_inner_loss(0, torch.tensor(1.5))
    assert defined == []

    monkeypatch.setattr(tar.wandb, "run", object())
    trainer._log_inner_loss(1, torch.tensor(1.0))
    trainer._log_inner_loss(2, torch.tensor(0.5))
    assert defined == [("train/inner_loss", "train/inner_step_total")]


def test_adamw_attack_relearns_forget_rows_under_no_grad(tar, tiny_lm, batch) -> None:
    torch = pytest.importorskip("torch")
    forget = {field: batch[field][:2] for field in tar.ROLE_FIELDS}
    before = tiny_lm(**forget).loss.item()
    losses = []
    with torch.no_grad():
        attacked = tar.adamw_attack(
            tiny_lm,
            batch,
            lr=0.1,
            on_loss_backward=lambda step, loss: losses.append((step, loss.item())),
        )
    assert attacked is tiny_lm
    assert [step for step, _ in losses] == list(range(8))
    assert tiny_lm(**forget).loss.item() < before


@pytest.mark.parametrize("trainer_name", ["TARGDTrainer", "TARMETrainer"])
def test_tar_forget_loss_backpropagates_to_trained_model(
    tar, tiny_lm, batch, trainer_name
) -> None:
    torch = pytest.importorskip("torch")
    logs = []
    trainer = make_trainer(tar, getattr(tar, trainer_name), logs)
    trainer.retain_coefficient = 0.0
    before = tiny_lm.weight.detach().clone()

    trainer.compute_loss(tiny_lm, batch).backward()

    assert torch.equal(tiny_lm.weight, before)
    assert tiny_lm.weight.grad.abs().sum() > 0
    assert tiny_lm.frozen_weight.grad is None
    assert [log["inner_step_total"] for log in logs[:-1]] == list(range(8))


def test_tar_does_not_log_inner_losses_outside_training(tar, tiny_lm, batch) -> None:
    torch = pytest.importorskip("torch")
    logs = []
    trainer = make_trainer(tar, tar.TARGDTrainer, logs)
    tiny_lm.eval()
    with torch.no_grad():
        trainer.compute_loss(tiny_lm, batch)
    assert logs == []


def test_tar_trainer_builds_attack_from_inner_args(tar, monkeypatch) -> None:
    monkeypatch.setattr(
        tar.MaxEntTrainer, "__init__", lambda self, *args, **kwargs: None
    )
    trainer = tar.TARMETrainer(inner_steps=16, inner_learning_rate=2.5e-6)
    assert trainer.sample_attack_fn.func is tar.adamw_attack
    assert trainer.sample_attack_fn.keywords == {"steps": 16, "lr": 2.5e-6}
