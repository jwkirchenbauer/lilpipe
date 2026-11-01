import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples/tamper-resistance/src")
)
from trainers.ul_me import MaxEntTrainer


@pytest.mark.parametrize("method", ["compute_forget_loss", "compute_retain_loss"])
def test_selected_loss_and_gradients_match_reference(method):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.1, 0.3, -0.2]))

        def forward(self, input_ids, attention_mask, labels=None):
            logits = input_ids[..., None].float() * self.weight
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 3), labels[:, 1:].reshape(-1)
                )
            return SimpleNamespace(logits=logits, loss=loss)

    trainer = object.__new__(MaxEntTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 3.0
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0], [2, 1, 0]]),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([[1, 2, -100], [2, 1, 0]]),
        "is_forget": torch.tensor([True, False]),
    }
    mask = (
        inputs["is_forget"] if method == "compute_forget_loss" else ~inputs["is_forget"]
    )
    selected = {key: value[mask] for key, value in inputs.items() if key != "is_forget"}
    model = Model()
    loss_fn = getattr(trainer, method)
    outputs = model(**selected)
    if method == "compute_forget_loss":
        log_probs = outputs.logits[:, :-1].float().log_softmax(-1)
        negative_entropy = (log_probs.exp() * log_probs).sum(-1)
        valid = selected["labels"][:, 1:] != -100
        expected_loss = trainer.forget_coefficient * negative_entropy[valid].mean()
    else:
        expected_loss = trainer.retain_coefficient * outputs.loss
    expected_grad = torch.autograd.grad(expected_loss, model.weight)[0]
    selected_loss = loss_fn(model, selected)
    actual_grad = torch.autograd.grad(selected_loss, model.weight)[0]
    torch.testing.assert_close(selected_loss, expected_loss)
    torch.testing.assert_close(actual_grad, expected_grad)
    if method == "compute_forget_loss":
        functional_loss = loss_fn(lambda **kwargs: model(**kwargs), selected)
        torch.testing.assert_close(functional_loss, selected_loss)
