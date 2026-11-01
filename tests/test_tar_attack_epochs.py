import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def attack():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/tamper-resistance/src/trainers/tr_tar.py"
    )
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "adamw_attack"
    )
    scope = {"torch": torch, "ROLE_FIELDS": ("input_ids", "attention_mask", "labels")}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope
    )
    return scope["adamw_attack"]


def test_epochs_revisit_all_rows_and_keep_partial_batches(attack):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, input_ids, attention_mask, labels):
            self.seen.append(input_ids.tolist())
            return SimpleNamespace(
                loss=(self.weight * input_ids.float()).square().mean()
            )

    model = Model()
    inputs = {
        key: torch.arange(1, 6) for key in ("input_ids", "attention_mask", "labels")
    }
    steps = []
    result = attack(
        model,
        inputs,
        attack_forget_batch_size_per_iteration=2,
        attack_epochs=3,
        attack_learning_rate=0.01,
        on_loss_backward=lambda step, loss: steps.append((step, loss.requires_grad)),
    )
    assert result is model
    assert model.seen == [[1, 2], [3, 4], [5]] * 3
    assert steps == [(step, False) for step in range(9)]
    assert model.weight.item() < 1.0
