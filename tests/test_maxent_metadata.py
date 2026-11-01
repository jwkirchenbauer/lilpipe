import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("axolotl")
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "examples/tamper-resistance/src")
)
from trainers.ul_me import MaxEntTrainer
from trainers.tr_tar import TARMETrainer
from trainers.utils import MetadataCollator


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


@pytest.mark.parametrize("trainer_cls", [MaxEntTrainer, TARMETrainer])
def test_metadata_batch_collation_loss_and_backward(trainer_cls, monkeypatch):
    from axolotl.core.trainers.base import AxolotlTrainer
    import wandb

    def base_init(self, **kwargs):
        self.data_collator = lambda rows: {
            key: torch.tensor([row[key] for row in rows]) for key in rows[0]
        }
        self.accelerator = SimpleNamespace(autocast=nullcontext)
        self.log = lambda values: None
        self.is_world_process_zero = lambda: True

    monkeypatch.setattr(AxolotlTrainer, "__init__", base_init)
    monkeypatch.setattr(wandb, "run", None)
    kwargs = {"attack_learning_rate": 0.01} if trainer_cls is TARMETrainer else {}
    trainer = trainer_cls(**kwargs)
    assert isinstance(trainer.data_collator, MetadataCollator)
    assert not isinstance(trainer.data_collator.collator, MetadataCollator)
    roles = ["defence::forget", "defence::retain"]
    if trainer_cls is TARMETrainer:
        roles.append("attack::forget")
    batch = trainer.data_collator(
        [
            {
                "input_ids": [1, 2, 0],
                "attention_mask": [1, 1, 1],
                "labels": [1, 2, 0],
                "metadata": role,
            }
            for role in roles
        ]
    )
    assert batch["metadata_ids"].tolist() == list(range(len(roles)))
    model = Model()
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()
