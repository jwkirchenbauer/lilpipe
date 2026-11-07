"""Our SEAM attacks retain checkpoints and use the shared evaluation protocol."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from lilpipe import load

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
START = "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"


@pytest.mark.parametrize("steps", [250, 1000])
def test_own_relearning_configs_and_evaluation_dependencies(monkeypatch, steps):
    monkeypatch.chdir(EXAMPLE)
    suffix = "-ms1000" if steps == 1000 else ""
    pipeline = load(
        f"configs/experiments/llama2-7b-seam-own-trainer-relearning{suffix}.yml"
    )
    stages = pipeline.plan().stages
    training = [stage for stage in stages if stage.id.startswith("train-")]
    evaluation = [stage for stage in stages if stage.id.startswith("eval-")]
    assert len(training) == (4 if steps == 1000 else 3)
    assert len(evaluation) == (8 if steps == 1000 else 3)
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4] if steps == 1000 else [1e-5, 5e-5, 2e-4]
    for stage, lr in zip(training, learning_rates, strict=True):
        config = yaml.safe_load(Path(stage.args[0]).read_text())
        assert config["base_model"] == f"artifacts/tamia/models/{START}"
        assert config["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
        assert config["learning_rate"] == lr
        assert config["max_steps"] == steps
        assert config["warmup_ratio"] == (0.5 if lr == 2e-4 else 0.1)
        assert config["save_steps"] == 50
        assert config["save_total_limit"] >= steps // 50
        assert config["micro_batch_size"] == 12
        assert config["gradient_accumulation_steps"] == 1
        assert config["sequence_len"] == 256
        assert config["train_on_inputs"] is False
        assert set(config["device_map"].values()) == {0, 1, 2, 3}
        assert config["wandb_mode"] == "online"
        assert config["wandb_project"] == "lp-tamper-resistance"
        assert config["wandb_name"] == config["output_dir"].split("/")[-1]
        assert "--relearn-fft-author-codebase" not in config["output_dir"]
        assert all(
            ds["message_property_mappings"]["offset"] == "0"
            for ds in config["datasets"][:2]
        )
    for stage in evaluation:
        assert stage.script.endswith("eval_harmfulness_and_capabilities.sbatch")
        assert stage.args[0] == "--checkpoints"
        assert "INCLUDE_GSM8K=1" in stage.sbatch_args[-1]
        model = stage.args[1].split("/")[-1]
        assert f"train-{model}" in stage.depends_on


def test_relearning_logs_cosine_without_optimizing_benign_loss(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(
        sys.modules, "trainers.rl_bf16", SimpleNamespace(BF16Trainer=object)
    )
    spec = importlib.util.spec_from_file_location(
        "seam_relearn_logging_test", EXAMPLE / "src/trainers/tr_seam.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = object.__new__(module.SEAMRelearnTrainer)
    trainer.current_gradient_accumulation_steps = 2
    trainer.args = SimpleNamespace(device="cpu")
    trainer._prepare_inputs = lambda inputs: inputs
    weight = torch.tensor(2.0, requires_grad=True)
    trainer.compute_ce = lambda model, inputs: weight * inputs["input_ids"].sum()

    def cosine(model, harmful, benign, compute_loss_only):
        assert compute_loss_only
        assert harmful["input_ids"].tolist() == [1.0, 4.0]
        assert benign["input_ids"].tolist() == [3.0, 6.0]
        return torch.tensor(0.25)

    trainer.accumulate_sd_grad = cosine
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    logged = []
    trainer.log = logged.append
    loss = trainer.training_step(None, {"input_ids": torch.arange(1.0, 7.0)})
    assert loss.item() == 5.0
    assert weight.grad.item() == 2.5
    assert logged == [{"loss_hh": 10.0, "loss_sd": 0.25, "cosine_similarity": 0.25}]


def test_cosine_plot_keeps_steps_and_does_not_duplicate_metric_aliases(tmp_path):
    import json

    spec = importlib.util.spec_from_file_location(
        "gradient_cosine_plot",
        EXAMPLE / "scripts/analysis/relearning-trajectory/plot_gradient_cosine.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = tmp_path / "trainer_state.json"
    state.write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 0, "cosine_similarity": 0.25, "loss_sd": 0.25},
                    {"step": 1, "loss": 2.0},
                    {"step": 1, "loss_sd": -0.5},
                ]
            }
        )
    )
    assert module.get_results([{"path": state, "lr": "5e-5"}]) == [
        {"Step": 0, "Cosine": 0.25, "LR": 5e-5},
        {"Step": 1, "Cosine": -0.5, "LR": 5e-5},
    ]
