"""Temporary Axolotl plugin for measuring harmfulness before attack step 50."""

from pathlib import Path

import torch
from axolotl.integrations.base import BasePlugin
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, TrainerCallback
from datasets import load_dataset

from scripts.evals.evaluate_beavertail_and_capabilities import (
    BEAVERTAILS,
    BEAVERTAILS_REVISION,
    atomic_json,
    generate,
    harmful_prompts,
    harmfulness,
)


class EarlyHarmfulnessInput(BaseModel):
    early_harmfulness_output: str
    early_harmfulness_steps: list[int] = Field(
        default_factory=lambda: [0, 5, 10, 25, 50]
    )
    early_harmfulness_stop: int = 50


class EarlyHarmfulnessCallback(TrainerCallback):
    def __init__(self, trainer, cfg):
        self.trainer = trainer
        self.output = cfg.early_harmfulness_output
        self.steps = set(cfg.early_harmfulness_steps)
        self.stop = cfg.early_harmfulness_stop
        # Keep the evaluation tokenizer independent of the training collator.
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        dataset = load_dataset(
            BEAVERTAILS, split="30k_test", revision=BEAVERTAILS_REVISION
        )
        self.prompts = harmful_prompts(dataset)
        if len(self.prompts) != 100:
            raise ValueError("Expected 100 held-out harmful prompts")

    def evaluate(self, state):
        model = self.trainer.model
        modes = [(module, module.training) for module in model.modules()]
        buffers = [(b, b.detach().clone()) for b in model.buffers()]
        old_cache = model.config.use_cache
        devices = [model.device.index] if model.device.type == "cuda" else []
        try:
            model.eval()
            model.config.use_cache = True
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                with torch.autocast(
                    model.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.trainer.args.bf16,
                ):
                    generations = generate(model, self.tokenizer, self.prompts)
                score, records = harmfulness(self.prompts, generations)
        finally:
            model.config.use_cache = old_cache
            with torch.no_grad():
                for buffer, original in buffers:
                    buffer.copy_(original)
            for module, training in modes:
                module.training = training
        atomic_json(
            Path(self.output) / f"checkpoint-{state.global_step}" / "summary.json",
            {
                "checkpoint_step": state.global_step,
                "post": {"harmfulness": score, "generations": records},
            },
        )
        self.trainer.log({"eval_harmfulness": score})
        print(f"EARLY_HARMFULNESS step={state.global_step} score={score}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            self.evaluate(state)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            self.evaluate(state)
        if state.global_step >= self.stop:
            control.should_training_stop = True


class EarlyHarmfulnessPlugin(BasePlugin):
    def get_input_args(self):
        return "scripts.misc.early_harmfulness.EarlyHarmfulnessInput"

    def post_trainer_create(self, cfg, trainer):
        trainer.add_callback(EarlyHarmfulnessCallback(trainer, cfg))
