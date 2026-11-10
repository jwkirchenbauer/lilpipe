"""Completion strategy for WMDP-Bio forget-set relearning."""

from axolotl.prompt_strategies.completion import (
    CompletionPrompter,
    CompletionPromptTokenizingStrategy,
)


class WmdpBioCompletionStrategy(CompletionPromptTokenizingStrategy):
    def parse_instruction_fields(self, row):
        document = "\n\n".join(row[field] for field in ("title", "abstract", "text"))
        return document, "", ""


def load(tokenizer, cfg, ds_cfg=None):
    return WmdpBioCompletionStrategy(
        CompletionPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        max_length=cfg.sequence_len,
    )
