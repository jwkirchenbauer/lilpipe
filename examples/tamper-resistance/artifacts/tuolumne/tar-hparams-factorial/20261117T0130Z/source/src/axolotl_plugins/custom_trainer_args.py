"""Axolotl plugin for declarative custom-trainer constructor arguments."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Mapping
from typing import Any

from axolotl.integrations.base import BasePlugin
from pydantic import BaseModel, ConfigDict, Field


class CustomTrainerArgsInput(BaseModel):
    """Configuration fields registered by the plugin."""

    model_config = ConfigDict(extra="allow")

    trainer_cls: str | None = None
    custom_trainer_args: dict[str, Any] = Field(default_factory=dict)


RESERVED_ARGUMENTS = frozenset(
    {
        "args",
        "callbacks",
        "data_collator",
        "eval_dataset",
        "model",
        "model_init",
        "optimizers",
        "preprocess_logits_for_metrics",
        "processing_class",
        "tokenizer",
        "train_dataset",
    }
)


def import_trainer(path: str):
    """Import and return the trainer class at a dotted path."""
    if not path or "." not in path:
        raise ValueError("trainer_cls must be a dotted class path")
    module_name, class_name = path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
        trainer_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import trainer_cls {path!r}") from exc
    if not inspect.isclass(trainer_cls):
        raise ValueError(f"trainer_cls {path!r} is not a class")
    return trainer_cls


def custom_parameter_names(trainer_cls: type) -> set[str]:
    """Collect explicit constructor parameters supplied by custom trainer classes."""
    names = set()
    for cls in trainer_cls.__mro__:
        if cls.__module__.startswith(("axolotl.", "transformers.")):
            break
        signature = inspect.signature(cls.__init__)
        names.update(
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        )
    return names


def validate_custom_args(trainer_cls: type, custom_args: Mapping[str, Any]) -> dict:
    """Validate custom arguments without allowing Axolotl constructor fields."""
    if not isinstance(custom_args, Mapping):
        raise TypeError("custom_trainer_args must be a mapping")
    reserved = set(custom_args) & RESERVED_ARGUMENTS
    if reserved:
        raise ValueError(
            f"custom_trainer_args contains Axolotl-reserved names: {sorted(reserved)}"
        )
    unknown = set(custom_args) - custom_parameter_names(trainer_cls)
    if unknown:
        raise ValueError(f"unknown custom trainer arguments: {sorted(unknown)}")
    return dict(custom_args)


def trainer_with_custom_args(trainer_cls: type, custom_args: Mapping[str, Any]):
    """Create a trainer subclass that injects validated constructor arguments."""
    configured = validate_custom_args(trainer_cls, custom_args)

    class ConfiguredTrainer(trainer_cls):
        def __init__(self, *args, **kwargs):
            collisions = configured.keys() & kwargs.keys()
            if collisions:
                raise ValueError(
                    "custom trainer argument collision: " f"{sorted(collisions)}"
                )
            super().__init__(*args, **kwargs, **configured)

    ConfiguredTrainer.__name__ = f"Configured{trainer_cls.__name__}"
    ConfiguredTrainer.__qualname__ = ConfiguredTrainer.__name__
    ConfiguredTrainer.__module__ = trainer_cls.__module__
    return ConfiguredTrainer


class CustomTrainerArgsPlugin(BasePlugin):
    """Load ``trainer_cls`` and pass ``custom_trainer_args`` to its constructor."""

    def get_input_args(self) -> str:
        return "axolotl_plugins.custom_trainer_args.CustomTrainerArgsInput"

    def get_trainer_cls(self, cfg):
        trainer_path = cfg.get("trainer_cls")
        custom_args = cfg.get("custom_trainer_args") or {}
        if not trainer_path:
            raise ValueError("CustomTrainerArgsPlugin requires trainer_cls")
        return trainer_with_custom_args(import_trainer(trainer_path), custom_args)
