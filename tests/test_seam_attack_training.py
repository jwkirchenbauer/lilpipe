"""Training-only SEAM copy preserves the authors' attack and saves its result."""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SEAM = Path(__file__).resolve().parents[1] / "examples/seam"


def main_function(path):
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def test_attack_branch_matches_authors():
    original = main_function(SEAM / "src/eval.py")
    copied = main_function(SEAM / "scripts/train.py")

    def attack_branch(main):
        return next(
            node
            for node in main.body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "eval_args.attack"
        )

    assert ast.dump(attack_branch(original)) == ast.dump(attack_branch(copied))
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(copied)
        if isinstance(node, ast.Call)
    ]
    assert not any(
        "eval_utility" in call or "eval_harmfulness" in call for call in calls
    )


@pytest.mark.parametrize("lora", [False, True])
def test_training_saves_final_model_without_evaluation(tmp_path, monkeypatch, lora):
    monkeypatch.setattr("sys.argv", ["train.py", "--output_dir", str(tmp_path)])
    main = main_function(SEAM / "scripts/train.py")
    training_args = SimpleNamespace(output_dir=str(tmp_path))
    eval_args = SimpleNamespace(
        model_name="starting-model",
        tokenizer_name="tokenizer",
        attack_size=1000,
        attack=True,
        lora_attack=lora,
        wandb_project_name="seam",
        wandb_run_name="attack",
    )
    tokenizer = Mock()
    model = Mock()
    merged = Mock()
    model.merge_and_unload.return_value = merged
    trainer = Mock()
    trainer_factory = Mock(return_value=trainer)
    harmful_dataset = object()
    loader = Mock(return_value=(harmful_dataset, None, None))
    parser = Mock()
    parser.parse_args_into_dataclasses.return_value = (training_args, eval_args)
    namespace = dict(
        argparse=argparse,
        TrainingArguments=object,
        EvalArguments=object,
        load_yaml_config=lambda path: {"TrainingArguments": {}, "EvalArguments": {}},
        MyParser=lambda types: parser,
        asdict=vars,
        wandb=Mock(),
        AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=Mock(from_pretrained=Mock(return_value=model)),
        torch=SimpleNamespace(bfloat16="bf16"),
        construct_beavertails_dataset=loader,
        Trainer=trainer_factory,
        LoraConfig=Mock(),
        TaskType=SimpleNamespace(CAUSAL_LM="causal"),
        get_peft_model=lambda model, config: model,
    )
    exec(
        compile(ast.Module(body=[main], type_ignores=[]), "train.py", "exec"), namespace
    )
    namespace["main"]()
    trainer_factory.assert_called_once_with(
        model=model,
        args=training_args,
        train_dataset=harmful_dataset,
        tokenizer=tokenizer,
    )
    trainer.train.assert_called_once_with()
    trainer.save_state.assert_called_once_with()
    (merged if lora else model).save_pretrained.assert_called_once_with(str(tmp_path))
    tokenizer.save_pretrained.assert_called_once_with(str(tmp_path))
    loader.assert_called_once_with(tokenizer, attack_size=1000)
