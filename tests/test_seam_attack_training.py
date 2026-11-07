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


@pytest.mark.parametrize("fail", [False, True])
def test_single_run_launcher_forwards_one_lr_and_saves_checkpoints(tmp_path, fail):
    import json
    import os
    import subprocess
    import sys
    import yaml

    script = (
        SEAM.parent
        / "tamper-resistance/scripts/slurm/tamia/seam_codebase_train_seam_relearning.sbatch"
    )
    workdir = tmp_path / "tamper-resistance"
    workdir.mkdir()
    seam = tmp_path / "seam/scripts"
    seam.mkdir(parents=True)
    (seam / "train.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "a=sys.argv[1:]\n"
        "Path(os.environ['CALLS']).write_text(json.dumps(a))\n"
        "if os.environ['FAIL']=='True':sys.exit(7)\n"
        "def v(k):return a[a.index('--'+k)+1]\n"
        "root=Path(v('output_dir'))\n"
        "for n in range(int(v('save_steps')),int(v('max_steps'))+1,int(v('save_steps'))):\n"
        " p=root/f'checkpoint-{n}';p.mkdir(parents=True)\n"
        " (p/'trainer_state.json').write_text(json.dumps({'global_step':n}))\n"
        " (p/'model.safetensors').touch()\n"
        "(root/'tokenizer_config.json').write_text('{}')\n"
    )
    config = tmp_path / "author-eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "TrainingArguments": {
                    "per_device_train_batch_size": 4,
                    "num_train_epochs": 1,
                },
                "EvalArguments": {
                    "attack": True,
                    "lora_attack": False,
                    "attack_size": 1000,
                },
            }
        )
    )
    starting = tmp_path / "starting-model"
    starting.mkdir()
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nexec "$TEST_PYTHON" "$@"\n')
    python.chmod(0o755)
    output = tmp_path / "models/run"
    calls = tmp_path / "calls.json"
    result = subprocess.run(
        [
            "bash",
            str(script),
            str(config),
            str(starting),
            str(output),
            "0.0002",
            "--max_steps",
            "250",
            "--seed",
            "42",
        ],
        cwd=workdir,
        env={
            **os.environ,
            "SCRATCH": str(tmp_path),
            "SEAM_ENV": str(python.parents[1]),
            "TEST_PYTHON": sys.executable,
            "CALLS": str(calls),
            "FAIL": str(fail),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == (7 if fail else 0), result.stderr
    args = json.loads(calls.read_text())
    assert args.count("--learning_rate") == 1
    assert args[args.index("--learning_rate") + 1] == "0.0002"
    assert args[args.index("--wandb_run_name") + 1] == "run"
    assert args[args.index("--config") + 1] == str(config)
    assert args[args.index("--model_name") + 1] == str(starting)
    assert args[-4:] == ["--max_steps", "250", "--seed", "42"]
    assert yaml.safe_load(config.read_text())["EvalArguments"]["attack_size"] == 1000
    assert len(list(output.glob("checkpoint-*"))) == (0 if fail else 5)
