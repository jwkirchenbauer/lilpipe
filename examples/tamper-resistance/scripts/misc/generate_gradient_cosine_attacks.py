"""Attach matched harmful attacks to a generated cosine-defense sweep."""

import argparse
import copy
import importlib.util
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--attack-base",
        type=Path,
        default=Path(
            "configs/stages-beavertails/1_relearn/qwen3-0.6b-beavertail--relearn-fft-ghost-ms250/base.yml"
        ),
    )
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("hpo", "scripts/hpo.py")
    hpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hpo)
    experiment_path = hpo.generate(args.sweep, output_root=args.output.resolve())
    experiment = yaml.safe_load(experiment_path.read_text())
    models_path = experiment_path.parent / "models.yml"
    models = yaml.safe_load(models_path.read_text())["models"]
    attack_base = yaml.safe_load(args.attack_base.read_text())
    selected = []
    for defense in list(experiment["models"]):
        attack = defense + "--relearn-fft-ms250"
        config = copy.deepcopy(attack_base)
        config.pop("revision_of_model", None)
        config.update(
            base_model=models[defense]["base_model_name_or_path"],
            output_dir=f"artifacts/mila/models/{attack}",
            wandb_name=attack,
            dataset_prepared_path=f"artifacts/mila/cache/axolotl/{attack}/prepared",
        )
        path = experiment_path.parent / "configs" / (attack + ".yml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        config_path = hpo.project_path(path, Path.cwd())
        models[attack] = {
            "base_model_name_or_path": config["output_dir"],
            "baseline_model_name_or_path": config["base_model"],
            "adapter_name_or_path": "-",
            "method": defense,
            "attack_learning_rate": str(config["learning_rate"]),
            "attack_data": "overlapping",
            "producer": {
                "id": "train-" + attack,
                "script": "scripts/slurm/mila/train.sbatch",
                "args": [config_path],
                "depends_on": [defense],
                "sbatch_args": [
                    "--partition=main",
                    "--gres=gpu:l40s:1",
                    "--cpus-per-task=4",
                    "--mem=24G",
                    "--time=03:00:00",
                ],
            },
        }
        selected.append(attack)
    models_path.write_text(yaml.safe_dump({"models": models}, sort_keys=False))
    experiment["models"] = selected
    experiment["evaluations"] = ["mila-l40s-q06b--beavertails-harmfulness-trajectory"]
    experiment_path.write_text(yaml.safe_dump(experiment, sort_keys=False))
    print(experiment_path)


if __name__ == "__main__":
    main()
