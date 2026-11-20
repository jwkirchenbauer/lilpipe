"""Keep the production schedule while stopping isolated K4/K256 pilot prefixes."""

import copy
import hashlib
import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]
ORIGIN = Path(
    "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z"
)


def main():
    root = ROOT / "pilots/cuda-training"
    assert not root.exists()
    root.mkdir(parents=True)
    (root / "status").mkdir()
    plans = {}
    for k, stop in ((4, 4), (256, 1)):
        source = ORIGIN / "cases" / f"rio-k{k}-defense/config.yml"
        original = yaml.safe_load(source.read_text())
        config = copy.deepcopy(original)
        folder = root / f"k{k}"
        folder.mkdir()
        config.update(
            output_dir=str(folder / "models/defense"),
            trainer_cls="training_pilot_observer.MatrixPilotTrainer",
            wandb_project="jwk-tar-matrix-runtime-pilot",
            wandb_name=f"cuda-pilot-k{k}",
        )
        assert config["max_steps"] == 125 and config["warmup_steps"] == 12
        assert config["custom_trainer_args"]["attack_max_steps"] == k
        changed = {
            key: {"original": original.get(key), "pilot": value}
            for key, value in config.items()
            if original.get(key) != value
        }
        assert set(changed) == {
            "output_dir",
            "trainer_cls",
            "wandb_project",
            "wandb_name",
        }
        (folder / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        plans[str(k)] = {
            "stop_step": stop,
            "max_steps": 125,
            "warmup_steps": 12,
            "source_config": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "config_changes": changed,
            "trainer_note": "Subclass adds only an on_train_end peak-memory observer; existing frozen stop callback preserves the original schedule",
        }
    (root / "plan.json").write_text(json.dumps(plans, indent=2) + "\n")


if __name__ == "__main__":
    main()
