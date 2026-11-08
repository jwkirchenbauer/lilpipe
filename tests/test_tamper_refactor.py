import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_z7b_ghost_comparison_generates_matched_gd_npo_attacks(monkeypatch, tmp_path):
    monkeypatch.chdir(EXAMPLE)
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "hpo_ghost_comparison")
    manifest = hpo.generate(
        EXAMPLE
        / "configs/stages-wmdp/2_relearn/z7b-wmdp-bio--relearn-fft-ghost-ms128/sweep.yml",
        output_root=tmp_path,
    )
    experiment = yaml.safe_load(manifest.read_text())
    assert len(experiment["models"]) == 2
    configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "configs").glob("*.yml"))
    ]
    assert {cfg["base_model"].rsplit("/", 1)[-1] for cfg in configs} == {"gd", "npo"}
    assert len({cfg["wandb_run_id"] for cfg in configs}) == 2
    for cfg in configs:
        assert cfg["trainer_cls"] == "trainers.tr_gip.GIPRelearnTrainer"
        assert cfg["micro_batch_size"] >= 2
        assert not cfg["gradient_checkpointing"] and not cfg["sample_packing"]
        assert cfg["device_map"] == {"": 0}
        assert cfg["max_steps"] == 128
        assert cfg["save_strategy"] == "no"
    for key in ("base_model", "output_dir", "wandb_name", "wandb_run_id"):
        for cfg in configs:
            cfg.pop(key)
    assert configs[0] == configs[1]


def test_custom_trainer_plugin_validates_and_forwards(monkeypatch) -> None:
    pytest.importorskip("pydantic")
    axolotl = types.ModuleType("axolotl")
    integrations = types.ModuleType("axolotl.integrations")
    base = types.ModuleType("axolotl.integrations.base")

    class BasePlugin:
        pass

    base.BasePlugin = BasePlugin
    monkeypatch.setitem(sys.modules, "axolotl", axolotl)
    monkeypatch.setitem(sys.modules, "axolotl.integrations", integrations)
    monkeypatch.setitem(sys.modules, "axolotl.integrations.base", base)
    plugin_module = load_script(
        EXAMPLE / "src/axolotl_plugins/custom_trainer_args.py", "custom_args_test"
    )

    class Trainer:
        def __init__(self, model=None, *, rho=0.01):
            self.model = model
            self.rho = rho

    configured = plugin_module.trainer_with_custom_args(Trainer, {"rho": 0.2})
    trainer = configured(model="model")
    assert trainer.rho == 0.2
    assert trainer.model == "model"
    with pytest.raises(ValueError, match="unknown"):
        plugin_module.trainer_with_custom_args(Trainer, {"missing": 1})
    with pytest.raises(ValueError, match="reserved"):
        plugin_module.trainer_with_custom_args(Trainer, {"model": "other"})
    with pytest.raises(ValueError, match="collision"):
        configured(model="model", rho=0.3)


def test_hpo_expansion_is_deterministic_and_supports_dotted_paths(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump(
            {
                "output_dir": "old",
                "learning_rate": 1e-5,
                "custom_trainer_args": {"rho": 0.01},
            }
        )
    )
    sweep = {
        "id": "test-sweep",
        "sweeps": [
            {
                "name": "grid",
                "parameters": {
                    "learning_rate": [1e-5, 2e-5],
                    "custom_trainer_args.rho": [0.1, 0.2],
                },
            }
        ],
        "producer": {"script": "train.sbatch"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    experiment_path = hpo.generate(sweep_path, output)
    first = sorted((output / "configs").glob("*.yml"))
    assert [path.stem for path in first] == [
        f"test-sweep--grid--{index:03d}" for index in range(1, 5)
    ]
    configs = [yaml.safe_load(path.read_text()) for path in first]
    assert [
        (item["learning_rate"], item["custom_trainer_args"]["rho"]) for item in configs
    ] == [
        (1e-5, 0.1),
        (2e-5, 0.1),
        (1e-5, 0.2),
        (2e-5, 0.2),
    ]
    assert [item["wandb_name"] for item in configs] == [path.stem for path in first]
    assert yaml.safe_load(experiment_path.read_text())["models"] == [
        f"test-sweep--grid--{index:03d}" for index in range(1, 5)
    ]


def test_hpo_can_use_selected_config_and_expose_parameters_to_model_fields(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_selected_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "selected.yml").write_text(
        yaml.safe_dump({"base_model": "old", "output_dir": "old-output"})
    )
    sweep = {
        "id": "test-relearn-0.6b",
        "base": "selected.yml",
        "sweeps": [
            {
                "name": "models",
                "parameters": {"base_model": ["model-a", "model-b"]},
            }
        ],
        "producer": {"script": "train.sbatch"},
        "model": {"baseline_model_name_or_path": "{parameter[base_model]}"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    experiment_path = hpo.generate(sweep_path, output)
    assert yaml.safe_load(experiment_path.read_text())["id"] == "test-relearn-0.6b"
    configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((output / "configs").glob("*.yml"))
    ]
    assert [config["base_model"] for config in configs] == ["model-a", "model-b"]
    models_path = Path(
        yaml.safe_load(experiment_path.read_text())["registries"]["models"]
    )
    models = yaml.safe_load(models_path.read_text())["models"]
    assert [model["baseline_model_name_or_path"] for model in models.values()] == [
        "model-a",
        "model-b",
    ]


def test_hpo_concatenates_named_sweeps_and_applies_stage_overrides(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_stages_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump(
            {
                "output_dir": "old",
                "learning_rate": 1e-5,
                "custom_trainer_args": {"rho": 0.01},
            }
        )
    )
    sweep = {
        "id": "test-stages",
        "sweeps": [
            {
                "name": "learning-rate",
                "parameters": {"learning_rate": [2e-5, 3e-5]},
            },
            {
                "name": "rho",
                "parameters": {"custom_trainer_args.rho": [0.1, 0.2]},
                "overrides": {"learning_rate": 4e-5},
            },
        ],
        "producer": {"script": "train.sbatch"},
        "evaluations": [],
    }
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    monkeypatch.chdir(EXAMPLE)

    output = tmp_path / "generated"
    hpo.generate(sweep_path, output)
    configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in sorted((output / "configs").glob("*.yml"))
    }

    assert list(configs) == [
        "test-stages--learning-rate--001",
        "test-stages--learning-rate--002",
        "test-stages--rho--001",
        "test-stages--rho--002",
    ]
    assert configs["test-stages--rho--001"]["learning_rate"] == 4e-5


@pytest.mark.parametrize(
    ("sweeps", "message"),
    [
        (None, "sweeps must be a non-empty list"),
        ([], "sweeps must be a non-empty list"),
        (
            [
                {"name": "same", "parameters": {"learning_rate": [1e-5]}},
                {"name": "same", "parameters": {"learning_rate": [2e-5]}},
            ],
            "sweep names must be unique",
        ),
    ],
)
def test_hpo_rejects_invalid_sweep_lists(
    tmp_path: Path, monkeypatch, sweeps, message: str
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_invalid_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump({"output_dir": "old", "learning_rate": 1e-5})
    )
    sweep_path = lineage / "sweep.yml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "id": "test-invalid",
                "sweeps": sweeps,
                "producer": {"script": "train.sbatch"},
            }
        )
    )
    monkeypatch.chdir(EXAMPLE)

    with pytest.raises(ValueError, match=message):
        hpo.generate(sweep_path, tmp_path / "generated")


def test_hpo_rejects_top_level_parameters_and_duplicate_configs(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tamper_hpo_duplicates_test")
    lineage = tmp_path / "lineage"
    lineage.mkdir()
    (lineage / "base.yml").write_text(
        yaml.safe_dump({"output_dir": "old", "learning_rate": 1e-5})
    )
    sweep_path = lineage / "sweep.yml"
    common = {"id": "test-invalid", "producer": {"script": "train.sbatch"}}
    monkeypatch.chdir(EXAMPLE)

    sweep_path.write_text(
        yaml.safe_dump({**common, "parameters": {"learning_rate": [2e-5]}})
    )
    with pytest.raises(ValueError, match="top-level sweep fields"):
        hpo.generate(sweep_path, tmp_path / "legacy")

    sweep_path.write_text(
        yaml.safe_dump(
            {
                **common,
                "sweeps": [
                    {"name": "first", "parameters": {"learning_rate": [2e-5]}},
                    {"name": "second", "parameters": {"learning_rate": [2e-5]}},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate generated configuration"):
        hpo.generate(sweep_path, tmp_path / "duplicates")


def test_only_canonical_registries_and_selected_configs_are_referenced() -> None:
    registries = EXAMPLE / "configs/registries"
    assert {path.name for path in registries.glob("*.yml")} == {
        "evals.yml",
        "models.yml",
    }
    models = yaml.safe_load((registries / "models.yml").read_text())["models"]
    for model in models.values():
        producer = model.get("producer")
        if producer is None:
            continue
        if producer["script"] != "scripts/slurm/tamia/train.sbatch":
            assert (EXAMPLE / producer["script"]).is_file()
            continue
        config = producer["args"][0]
        assert config.endswith("/selected.yml")
        assert (EXAMPLE / config).is_file()


def test_wmdp_landscape_attacks_retain_matching_trajectories(monkeypatch) -> None:
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    plan = (
        load("configs/experiments/z7b-wmdp-bio-relearn-128--5pp.yml")
        .select(
            models=[
                "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr5e-6-ms128",
                "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft-lr5e-6-ms128",
            ],
        )
        .plan()
    )
    assert len(plan.stages) == 4  # Two attacks and two paired trajectory jobs.
    training = [stage for stage in plan.stages if stage.id.startswith("train-")]
    evaluations = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    assert len(training) == 2
    assert len(evaluations) == 2
    for stage in evaluations:
        assert Path(stage.script).name == "eval_wmdp_mmlu.sbatch"
        assert not any(arg.startswith("--array=") for arg in stage.sbatch_args)
        model = Path(stage.args[1]).name
        assert stage.args == (
            "--checkpoints",
            f"artifacts/tamia/models/{model}",
            f"artifacts/tamia/evals/{model}",
        )  # Removal flags are omitted, so both default to false.
        assert stage.depends_on == (f"train-{model}",)
    configs = []
    for stage in training:
        assert Path(stage.script).name == "train.sbatch"
        assert stage.args[-2:] == ("--launcher", "python")
        config = yaml.safe_load((EXAMPLE / stage.args[0]).read_text())
        assert config["learning_rate"] == 5e-6
        assert config["max_steps"] == 128
        assert config["save_steps"] == 16
        assert config["save_total_limit"] >= (
            config["max_steps"] // config["save_steps"]
        )
        assert config["save_only_model"] is True
        assert config["output_dir"].endswith("--relearn-fft-lr5e-6-ms128")
        configs.append(config)
    assert {Path(config["base_model"]).name for config in configs} == {
        "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16",
        "z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2",
    }
    distinct_fields = {
        "base_model",
        "output_dir",
        "dataset_prepared_path",
        "wandb_name",
    }
    assert {k: v for k, v in configs[0].items() if k not in distinct_fields} == {
        k: v for k, v in configs[1].items() if k not in distinct_fields
    }


def test_qwen_seam_evaluation_waits_for_training(monkeypatch) -> None:
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    pipeline = load("configs/experiments/qwen3-06b-beavertail.yml")
    training, evaluation = pipeline.plan().stages
    assert evaluation.depends_on == (training.id,)
    config = yaml.safe_load((EXAMPLE / training.args[0]).read_text())
    assert evaluation.args[1] == config["output_dir"]
    assert evaluation.args[2] == "SEAM"
    assert all(
        "--gres=gpu:l40s:1" in stage.sbatch_args for stage in (training, evaluation)
    )
    reused = pipeline.plan(existing_models=pipeline.selected_models)
    assert reused.skipped == frozenset({training.id})


@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "seam"])
def test_qwen_relearning_grid_matches_defense_betas_and_evaluates_attacks(
    legacy: bool, tmp_path: Path, monkeypatch
) -> None:
    from lilpipe import load

    hpo = load_script(EXAMPLE / "scripts/hpo.py", "qwen_relearn_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    name = "qwen3-0.6b-beavertail--align-fft-seam" + ("-legacy" if legacy else "")
    defense_manifest = hpo.generate(
        EXAMPLE / "configs/stages-beavertails/0_alignment" / name / "sweep.yml",
        tmp_path / "defense",
    )
    defense_pipeline = load(defense_manifest)
    defenses = [
        stage
        for stage in defense_pipeline.plan().stages
        if stage.id.startswith("train-")
    ]
    defense_configs = [
        yaml.safe_load(Path(stage.args[0]).read_text()) for stage in defenses
    ]
    coefficient = "beta" if legacy else "coeff_sd"
    assert [
        config["custom_trainer_args"][coefficient] for config in defense_configs
    ] == [
        0.01,
        0.1,
        1.0,
    ]
    if not legacy:
        for config in defense_configs:
            assert config["trainer_cls"] == "trainers.tr_seam.SEAMTrainer"
            assert "device_map" not in config
            assert config["base_model"] == "Qwen/Qwen3-0.6B"
            assert [
                dataset["message_property_mappings"]["role"]
                for dataset in config["datasets"][:2]
            ] == ["hh", "hb"]
            assert all(
                dataset["message_property_mappings"]["train_on_inputs"] == "false"
                for dataset in config["datasets"][:2]
            )
    manifest = hpo.generate(
        EXAMPLE / "configs/stages-beavertails/1_relearn" / name / "sweep.yml",
        tmp_path / "attack",
    )
    plan = load(manifest).plan()
    training = [stage for stage in plan.stages if stage.id.startswith("train-")]
    evaluation = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    configs = [yaml.safe_load(Path(stage.args[0]).read_text()) for stage in training]
    assert [stage.id for stage in training] == [
        f"train-{name}--relearn--hpo--beta{beta}--{index:03d}"
        for beta in (0.01, 0.1, 1.0)
        for index in (1, 2)
    ]
    assert [(config["base_model"], config["learning_rate"]) for config in configs] == [
        (defense["output_dir"], lr)
        for defense in defense_configs
        for lr in (5e-5, 1e-4)
    ]
    for config, train, evaluate in zip(configs, training, evaluation, strict=True):
        assert train.depends_on == ()
        assert evaluate.args[1] == config["base_model"]
        assert config["max_steps"] == 250
        assert config["save_steps"] == 100
        if legacy:
            assert config["datasets"][0]["message_property_mappings"]["role"] == (
                "attack_disjoint"
            )
        else:
            assert config["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
            assert config["custom_trainer_args"] == {
                "n_hh": 1000,
                "n_hb": 1000,
                "n_bb": 1000,
            }
            assert config["micro_batch_size"] == 12
            assert config["train_on_inputs"] is False
            assert [
                dataset["message_property_mappings"]["offset"]
                for dataset in config["datasets"][:2]
            ] == ["8000", "8000"]
        assert evaluate.depends_on == (train.id,)
        assert evaluate.args[2] == config["output_dir"]
        assert len(evaluate.args) == 7
        assert evaluate.args[4] == "disjoint"
    assert len(training) == len(evaluation) == 6


def test_active_manifests_reference_only_registered_evaluations() -> None:
    registered = set(
        yaml.safe_load((EXAMPLE / "configs/registries/evals.yml").read_text())[
            "evaluations"
        ]
    )
    referenced = set()
    paths = list((EXAMPLE / "configs/experiments").glob("*.yml"))
    paths.extend(EXAMPLE.glob("configs/stages-*/**/sweep.yml"))
    for path in paths:
        referenced.update(yaml.safe_load(path.read_text()).get("evaluations", ()))

    assert referenced <= registered
    assert referenced == registered


def test_seam_defense_sweep_preserves_beta_and_paper_settings(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "seam_defense_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    directory = (
        EXAMPLE / "configs/stages-beavertails/0_alignment/"
        "llama2-7b-beavertail--align-fft-seam"
    )
    base = yaml.safe_load((directory / "base.yml").read_text())
    selected = yaml.safe_load((directory / "selected.yml").read_text())
    base["custom_trainer_args"]["coeff_sd"] = 1.0
    assert base == selected

    output = tmp_path / "generated"
    manifest = hpo.generate(directory / "sweep.yml", output)
    experiment = yaml.safe_load(manifest.read_text())
    models = yaml.safe_load((output / "models.yml").read_text())["models"]
    assert experiment["evaluations"] == [
        "tamia-h100-llama2-7b--beavertails-and-capabilities-single"
    ]
    assert len(models) == 4
    assert {model["method"] for model in models.values()} == {"SEAM"}
    paper_id = "llama2-7b-beavertail--align-fft-seam--paper-config"
    paper = yaml.safe_load((output / "configs" / f"{paper_id}.yml").read_text())
    assert paper["trainer_cls"] == "trainers.tr_seam.SEAMTrainer"
    assert paper["learning_rate"] == 2e-5
    assert paper["warmup_ratio"] == 0.1
    assert paper["micro_batch_size"] == 24
    assert paper["max_steps"] == 500
    assert paper["datasets"][2]["split"] == "train"
    assert "paired_token_mask" not in paper["custom_trainer_args"]
    assert "ignore_label_token_id" not in paper["custom_trainer_args"]

    configs = [
        yaml.safe_load(p.read_text()) for p in (output / "configs").glob("*.yml")
    ]
    assert len({config["output_dir"] for config in configs}) == 4
    assert {config["custom_trainer_args"]["coeff_sd"] for config in configs} == {
        0.01,
        0.5,
    }
    for suffix, streams in [
        ("--refusal-space", [1]),
        ("--harmful-refusal-space", [0, 1]),
    ]:
        slug = paper_id + suffix
        config = yaml.safe_load((output / "configs" / f"{slug}.yml").read_text())
        for stream in streams:
            mapping = config["datasets"][stream]["message_property_mappings"]
            assert mapping["response_separator"] == " "
            mapping["response_separator"] = "\n"
        for key in ["output_dir", "wandb_name", "dataset_prepared_path"]:
            config[key] = paper[key]
        assert config == paper


def test_beavertail_trajectory_sweep_uses_disjoint_attack_data(
    tmp_path: Path, monkeypatch
) -> None:
    attack_data = "disjoint"
    offset = "8000"
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "beavertail_trajectory_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    defense = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/0_alignment/"
            "llama2-7b-beavertail--align-fft-seam/selected.yml"
        ).read_text()
    )
    assert defense["max_steps"] == 500
    assert defense["micro_batch_size"] == 24
    assert set(defense["device_map"].values()) == {0, 1, 2, 3}
    assert defense["output_dir"].endswith("--align-fft-seam")
    sweep_path = (
        EXAMPLE / "configs/stages-beavertails/1_relearn/"
        f"llama2-7b-beavertail--align-fft-seam--data-{attack_data}--"
        "relearn-fft-ms250/sweep.yml"
    )
    experiment_path = hpo.generate(sweep_path, tmp_path / attack_data)
    experiment = yaml.safe_load(experiment_path.read_text())
    registry = yaml.safe_load((tmp_path / attack_data / "models.yml").read_text())[
        "models"
    ]
    attacks = [registry[model_id] for model_id in experiment["models"]]
    generated_configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / attack_data / "configs").glob("*.yml"))
    ]

    assert len(attacks) == 2
    assert experiment["evaluations"] == [
        "tamia-h100-llama2-7b--beavertails-and-capabilities-trajectory"
    ]
    assert {model["attack_learning_rate"] for model in attacks} == {
        "5e-05",
        "0.0001",
    }
    assert {model["baseline_model_name_or_path"] for model in attacks} == {
        "artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam"
    }
    assert {model["attack_data"] for model in attacks} == {attack_data}
    assert {model["method"] for model in attacks} == {"SEAM"}
    assert {model["evaluation_checkpoint_steps"] for model in attacks} == {
        "0,100,200,250"
    }
    assert {model["evaluation_final_step"] for model in attacks} == {"250"}
    assert {
        config["datasets"][0]["message_property_mappings"]["offset"]
        for config in generated_configs
    } == {offset}
    assert all(
        config["datasets"][0]["type"] == "data.beavertails_hh"
        and config["datasets"][0]["message_property_mappings"]["role"] == "hh"
        and config["train_on_inputs"] is False
        for config in generated_configs
    )
    assert {config["dataset_prepared_path"] for config in generated_configs} == {
        "artifacts/tamia/cache/axolotl/"
        f"llama2-7b-beavertail--align-fft-seam--data-{attack_data}--"
        "relearn-fft-ms250/prepared"
    }
    assert {config["trainer_cls"] for config in generated_configs} == {
        "trainers.tr_seam.SEAMRelearnTrainer"
    }
    assert {config["micro_batch_size"] for config in generated_configs} == {12}
    assert {config["save_steps"] for config in generated_configs} == {100}
    assert {config["save_total_limit"] for config in generated_configs} == {3}
    assert all(
        config["datasets"][1]["message_property_mappings"]["offset"] == offset
        for config in generated_configs
    )
    assert all(
        model["producer"]["script"] == "scripts/slurm/tamia/train.sbatch"
        and model["producer"]["args"][-3:] == ["--single-gpu", "--launcher", "python"]
        for model in attacks
    )
    assert all(
        model["producer"]["depends_on"] == ["llama2-7b-beavertail--align-fft-seam"]
        for model in attacks
    )


def test_beavertail_base_disjoint_sweep_attacks_the_undefended_chat_model(
    tmp_path: Path, monkeypatch
) -> None:
    attack_data = "disjoint"
    offset = "8000"
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "beavertail_base_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    sweep_path = (
        EXAMPLE / "configs/stages-beavertails/1_relearn/"
        f"llama2-7b-beavertail--data-{attack_data}--relearn-fft-ms250/sweep.yml"
    )
    experiment_path = hpo.generate(sweep_path, tmp_path / "generated")
    experiment = yaml.safe_load(experiment_path.read_text())
    registry = yaml.safe_load((tmp_path / "generated/models.yml").read_text())["models"]
    attacks = [registry[model_id] for model_id in experiment["models"]]
    generated_configs = [
        yaml.safe_load(path.read_text())
        for path in sorted((tmp_path / "generated/configs").glob("*.yml"))
    ]

    assert len(attacks) == 2
    assert {model["attack_learning_rate"] for model in attacks} == {
        "5e-05",
        "0.0001",
    }
    assert {model["baseline_model_name_or_path"] for model in attacks} == {
        "meta-llama/Llama-2-7b-chat-hf"
    }
    assert {model["attack_data"] for model in attacks} == {attack_data}
    assert {model["method"] for model in attacks} == {"Base"}
    assert {model["evaluation_checkpoint_steps"] for model in attacks} == {
        "0,100,200,250"
    }
    assert {model["evaluation_final_step"] for model in attacks} == {"250"}
    assert all("depends_on" not in model["producer"] for model in attacks)
    assert {config["base_model"] for config in generated_configs} == {
        "meta-llama/Llama-2-7b-chat-hf"
    }
    assert {config["save_steps"] for config in generated_configs} == {100}
    assert {config["save_total_limit"] for config in generated_configs} == {3}
    assert {
        config["datasets"][0]["message_property_mappings"]["offset"]
        for config in generated_configs
    } == {offset}
    assert all(
        config["datasets"][0]["type"] == "data.beavertails_hh"
        and config["datasets"][0]["message_property_mappings"]["role"] == "hh"
        and config["datasets"][1]["type"] == "data.beavertails_hh"
        and config["datasets"][1]["message_property_mappings"]["role"] == "hb"
        and config["datasets"][1]["message_property_mappings"]["offset"] == offset
        and config["train_on_inputs"] is False
        and config["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
        and config["micro_batch_size"] == 12
        for config in generated_configs
    )


def test_beavertail_ms1000_sweep_evaluates_the_full_trajectory(
    tmp_path: Path, monkeypatch
) -> None:
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "beavertail_ms1000_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    sweep = (
        EXAMPLE / "configs/stages-beavertails/1_relearn/"
        "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
        "relearn-fft-ms1000/sweep.yml"
    )
    experiment_path = hpo.generate(sweep, tmp_path / "generated")
    experiment = yaml.safe_load(experiment_path.read_text())
    models = yaml.safe_load((tmp_path / "generated/models.yml").read_text())["models"]
    configs = [
        yaml.safe_load(path.read_text())
        for path in (tmp_path / "generated/configs").glob("*.yml")
    ]

    assert experiment["evaluations"] == [
        "tamia-h100-llama2-7b--beavertails-and-capabilities-trajectory"
    ]
    assert {
        models[model_id]["evaluation_checkpoint_steps"]
        for model_id in experiment["models"]
    } == {",".join(map(str, range(0, 1001, 100)))}
    assert {
        models[model_id]["evaluation_final_step"] for model_id in experiment["models"]
    } == {"1000"}
    assert {config["save_steps"] for config in configs} == {100}
    assert {config["save_total_limit"] for config in configs} == {11}


def test_retired_perturbation_sweep_preserves_research_settings() -> None:
    retired = yaml.safe_load(
        (EXAMPLE / "docs/retired-tamia/perturb-sweep.yml").read_text()
    )
    assert "producer" not in retired
    assert "script" not in retired["retired_producer"]
    values = {
        sigma
        for stage in retired["sweeps"]
        for sigma in stage["parameters"]["gaussian_sigma"]
    }
    assert values == {1e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 1e-3}
    assert not (
        EXAMPLE
        / "configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-perturb/sweep.yml"
    ).exists()


def test_seam_paper_defense_uses_four_gpu_layer_sharding() -> None:
    model = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]["llama2-7b-beavertail--align-fft-seam"]
    producer = model["producer"]
    assert producer["script"] == "scripts/slurm/tamia/train.sbatch"
    assert producer["args"] == [
        "configs/stages-beavertails/0_alignment/"
        "llama2-7b-beavertail--align-fft-seam/selected.yml",
        "--launcher",
        "python",
    ]
    assert "--gpus-per-node=h100:4" in producer["sbatch_args"]
    assert "--time=03:00:00" in producer["sbatch_args"]

    script = (EXAMPLE / "scripts/slurm/tamia/train.sbatch").read_text()
    assert "axolotl train" in script
    assert "scontrol requeue" not in script

    config = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/0_alignment/"
            "llama2-7b-beavertail--align-fft-seam/selected.yml"
        ).read_text()
    )
    assert config["micro_batch_size"] == 24
    assert set(config["device_map"].values()) == {0, 1, 2, 3}
    for index in range(32):
        assert config["device_map"][f"model.layers.{index}"] == index // 8


def test_llama2_selected_seam_pipeline_runs_defense_attack_and_evaluation(
    monkeypatch,
) -> None:
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    pipeline = load("configs/experiments/llama2-7b-beavertail.yml")
    defense, attack, evaluation = pipeline.plan().stages

    assert defense.id == "train-llama2-7b-beavertail--align-fft-seam-legacy"
    assert attack.depends_on == (defense.id,)
    assert evaluation.depends_on == (attack.id,)
    assert defense.args == (
        "configs/stages-beavertails/0_alignment/"
        "llama2-7b-beavertail--align-fft-seam-legacy/selected.yml",
        "--launcher",
        "python",
    )
    assert attack.args == (
        "configs/stages-beavertails/1_relearn/"
        "llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--"
        "relearn-fft-ms250/selected.yml",
        "--single-gpu",
        "--launcher",
        "python",
    )
    assert evaluation.args[3:4] + evaluation.args[-2:-1] == (
        "artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-legacy",
        "artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-legacy--"
        "data-disjoint--relearn-fft-ms250",
    )
    assert evaluation.args[1] == "0,100,200,250"
    assert evaluation.args[-1].endswith("data-disjoint--relearn-fft-ms250")
    assert "--export=ALL,PYTHONHASHSEED=0" in defense.sbatch_args
    assert all(
        "--gpus-per-node=h100:4" in stage.sbatch_args
        and "--time=03:00:00" in stage.sbatch_args
        for stage in (defense, attack, evaluation)
    )


def test_legacy_seam_relearning_grid_matches_defense_seeds(
    tmp_path: Path, monkeypatch
) -> None:
    from lilpipe import load

    hpo = load_script(EXAMPLE / "scripts/hpo.py", "legacy_seed_hpo_test")
    monkeypatch.chdir(EXAMPLE)
    defense_name = "llama2-7b-beavertail--align-fft-seam-legacy"
    attack_name = f"{defense_name}--data-disjoint--relearn-fft-ms250"
    defense_manifest = hpo.generate(
        EXAMPLE / "configs/stages-beavertails/0_alignment" / defense_name / "sweep.yml",
        tmp_path / "defense",
    )
    defense_pipeline = load(defense_manifest)
    defenses = [
        stage
        for stage in defense_pipeline.plan().stages
        if stage.id.startswith("train-")
    ]
    defense_configs = [
        yaml.safe_load(Path(stage.args[0]).read_text()) for stage in defenses
    ]
    assert [config["seed"] for config in defense_configs] == [40, 41, 42]
    assert [config["data_seed"] for config in defense_configs] == [40, 41, 42]
    assert not any(
        config["custom_trainer_args"]["fp32_accumulation"] for config in defense_configs
    )

    defense_sweep = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-beavertails/0_alignment"
            / defense_name
            / "sweep.yml"
        ).read_text()
    )
    assert not any(
        "PYTHONHASHSEED" in argument
        for argument in defense_sweep["producer"]["sbatch_args"]
    )

    selected_defense = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-beavertails/0_alignment"
            / defense_name
            / "selected.yml"
        ).read_text()
    )
    selected_attack = yaml.safe_load(
        (
            EXAMPLE
            / "configs/stages-beavertails/1_relearn"
            / attack_name
            / "selected.yml"
        ).read_text()
    )
    assert selected_attack["base_model"] == selected_defense["output_dir"]
    assert selected_attack["output_dir"].endswith(attack_name)
    assert selected_attack["wandb_name"] == attack_name

    attack_manifest = hpo.generate(
        EXAMPLE / "configs/stages-beavertails/1_relearn" / attack_name / "sweep.yml",
        tmp_path / "attack",
    )
    plan = load(attack_manifest).plan()
    training = [stage for stage in plan.stages if stage.id.startswith("train-")]
    evaluations = [stage for stage in plan.stages if stage.id.startswith("eval-")]
    attack_configs = [
        yaml.safe_load(Path(stage.args[0]).read_text()) for stage in training
    ]
    assert [stage.id for stage in training] == [
        f"train-{attack_name}--hpo--seed{seed}--001" for seed in (40, 41, 42)
    ]
    assert [
        (config["base_model"], config["learning_rate"]) for config in attack_configs
    ] == [(config["output_dir"], 1e-4) for config in defense_configs]
    assert {config["trainer_cls"] for config in attack_configs} == {
        "trainers.rl_bf16.BF16Trainer"
    }
    assert {config["micro_batch_size"] for config in attack_configs} == {4}
    assert {
        tuple(
            dataset["message_property_mappings"]["role"]
            for dataset in config["datasets"]
        )
        for config in attack_configs
    } == {("attack_disjoint",)}
    for config, train, evaluation in zip(
        attack_configs, training, evaluations, strict=True
    ):
        assert train.depends_on == ()
        assert evaluation.depends_on == (train.id,)
        assert evaluation.args[3] == config["base_model"]
        assert evaluation.args[-2] == config["output_dir"]


LINEAGES = sorted(
    {
        path.parent
        for name in ("base.yml", "selected.yml", "sweep.yml")
        for path in EXAMPLE.glob(f"configs/stages-*/*/*/{name}")
    }
)


def differing_keys(first: dict, second: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key in first.keys() | second.keys():
        path = f"{prefix}{key}"
        a, b = first.get(key), second.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys |= differing_keys(a, b, f"{path}.")
        elif a != b:
            keys.add(path)
    return keys


@pytest.mark.parametrize("lineage", LINEAGES, ids=lambda path: path.name)
def test_lineage_configs_follow_conventions(lineage: Path) -> None:
    name = lineage.name
    configs = {
        stem: yaml.safe_load((lineage / f"{stem}.yml").read_text())
        for stem in ("base", "selected")
        if (lineage / f"{stem}.yml").is_file()
    }
    for config in configs.values():
        if "output_dir" in config:
            assert config["output_dir"].rstrip("/").endswith(f"/{name}")
        if "wandb_name" in config:
            assert config["wandb_name"] == name
        if "trainer_cls" in config:
            module, _, cls = config["trainer_cls"].rpartition(".")
            source = EXAMPLE / "src" / f"{module.replace('.', '/')}.py"
            assert source.is_file(), config["trainer_cls"]
            assert re.search(
                rf"^\s*class {re.escape(cls)}\(",
                source.read_text(),
                re.MULTILINE,
            ), config["trainer_cls"]
            if module == "trainers.tr_wd":
                assert "anchor_model_name_or_path" in config["custom_trainer_args"]

    sweep_path = lineage / "sweep.yml"
    if not sweep_path.is_file():
        return
    sweep = yaml.safe_load(sweep_path.read_text())
    base_stem = Path(sweep.get("base", "base.yml")).stem
    assert base_stem in configs
    expected_ids = {f"{name}--hpo"}
    if lineage.parent.name == "1_relearn":
        expected_ids.add(f"{name}--relearn--hpo")
    assert sweep["id"] in expected_ids
    if re.search(r"--(\d+pp)--", name):
        for evaluation in sweep.get("evaluations", ()):
            assert evaluation in {
                "mila-l40s-q06b--wmdp-and-mmlu-final",
                "mila-l40s-q06b--wmdp-and-mmlu-single",
                "tamia-h100-z7b--wmdp-and-mmlu-final",
                "tamia-h100-z7b--wmdp-and-mmlu-single",
                "tamia-h100-z7b--wmdp-and-mmlu-trajectory",
            }
    if "selected" in configs:
        swept = {
            key
            for stage in sweep["sweeps"]
            for key in {**stage.get("parameters", {}), **stage.get("overrides", {})}
        }
        allowed = swept | {
            "output_dir",
            "wandb_name",
            "dataset_prepared_path",
        }
        if configs["selected"].get("trainer_cls") in {
            "trainers.rl_cosim.CosineTrackedBF16Trainer",
            "trainers.tr_seam.SEAMRelearnTrainer",
        }:
            allowed |= {
                "custom_trainer_args",
                "datasets",
                "logging_steps",
                "micro_batch_size",
                "plugins",
                "remove_unused_columns",
                "train_on_inputs",
                "trainer_cls",
            }
        assert differing_keys(configs[base_stem], configs["selected"]) <= allowed


def test_tar_k16_lr_sweep_reuses_defense_and_evaluates_each_attack(
    tmp_path, monkeypatch
):
    from lilpipe import load

    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tar_k16_lr_hpo")
    name = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-ms128"
    defense = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
    monkeypatch.chdir(EXAMPLE)
    output = tmp_path / "generated"
    manifest = hpo.generate(
        EXAMPLE / f"configs/stages-wmdp/2_relearn/{name}/sweep.yml", output
    )
    candidates = [f"{name}--hpo--grid-001--{index:03d}" for index in range(1, 4)]
    configs = [
        yaml.safe_load((output / f"configs/{candidate}.yml").read_text())
        for candidate in candidates
    ]
    assert [config["learning_rate"] for config in configs] == [1e-6, 5e-6, 1e-5]
    for config in configs:
        assert config["datasets"] == [
            {
                "path": "cais/wmdp-bio-forget-corpus",
                "split": "train",
                "type": "data._base_document_strategy",
            }
        ]
        assert config["remove_unused_columns"] is True
        assert config["dataset_prepared_path"].endswith("/metadata-api/prepared")
    assert all(
        config["base_model"] == f"artifacts/tamia/models/{defense}"
        for config in configs
    )
    assert all(
        config["max_steps"] == 128 and config["save_steps"] == 16 for config in configs
    )
    pipeline = load(manifest)
    for candidate in candidates:
        plan = pipeline.select(models=[candidate]).plan(existing_models=[defense])
        rendered = plan.render()
        assert len(rendered.splitlines()) == 2
        assert f"train-{defense}" in plan.skipped
        assert f"train-{candidate}" in rendered
        assert "eval_wmdp_mmlu.sbatch" in rendered


def test_tar_k16_defense_generates_grid_and_low_retain_refinement(
    tmp_path, monkeypatch
):
    from lilpipe import load

    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tar_k16_numbered_hpo")
    name = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
    monkeypatch.chdir(EXAMPLE)
    manifest = hpo.generate(
        EXAMPLE / f"configs/stages-wmdp/0_unlearn/{name}/sweep.yml", tmp_path
    )
    expected = [f"{name}--hpo--grid-001--{index:03d}" for index in range(1, 10)]
    refinement = [f"{name}--hpo--grid-002--{index:03d}" for index in range(1, 3)]
    lower_attack_lr = [f"{name}--hpo--grid-003--{index:03d}" for index in range(1, 3)]
    higher_outer_lr = [f"{name}--hpo--grid-004--{index:03d}" for index in range(1, 3)]
    higher_retain = [f"{name}--hpo--grid-005--{index:03d}" for index in range(1, 4)]
    mid_retain = [f"{name}--hpo--grid-006--001"]
    assert (
        yaml.safe_load(manifest.read_text())["models"]
        == expected
        + refinement
        + lower_attack_lr
        + higher_outer_lr
        + higher_retain
        + mid_retain
    )
    pipeline = load(manifest)
    assert len(pipeline.plan().stages) == 38
    assert len(pipeline.select(models=refinement).plan().stages) == 4
    for candidate, retain in zip(refinement, [0.1, 0.01]):
        config = yaml.safe_load((tmp_path / f"configs/{candidate}.yml").read_text())
        assert config["learning_rate"] == 1e-6
        assert config["custom_trainer_args"]["retain_coefficient"] == retain
    assert len(pipeline.select(models=lower_attack_lr).plan().stages) == 4
    for previous, candidate in zip(refinement, lower_attack_lr):
        before = yaml.safe_load((tmp_path / f"configs/{previous}.yml").read_text())
        after = yaml.safe_load((tmp_path / f"configs/{candidate}.yml").read_text())
        assert before["custom_trainer_args"]["attack_learning_rate"] == 1e-5
        before["custom_trainer_args"]["attack_learning_rate"] = 1e-6
        for key in ("output_dir", "wandb_name"):
            before[key] = before[key].replace("grid-002", "grid-003")
        assert after == before
    assert len(pipeline.select(models=higher_outer_lr).plan().stages) == 4
    for previous, candidate in zip(lower_attack_lr, higher_outer_lr):
        before = yaml.safe_load((tmp_path / f"configs/{previous}.yml").read_text())
        after = yaml.safe_load((tmp_path / f"configs/{candidate}.yml").read_text())
        before["learning_rate"] = 5e-6
        for key in ("output_dir", "wandb_name"):
            before[key] = before[key].replace("grid-003", "grid-004")
        assert after == before
    assert len(pipeline.select(models=higher_retain).plan().stages) == 6
    for candidate, retain in zip(higher_retain, [1, 5, 10]):
        before = yaml.safe_load(
            (tmp_path / f"configs/{higher_outer_lr[0]}.yml").read_text()
        )
        after = yaml.safe_load((tmp_path / f"configs/{candidate}.yml").read_text())
        before["custom_trainer_args"]["retain_coefficient"] = retain
        before["output_dir"] = after["output_dir"]
        before["wandb_name"] = candidate
        assert after == before
    assert len(pipeline.select(models=mid_retain).plan().stages) == 2
    before = yaml.safe_load((tmp_path / f"configs/{higher_retain[1]}.yml").read_text())
    after = yaml.safe_load((tmp_path / f"configs/{mid_retain[0]}.yml").read_text())
    before["custom_trainer_args"]["retain_coefficient"] = 2.5
    before["output_dir"] = after["output_dir"]
    before["wandb_name"] = mid_retain[0]
    assert after == before
    candidates = [
        yaml.safe_load((tmp_path / f"configs/{candidate}.yml").read_text())
        for candidate in expected
    ]
    assert {
        (config["learning_rate"], config["custom_trainer_args"]["retain_coefficient"])
        for config in candidates
    } == {(lr, retain) for lr in (1e-6, 5e-6, 1e-5) for retain in (1, 5, 10)}
    for filename, learning_rate in (("base.yml", 5e-6), ("selected.yml", 5e-6)):
        config = yaml.safe_load(
            (EXAMPLE / f"configs/stages-wmdp/0_unlearn/{name}/{filename}").read_text()
        )
        assert config["custom_trainer_args"] == {
            "forget_coefficient": 1.0,
            "retain_coefficient": 2.5 if filename == "selected.yml" else 4.0,
            "defence_forget_batch_size": 4,
            "defence_retain_batch_size": 4,
            "attack_forget_batch_size": 16,
            "attack_forget_batch_size_per_iteration": 1,
            "attack_epochs": 1,
            "attack_learning_rate": 1e-6 if filename == "selected.yml" else 1e-5,
        }
        assert config["micro_batch_size"] == 24
        assert config["learning_rate"] == learning_rate
        assert config["max_steps"] == config["save_steps"] == 125
        assert "/metadata-api/prepared" in config["dataset_prepared_path"]
        assert [
            ds["message_property_mappings"]["metadata"] for ds in config["datasets"]
        ] == ["defence::forget", "attack::forget", "defence::retain"]
        assert all(
            ds["type"] == "data._base_document_strategy" for ds in config["datasets"]
        )
        assert all(ds["split"] == "train" for ds in config["datasets"][:2])


def test_tar_k16_ms250_keeps_initial_candidate_and_plans_search_grid(
    tmp_path, monkeypatch
):
    from lilpipe import load

    hpo = load_script(EXAMPLE / "scripts/hpo.py", "tar_k16_ms250_hpo")
    original = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
    name = f"{original}--ms250"
    root = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    previous = yaml.safe_load((root / original / "base.yml").read_text())
    base = yaml.safe_load((root / name / "base.yml").read_text())
    expected = dict(previous)
    expected.update(max_steps=250, save_steps=250)
    for key in ("output_dir", "wandb_name", "dataset_prepared_path"):
        expected[key] = previous[key].replace(original, name)
    for key in (
        "custom_trainer_args",
        "datasets",
        "micro_batch_size",
        "dataset_prepared_path",
    ):
        expected.pop(key)
    assert {key: base[key] for key in expected} == expected

    monkeypatch.chdir(EXAMPLE)
    output = tmp_path / "generated"
    manifest = hpo.generate(root / name / "sweep.yml", output)
    candidate = f"{name}--hpo--grid-001--001"
    configs = list((output / "configs").glob("*.yml"))
    assert len(configs) == 10
    config = yaml.safe_load((output / f"configs/{candidate}.yml").read_text())
    assert config["learning_rate"] == 1e-5
    assert config["custom_trainer_args"]["retain_coefficient"] == 4.0
    assert config["max_steps"] == config["save_steps"] == 250
    assert config["save_total_limit"] == 1
    pipeline = load(manifest)
    rendered = pipeline.select(models=[candidate]).plan().render()
    assert len(rendered.splitlines()) == 2
    assert f"train-{candidate}" in rendered
    assert "eval_wmdp_mmlu.sbatch" in rendered
    assert "--dependency=afterok:" in rendered
    assert f"{candidate}/checkpoint-250" in rendered

    candidates = [f"{name}--hpo--grid-002--{index:03d}" for index in range(1, 10)]
    search_configs = [
        yaml.safe_load((output / f"configs/{model}.yml").read_text())
        for model in candidates
    ]
    assert {
        (item["learning_rate"], item["custom_trainer_args"]["retain_coefficient"])
        for item in search_configs
    } == {(lr, retain) for lr in (1e-6, 5e-6, 1e-5) for retain in (0.1, 1, 5)}
    assert all(
        item["max_steps"] == item["save_steps"] == 250 and item["save_total_limit"] == 1
        for item in search_configs
    )
    rendered = pipeline.select(models=candidates).plan().render()
    assert len(rendered.splitlines()) == 18
    assert f"train-{candidate}" not in rendered
    assert rendered.count("eval_wmdp_mmlu.sbatch") == 9
    assert rendered.count("--dependency=afterok:") == 9


def test_sweep_name_template_drives_canonical_paths_and_wandb(tmp_path, monkeypatch):
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "named_sweep_test")
    monkeypatch.chdir(EXAMPLE)
    (tmp_path / "base.yml").write_text(
        yaml.safe_dump(
            {
                "learning_rate": 1e-5,
                "max_steps": 250,
                "save_steps": 50,
                "warmup_ratio": 0.1,
            }
        )
    )
    sweep = {
        "id": "named-experiment",
        "name_template": "model--lr{learning_rate:.0e}-ms{max_steps}",
        "model_root": "artifacts/tamia/models",
        "sweeps": [{"name": "lrs", "parameters": {"learning_rate": [1e-5, 5e-4]}}],
        "producer": {"script": "scripts/slurm/tamia/train.sbatch"},
        "evaluations": [],
    }
    source = tmp_path / "sweep.yml"
    source.write_text(yaml.safe_dump(sweep))
    manifest = hpo.generate(source, tmp_path / "generated")
    models = yaml.safe_load((manifest.parent / "models.yml").read_text())["models"]
    assert set(models) == {"model--lr1e-5-ms250", "model--lr5e-4-ms250"}
    for name, model in models.items():
        config = yaml.safe_load(Path(model["producer"]["args"][0]).read_text())
        assert config["wandb_name"] == name
        assert config["output_dir"] == f"artifacts/tamia/models/{name}"
        assert model["base_model_name_or_path"] == config["output_dir"]
        assert model["evaluation_root"] == "artifacts/tamia/evals"


@pytest.mark.parametrize(
    "template",
    [
        "{unknown}",
        "{learning_rate.real}",
        "{learning_rate!r}",
        "../bad",
        "{learning_rate:{max_steps}}",
    ],
)
def test_name_template_rejects_invalid_placeholders_and_slugs(template):
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "invalid_name_test")
    with pytest.raises(ValueError):
        hpo.candidate_name(template, {"learning_rate": 1e-5, "max_steps": 250})


def test_gip_tracking_defenses_preserve_selected_settings_and_evaluate(monkeypatch):
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    pipeline = load("configs/experiments/z7b-wmdp-bio-gd15pp-npo5pp-ghost-track.yml")
    stages = pipeline.plan().stages
    assert len(stages) == 4
    for original, cls in [
        ("z7b-wmdp-bio--15pp--unlearn-fft-gd", "GIPGradDiffTrainer"),
        ("z7b-wmdp-bio--5pp--unlearn-fft-npo", "GIPNPOTrainer"),
    ]:
        name = original + "--ghost-track"
        parent = EXAMPLE / "configs/stages-wmdp/0_unlearn"
        before = yaml.safe_load((parent / original / "selected.yml").read_text())
        after = yaml.safe_load((parent / name / "selected.yml").read_text())
        assert after["trainer_cls"] == "trainers.tr_gip." + cls
        assert after["output_dir"] == "artifacts/tamia/models/" + name
        assert after["wandb_name"] == name
        changed = {"trainer_cls", "output_dir", "wandb_name"}
        assert {k: v for k, v in before.items() if k not in changed} == {
            k: v for k, v in after.items() if k not in changed
        }
        training = next(s for s in stages if s.id == "train-" + name)
        evaluation = next(
            s
            for s in stages
            if s.id.startswith("eval-") and s.args[0] == after["output_dir"]
        )
        assert evaluation.depends_on == (training.id,)


def test_npo_sam_ghost_tracking_preserves_selected_defense(monkeypatch):
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    original = "z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2"
    name = original + "--ghost-track"
    parent = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    before = yaml.safe_load((parent / original / "selected.yml").read_text())
    after = yaml.safe_load((parent / name / "selected.yml").read_text())
    changed = {"trainer_cls", "output_dir", "wandb_name"}
    assert {k: v for k, v in before.items() if k not in changed} == {
        k: v for k, v in after.items() if k not in changed
    }
    assert after["trainer_cls"] == "trainers.tr_gip.GIPNPOSAMTrainer"
    training, evaluation = (
        load("configs/experiments/z7b-wmdp-bio-npo-sam5pp-ghost-track.yml")
        .plan()
        .stages
    )
    assert evaluation.depends_on == (training.id,)
    assert evaluation.args[0] == after["output_dir"]


@pytest.mark.parametrize("method,pp", [("gd", "15pp"), ("npo", "5pp")])
def test_squared_gip_sweep_preserves_base_defense_and_has_fp32_control(
    monkeypatch, tmp_path, method, pp
):
    from lilpipe import load

    monkeypatch.chdir(EXAMPLE)
    name = f"z7b-wmdp-bio--{pp}--unlearn-fft-{method}"
    parent = EXAMPLE / "configs/stages-wmdp/0_unlearn"
    before = yaml.safe_load(
        (parent / (name + "--ghost-track") / "selected.yml").read_text()
    )
    directory = parent / (name + "--gip-squared-fp32")
    base = yaml.safe_load((directory / "base.yml").read_text())
    changed = {
        "custom_trainer_args",
        "bf16",
        "fp16",
        "torch_dtype",
        "output_dir",
        "wandb_name",
    }
    assert {k: v for k, v in before.items() if k not in changed} == {
        k: v for k, v in base.items() if k not in changed
    }
    for key, value in before["custom_trainer_args"].items():
        assert base["custom_trainer_args"][key] == value
    assert base["bf16"] is False and base["fp16"] is False
    assert base["torch_dtype"] == "float32"
    assert base["custom_trainer_args"]["cosine_rho"] == 1e-4
    hpo = load_script(EXAMPLE / "scripts/hpo.py", "squared_gip_hpo")
    manifest = hpo.generate(directory / "sweep.yml", output_root=tmp_path)
    configs = [
        yaml.safe_load(p.read_text()) for p in (tmp_path / "configs").glob("*.yml")
    ]
    assert sorted(c["custom_trainer_args"]["cosine_coefficient"] for c in configs) == [
        0,
        0.1,
        1,
        10,
    ]
    assert all(c["custom_trainer_args"]["cosine_pairs"] == 2 for c in configs)
    stages = load(manifest).plan().stages
    assert len(stages) == 8
    assert all(len(s.depends_on) == 1 for s in stages if s.id.startswith("eval-"))
