"""Noise baselines and Axolotl sweeps stay independent of author training scripts."""

from pathlib import Path
import yaml
from lilpipe import load

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"
START = "llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config"


def test_static_registry_contains_sources_not_sweep_candidates():
    models = yaml.safe_load((EXAMPLE / "configs/registries/models.yml").read_text())[
        "models"
    ]
    assert not any(
        "--data-overlapping--relearn-fft-author-codebase-lr" in name for name in models
    )
    assert not any(
        name.startswith(START + "--data-overlapping--relearn-fft-lr") for name in models
    )
    assert all(
        "relearning_sweep.sbatch" not in value.get("producer", {}).get("script", "")
        for value in models.values()
    )


def test_base_noise_evaluation_is_independent_of_training(monkeypatch):
    monkeypatch.chdir(EXAMPLE)
    plan = (
        load("configs/experiments/llama2-7b-base-noise.yml")
        .select(models=["llama2-7b-beavertail--perturb-gaussian-l2r0.05-seed42"])
        .plan()
    )
    generation, evaluation = plan.stages
    assert evaluation.depends_on == (generation.id,)
    assert "INCLUDE_HARMFULNESS=0" in evaluation.sbatch_args[0]


def test_author_training_has_no_axolotl_lineage_configs():
    assert not list(
        EXAMPLE.glob(
            "configs/stages-beavertails/**/*relearn-fft-author-codebase/base.yml"
        )
    )
    assert not list(
        EXAMPLE.glob(
            "configs/stages-beavertails/**/*relearn-fft-author-codebase/sweep.yml"
        )
    )
