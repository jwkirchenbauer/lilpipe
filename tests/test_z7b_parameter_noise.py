"""Authors' NPO noise experiment evaluates original and perturbed checkpoints."""

from pathlib import Path

import pytest

from lilpipe import load

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/tamper-resistance"


@pytest.mark.parametrize(
    "percentage,fraction,with_baseline", [(5, "0.05", True), (1, "0.01", False)]
)
def test_npo_noise_evaluates_both_tasks_and_preserves_the_author_model(
    monkeypatch, percentage, fraction, with_baseline
):
    monkeypatch.chdir(EXAMPLE)
    plan = load(f"configs/experiments/z7b-npo-author-noise{percentage}.yml").plan()
    assert len(plan.stages) == (5 if with_baseline else 3)
    generation = next(s for s in plan.stages if s.id.startswith("generate-"))
    assert generation.args[generation.args.index("--percentages") + 1] == str(
        percentage
    )
    assert generation.args[-2:] == ("--seed", "42")
    source = generation.args[generation.args.index("--model") + 1]
    assert (
        "models--OPTML-Group--NPO-WMDP/snapshots/fec7960fb3925037a6befc5819036cbc55664b18"
        in source
    )
    evaluations = [s for s in plan.stages if s.id.startswith("eval-")]
    expected = 2 if with_baseline else 1
    assert len(evaluations) == 2 * expected
    for stage in evaluations:
        assert stage.script.endswith("eval_capabilities.sbatch")
        assert stage.args[-2:] == ("--include_path", "lm_eval_tasks")
        assert "NUM_FEWSHOT=0" in stage.sbatch_args[-1]
        assert "--limit" not in stage.args
        if "--perturb-" in stage.args[0]:
            assert stage.depends_on == (generation.id,)
            assert f"l2r{fraction}-seed42" in stage.args[0]
        else:
            assert stage.args[0] == source
            assert not stage.depends_on
    assert (
        sum(
            "CAPABILITY_TASKS=wmdp_bio_robust" in s.sbatch_args[-1] for s in evaluations
        )
        == expected
    )
    assert (
        sum("CAPABILITY_TASKS=mmlu_no_bio" in s.sbatch_args[-1] for s in evaluations)
        == expected
    )
