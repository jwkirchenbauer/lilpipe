# Tamia workflow consolidation

Exactly six Slurm entry points remain. WMDP-Bio robust and MMLU-no-bio run as a pair; BeaverTails retains harmfulness, MMLU, and ARC Easy. Each trajectory uses one allocation. Historical artifacts are unchanged.

## Supported interfaces

Evaluation IDs follow `<cluster>-<gpu>-<model-family>--<tasks>-<mode>` throughout
the registry. For example,
`mila-l40s-q06b--beavertails-and-capabilities-trajectory` targets Qwen3-0.6B on
Mila L40S GPUs. Each registry entry documents its complete output path in an
`Output` comment; the result root is still an explicit script argument. The
Zephyr abbreviation is `z7b`, and the other families are `llama2-7b` and
`q06b`. Renaming an evaluation changes its stage ID but preserves its
artifact paths. Regenerate generated manifests that contain previous IDs.

Run from the example root. All six wrappers declare one node, one task, four
H100s, 48 CPUs, 480 GB, and three hours; explicit registry/CLI resource arguments
override these defaults. Each wrapper creates an offline, job-local uv environment.

```text
train.sbatch CONFIG [--single-gpu] [--merge] [AXOLOTL_ARGS...]
analysis_gen_ll.sbatch [GEN_LL_ARGS...]
eval_wmdp_mmlu.sbatch ID MODEL ADAPTER RESULT_ROOT
    [--checkpoint-step N] [--final-model PATH]
eval_wmdp_mmlu_trajectory.sbatch ID BASELINE TRAJECTORY BASE_MODEL ADAPTER RESULT_ROOT STEPS
    --final-step N [--final-model PATH] [--delete-trajectory]
eval_beavertails_capabilities.sbatch ID MODEL METHOD RESULT_ROOT
eval_beavertails_capabilities_trajectory.sbatch ID BASELINE TRAJECTORY LR DATA METHOD RESULT_ROOT STEPS
    --final-step N [--final-model PATH] [--gaussian-sigma S --gaussian-seed N]
    [--delete-trajectory]
```

Use `-` for ADAPTER when evaluating full models. For adapter trajectories,
TRAJECTORY is the adapter output directory and BASE_MODEL is its pretrained base.
STEPS is an increasing comma-separated list, for example `0,100,200,250`.
`--final-model` explicitly identifies a terminal model saved at an output root;
it is eligible only at the declared final step, after checking for that checkpoint.
Missing intermediate checkpoints are errors. A saved `checkpoint-0` takes
precedence over the baseline, including a saved perturbed initial model.

The shared Python runner `scripts/evals/tamia_evaluation.py` handles checkpoint
resolution and BeaverTails evaluation. Both WMDP wrappers keep their Accelerate
command visible in Bash: one launch per model/checkpoint with
`--tasks mmlu_no_bio,wmdp_bio_robust` and the allocated GPU count. The combined
result is written to a fresh job-local directory, validated for both tasks, and
copied into the existing `wmdp-bio-robust` and `mmlu-no-bio` result directories.
Each trajectory wrapper invokes its guarded cleanup
mode **after** successful evaluation. Cleanup is a no-op without the deletion flag.
It requires checkpoint zero and the terminal step, valid results for every
requested point, matching model identity, and an allowed trajectory path. The
existing canonical allowlists and BeaverTails generated-candidate restrictions
remain; arbitrary paths are rejected. WMDP result verification checks both task
scores and their recorded pretrained/adapter arguments. BeaverTails verification
checks model/step identity and harmfulness, MMLU, and ARC Easy in both phases.

The registry has five Tamia entries (`tamia-h100-z7b--wmdp-and-mmlu-single`, `tamia-h100-z7b--wmdp-and-mmlu-final`,
`tamia-h100-z7b--wmdp-and-mmlu-trajectory`, and the two Tamia BeaverTails IDs) plus two Mila
entries. Single results retain their direct model-directory layout; final and
trajectory results use `checkpoint-N`, with separate WMDP and MMLU directories.
Retries rerun both tasks. Existing BeaverTails registry workflows explicitly
request deletion; WMDP workflows retain checkpoints by default.

Model metadata supplies `evaluation_root`, `evaluation_trajectory`,
`evaluation_checkpoint_steps`, `evaluation_final_step`, and
`evaluation_final_model`. HPO derives the final step and checkpoint list from each
concrete candidate's `max_steps` and `save_steps`, including a non-divisible final
step. Existing explicit step selections are retained. Launcher/GPU choices are
producer arguments rather than separate script names. Model budgets no longer
select distinct evaluation IDs.

The two remaining Mila registry entries keep their original positional interfaces, with
the new descriptive evaluation IDs. Their BeaverTails
trajectory wrapper now translates interval/maximum arguments to explicit steps
and uses the shared Python runner, removing its dependency on the deleted Tamia
cleanup script and including the terminal model.

## Retired wrapper dependency inventory

This inventory records references **before migration**, including obsolete tests. Paths outside this example are repository-relative. A missing caller means no tracked caller was found; shell history and untracked generated manifests are not covered.

Shared Python entry points remain supported. Implementations used only by retired workflows are retained as candidates for later cleanup; no Python research code or historical results are deleted here.

### `authors_relearn_preserve.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `src/exec/relearn_model.py`
- Configuration literals: `configs/unlearn/wmdp/Relearn+Forget.json`
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `cleanup_beavertail_trajectory.sbatch`

- Disposition: replaced by `opt-in deletion at the bottom of the BeaverTails trajectory job`; dependencies remain shared.
- Invoked Python: None found in tracked files.
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: `scripts/slurm/mila/eval_beavertail_and_capabilities_trajectory.sbatch`, `scripts/slurm/tamia/eval_beavertail_and_capabilities_trajectory.sbatch`
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_refactor.py`

### `cleanup_wmdp_trajectory.sbatch`

- Disposition: replaced by `opt-in deletion at the bottom of the WMDP trajectory job`; dependencies remain shared.
- Invoked Python: None found in tracked files.
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_resistance.py`

### `eval_alignment.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `scripts/evaluate_alignment.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_authors_table3.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_beavertail_and_capabilities_single.sbatch`

- Disposition: replaced by `eval_beavertails_capabilities.sbatch`; dependencies remain shared.
- Invoked Python: `scripts/evaluate_beavertail_and_capabilities.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_beavertail_and_capabilities_trajectory.sbatch`

- Disposition: replaced by `eval_beavertails_capabilities_trajectory.sbatch`; dependencies remain shared.
- Invoked Python: `scripts/evaluate_beavertail_and_capabilities.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_resistance.py`

### `eval_local_trajectory.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu_trajectory.sbatch`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_local_trajectory_series.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu_trajectory.sbatch`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_local_trajectory_steps.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu_trajectory.sbatch`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_resistance.py`

### `eval_mmlu_no_bio_ckpts.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu.sbatch --checkpoint-step`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_mmlu_no_bio_single.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu.sbatch`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_trajectory.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu_trajectory.sbatch (explicit cached baseline)`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_wmdp_bio_mcqa_ckpts.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu.sbatch --checkpoint-step`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_wmdp_bio_mcqa_single.sbatch`

- Disposition: replaced by `eval_wmdp_mmlu.sbatch`; dependencies remain shared.
- Invoked Python: `lm_eval (Accelerate module invocation)`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `gen_ll.sbatch`

- Disposition: replaced by `analysis_gen_ll.sbatch`; dependencies remain shared.
- Invoked Python: `scripts/analysis/gen_ll.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `model_dist.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `scripts/analysis/get_model_dists.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `optuna_worker_model_parallel.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `scripts/optuna_hpo.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `perturb_model.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `scripts/analysis/gen_perturbed_model.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-perturb/sweep.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_refactor.py`

### `plot_smoothness.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `scripts/analysis/gen_smoothness.py`, `scripts/analysis/plot_smoothness.py`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `train_model_parallel.sbatch`

- Disposition: replaced by `train.sbatch --launcher python`; dependencies remain shared.
- Invoked Python: `axolotl CLI`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/models.yml`, `configs/stages-beavertails/0_alignment/llama2-7b-beavertail--align-fft-seam-legacy/sweep.yml`, `configs/stages-beavertails/0_alignment/llama2-7b-beavertail--align-fft-seam/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--15pp--unlearn-fft-gd-sam-rho1e-2/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--15pp--unlearn-fft-gd/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--15pp--unlearn-fft-npo-sam-rho1e-2/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--15pp--unlearn-fft-npo/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-gd-seam/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-wd/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-npo-wd/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu-sam-rho1e-2/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu-wd/sweep.yml`, `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu/sweep.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_refactor.py`

### `train_seam_segmented.sbatch`

- Disposition: retired; workflow-specific dependencies are historical-only / candidates for later removal.
- Invoked Python: `axolotl CLI`
- Configuration literals: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_resistance.py`

### `train_single_gpu.sbatch`

- Disposition: replaced by `train.sbatch --single-gpu --launcher python`; dependencies remain shared.
- Invoked Python: `axolotl CLI`
- Configuration literals: None found in tracked files.
- Config/manifest callers: `configs/registries/models.yml`, `configs/stages-beavertails/1_relearn/llama2-7b-beavertail--align-fft-seam--data-disjoint--relearn-fft-ms1000/sweep.yml`, `configs/stages-beavertails/1_relearn/llama2-7b-beavertail--align-fft-seam--data-disjoint--relearn-fft-ms250/sweep.yml`, `configs/stages-beavertails/1_relearn/llama2-7b-beavertail--align-fft-seam-legacy--data-disjoint--relearn-fft-ms250/sweep.yml`, `configs/stages-beavertails/1_relearn/llama2-7b-beavertail--data-disjoint--relearn-fft-ms250/sweep.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_refactor.py`

## Supporting files that become stale

- **Deleted perturbation analysis:** the perturbation generator and plotter are removed. WMDP perturbation `base.yml` / `selected.yml` remain historical research inputs; the standalone sweep is recorded in `docs/retired-tamia/perturb-sweep.yml`.
- **Deleted Python workflows:** `scripts/optuna_hpo.py`, the old alignment evaluators/comparison, `scripts/weight_steering.py`, the five TOFU scripts, and `src/data/tofu.py` have now been removed with their dedicated tests. MiniCPM configs and the associated workers were already retired.
- `scripts/slurm/mila/eval_local_trajectory.sbatch`: its MiniCPM registry callers were removed; it is now a candidate for later removal.
- **Deleted model-distance and smoothness analysis:** the model-distance generator, smoothness generator, and both smoothness plotters are removed, together with `tests/test_model_dist.py`, `tests/test_smoothness_plot.py`, and `tests/test_smoothness_2d_plot.py`. Their retired wrappers above list historical invocation paths.
- **Active evaluators:** `scripts/evals/evaluate_beavertail_and_capabilities.py` and `scripts/evals/tamia_evaluation.py` remain supported. The Tamia helper was relocated without changing its behavior.
- External authors-codebase relearning/table-3 commands and their experiment-specific paths: historical-only; no replacement Slurm launcher.
- Segmented/requeue Slurm behavior is retired. Retain trainer/config code; ordinary training has no implicit resume or requeue mode.
- Untracked generated HPO manifests referencing old scripts must be regenerated from migrated sweeps before use. Existing artifacts and result directories remain intact.

Analysis Python entry points have since been renamed; see
[the analysis rename mapping](analysis-scripts.md). Former names below are historical.
