# Tamia workflow consolidation

Run from `examples/tamper-resistance`. Permanent launchers are:

```text
train.sbatch CONFIG
noise.sbatch --model MODEL --output-prefix PREFIX --output REPORT [NOISE_ARGS...]
eval_harmfulness_and_capabilities.sbatch [--steps STEPS] [--baseline MODEL]
    [--max-step N] [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]
eval_wmdp_mmlu.sbatch [--checkpoints] MODEL OUTPUT
    [REMOVE_CHECKPOINTS=false] [REMOVE_FINAL=false] [BASELINE]
seam_codebase_train_seam_defence.sbatch CONFIG MODEL RUN [STEPS]
seam_codebase_train_seam_relearning.sbatch CONFIG MODEL OUTPUT LR [AUTHOR_ARGS...]
seam_codebase_eval_seam_defence.sbatch MODEL CONFIG OUTPUT RUN
```

The author-codebase launchers retain their training and defense implementation.
The two domain evaluation launchers are standalone; no shared evaluation helper
is introduced. Existing Python evaluators remain available for older research
and Mila workflows.

BeaverTails uses four-GPU LM Eval and distributed DeBERTa harmfulness. Set
`INCLUDE_GSM8K=1` for five-shot GSM8K, `INCLUDE_HARMFULNESS=0` for capabilities
only, or `HARMFULNESS_ONLY=1` for harmfulness only. `CAPABILITY_TASKS`,
`NUM_FEWSHOT`, and `MAX_BATCH_SIZE` select capability settings.
`CAPABILITIES_SUBDIR=capabilities-4gpu` preserves the baseline result layout.
Model/output pairs permit evaluating several models in one allocation.

`RESUME_EVALUATION=1` reuses valid complete stages; `EVAL_LOCAL_LOG=1` archives
node-local logs on exit. CUDA caches always use node-local storage. Checkpoint
ranges default to every 50 steps through `CHECKPOINT_MAX_STEP` (250). Set
`CHECKPOINT_START_STEP` / `CHECKPOINT_END_STEP` to split a trajectory across jobs,
or pass explicit increasing `--steps 0,100,200,250` and `--baseline MODEL`.
The final alias is written only after evaluating the terminal checkpoint and
verifying identical final weights, tokenizer, and generation configuration.
Evaluation does not delete BeaverTails models.

WMDP defaults to `mmlu_no_bio,wmdp_bio_robust`, zero-shot, BF16, batch size 32,
and four GPU replicas. `CAPABILITY_TASKS` selects an individual task;
`BATCH_SIZE=auto:4` / `MAX_BATCH_SIZE` select automatic batching. Trajectory
checkpoints are visited numerically. Optional removal happens only after every
requested evaluation succeeds. No deletion flags are set by the registry.

Axolotl sweeps generate independent single-run producers from a shared YAML plus CLI
overrides. The canonical registry contains sources and fixed models, not repeated
LR candidates. Source dependencies can be assigned per sweep group. Author training uses standalone
script wrappers and native configs under `../seam/config/`, outside the Axolotl
sweep/selected-config structure.
The old packed author sweep remains only under `scripts/misc/` for historical use.

Temporary experiment orchestration and the six experiment-specific TAR loss-landscape/Hessian launchers live in `scripts/misc/`. Generic analysis launchers remain under `scripts/slurm/tamia/`. Run metadata and
submitted source snapshots remain under `artifacts/tamia/runs/`; historical
filenames are immutable, even when current launchers are renamed.

Generate our SEAM attacks from the shared lineage `base.yml` and `sweep.yml`.
The sweep defines exactly seven combinations, with descriptive model IDs and
matching W&B/model/eval names. `config_mode: overrides` writes the shared base, models/experiment definitions, and
per-run effective settings under `artifacts/generated/SWEEP_ID/`. Existing resolved
sweeps retain their defaults. Generation does not submit jobs.

The inventory below describes an earlier retirement and is historical.

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
