# Mila launch script cleanup

Four supported entry points remain: `train.sbatch`, `eval_beavertails_capabilities.sbatch`, `eval_beavertails_capabilities_trajectory.sbatch`, and `analysis_get_param_norm.sbatch`. The other 16 workflows have no live configuration callers and their wrappers are retired. Python research implementations and historical artifacts remain intact.

## Interfaces

```text
train.sbatch CONFIG
analysis_get_param_norm.sbatch [--model MODEL] [--output OUTPUT]
eval_beavertails_capabilities.sbatch ID MODEL METHOD RESULT_ROOT
eval_beavertails_capabilities_trajectory.sbatch ID BASELINE MODEL_DIR LR DATA METHOD RESULT_ROOT
    [RM_INTERMEDIATE=false] [RM_FINAL=false]
```

Training directly invokes Axolotl with `--launcher python`. Evaluations directly invoke the existing BeaverTails evaluator, preserving harmfulness, MMLU, ARC Easy, metadata, and summary JSON. Single-model output is `RESULT_ROOT/ID/summary.json`; trajectory output is `RESULT_ROOT/ID/checkpoint-<step>/summary.json`.

Trajectories discover only `checkpoint-*` directories. They do not synthesize baseline or output-root final evaluation points. A saved checkpoint zero is evaluated when present. The Python evaluator still records its cached baseline metrics in `pre` and checkpoint metrics in `post`.

Both removal arguments default to `false`. When enabled, intermediate removal runs only after the corresponding evaluation succeeds. Final removal runs after the loop and removes the entire model directory, including any remaining checkpoints. An evaluation failure stops the job before subsequent removal.

## Cluster setup

All four wrappers have separate cluster and application sections. Training and evaluation default to one node, one task, one L40S GPU, eight CPUs, 32 GB, and three hours. The moved parameter-norm analysis script preserves its existing long partition, one L40S GPU, four CPUs, 64 GB, and 30-minute limit; its Slurm logs now live under `artifacts/mila/logs/`. Existing submission overrides remain effective. The uv cache and environment are both job-local under `$SLURM_TMPDIR`, with the job ID in their paths. Hugging Face model caching remains on scratch. Mila installs can access the network; no offline mode is imposed. Copy link mode is retained.

The shared-cache lock and install-concurrency override are removed because each job now has its own uv cache. The job-ID suffix also isolates jobs when Mila sets `$SLURM_TMPDIR` to `/tmp`. No HTTP proxy module or additional W&B cache, staging, or timeout overrides are added.

## Wrapper dependency inventory

This inventory records references before cleanup, including historical documentation and obsolete test fixtures. Paths outside the example are repository-relative. Untracked generated manifests and shell history are not covered. Regenerate manifests that reference old wrapper names.

### `eval_alignment.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/evaluate_alignment.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_alignment_trajectory.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/evaluate_alignment_trajectory.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_beavertail_and_capabilities_single.sbatch`

- Disposition: renamed to `eval_beavertails_capabilities.sbatch`; evaluator dependencies remain shared.
- Invoked Python / CLI: `scripts/evaluate_beavertail_and_capabilities.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_beavertail_and_capabilities_trajectory.sbatch`

- Disposition: renamed to `eval_beavertails_capabilities_trajectory.sbatch`; evaluator dependencies remain shared.
- Invoked Python / CLI: `scripts/tamia_evaluation.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: `configs/registries/evals.yml`
- Script callers: None found in tracked files.
- Documentation references: `docs/tamia-refactor.md`
- Affected tests: None found in tracked files.

### `eval_local_trajectory.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `lm_eval (Accelerate module invocation)`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: `docs/tamia-refactor.md`
- Affected tests: None found in tracked files.

### `eval_mmlu_no_bio_ckpts.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `lm_eval (Accelerate module invocation)`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_mmlu_no_bio_single.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `lm_eval (Accelerate module invocation)`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_tofu.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/evaluate_tofu.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_wmdp_bio_mcqa_ckpts.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `lm_eval (Accelerate module invocation)`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `eval_wmdp_bio_mcqa_single.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `lm_eval (Accelerate module invocation)`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `merge_lora.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `axolotl merge-lora CLI`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `optuna_worker.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/optuna_hpo.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: `tests/test_tamper_resistance.py`

### `relearn.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `axolotl train CLI`, `axolotl merge-lora CLI`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `relearn_alignment.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `axolotl train CLI`, `axolotl merge-lora CLI`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `relearn_tofu.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/evaluate_tofu_attack.py`, `scripts/run_tofu_relearn.py`
- Configuration references: `configs/stages-tofu/2_relearn/${method}--relearn-fft/${attack}.yml`
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `train_select_tofu.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/select_tofu_checkpoint.py`, `axolotl train CLI`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `train_tofu_base.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/evaluate_tofu.py`, `axolotl train CLI`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

### `weight_steering.sbatch`

- Disposition: retired; workflow-only dependencies are historical-only / candidates for later removal.
- Invoked Python / CLI: `scripts/weight_steering.py`
- Configuration references: None found in tracked files.
- Config/manifest callers: None found in tracked files.
- Script callers: None found in tracked files.
- Documentation references: None found in tracked files.
- Affected tests: None found in tracked files.

## Supporting files

- **Still shared:** the two evaluators are now `scripts/evals/evaluate_beavertail_and_capabilities.py` and `scripts/evals/tamia_evaluation.py`. The Qwen stage configs, `scripts/hpo.py`, and shared custom trainers remain active. Mila calls the BeaverTails evaluator directly; the Tamia helper keeps its existing behavior.
- **Deleted implementations:** the five TOFU scripts and `src/data/tofu.py`, both old alignment evaluators and their comparison script, `scripts/weight_steering.py`, and `scripts/optuna_hpo.py` have been removed along with their dedicated tests. TOFU in other examples is unchanged.
- **Retained manual analysis:** `scripts/analysis/`, `scripts/analysis.ipynb`, and Slurm templates remain unchanged. Absence of a launch wrapper alone does not establish that a research analysis is obsolete.
- The paths listed in the wrapper inventory above describe historical callers before retirement; they are not supported execution instructions. Missing historical configuration templates are not recreated.

### Deleted Python implementations

| Retired workflow | Removed files in this example |
| --- | --- |
| TOFU | `scripts/evaluate_tofu.py`, `scripts/evaluate_tofu_attack.py`, `scripts/select_tofu_checkpoint.py`, `scripts/run_tofu_relearn.py`, `scripts/summarize_tofu_table2.py`, `src/data/tofu.py` |
| Old alignment evaluation | `scripts/evaluate_alignment.py`, `scripts/evaluate_alignment_trajectory.py`, `scripts/compare_alignment_trajectories.py` |
| Weight steering | `scripts/weight_steering.py` |
| Optuna | `scripts/optuna_hpo.py` |

Dedicated tests for these removed workflows were removed as well. The custom
trainers, shared data adapters, active grid-HPO tests, analysis tools, and TOFU
implementations in other examples remain available. The direct Optuna training
dependency and its now-unreachable lock entries were pruned without upgrading
the remaining dependencies.

Analysis Python entry points have since been renamed; see
[the analysis rename mapping](analysis-scripts.md). Former names below are historical.
