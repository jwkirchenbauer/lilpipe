# Hyperparameter follow-up operations

This run contains four new paired defenses and attacks: R changes retain coefficient to 4, I changes inner learning rate to 1e-5, O changes outer learning rate to 1e-5, and RIO changes all three. B is the completed external local baseline. The exact scientific matrix is in `matrix.json`; canonical configs and source snapshots are already frozen. The numerical landscape module remains byte-identical to the collaborator analysis-v3 module (SHA256 `e8ab333af2f9106204ba0bb17a66d7a0adb21e80dd2970d16fb9aa8f1c953054`).

The parent owns submissions. This preparation task submitted no jobs. Original training and downstream dry-run files are retained as provenance. Scheduling revisions must use the parent's current job manifests; original afterok IDs can become stale after cancellations. The input manifest deliberately excludes job records, runtime statuses, and root metadata so scheduling revisions do not change scientific input identities.

## Frozen profiles

| Profile | Entry | Purpose |
|---|---|---|
| Training | `source/profiles/training-v1/run.sh` | Delegates to the canonical training worker; existing parent training submissions already use that canonical worker. |
| Evaluation | `source/profiles/evaluation-v1/run.sh` | Uses the new declared evaluation scope and the frozen trainer/plugin tree. |
| Native landscapes and profiles | `source/profiles/landscapes-v1/run.sh` | Uses the unchanged numerical module and the collaborator analysis-v3 trace wrapper. |
| Original static recipe | `source/profiles/static-v1/run.sh` | Uses exact original static get_ll/plotter bytes and separate numerical/rendering processes. |

Each operational entry checks its profile manifest and `operational-inputs.sha256`. Workers require a completed training status and a passed `training-validation.json` proof, including exact observed batch tensors/order against B. Scheduler afterok is useful ordering information but does not substitute for these checks. All allocations use the verival bank and `torchrun=false` through the launcher. The native landscape wrapper suppresses periodic asynchronous stack walks while retaining fatal-fault reporting; this does not establish the cause of any earlier crash.

`preparation-validation.json` records config comparisons, immutable prepared-cache hardlink identities, probes, source hashes, and byte equality of the active trainer, plugin, sampler, and data helper against B. The recorder is byte-identical to the completed compute study. The optional sampler branch remains disabled by default. Original BF16/AdamW training behavior, including inner weight decay 0.01, is unchanged. Saved training tensors can have mixed dtypes; the existing landscape convention casts evaluated model parameters to BF16.

## Reviewable downstream plans

`operational-dryrun-plan.json` contains ten jobs, exact launcher dry-run arguments, generated launch directories, logs, prerequisites, and the launcher-emitted submission arguments. `static-dryrun-plan-v2.json` contains the four static measurements and their final common-scale report, with replacement defense dependencies from `defense-jobs-v2.json`. The original `static-dryrun-plan.json` and its launch directories remain as superseded provenance. These plans were generated with `--dryrun`; their logs explicitly state that no jobs were launched. The parent subsequently submitted and revised operational dependencies; use the parent's current scheduling manifests for those jobs. Preserve the old launch scripts rather than overwriting them.

The planned sequence is:

1. Complete all four defenses and attacks, with strict post-training batch-order validation.
2. Run only the four attack benchmark workers. Step 0 measures each paired defense endpoint. WMDP uses the existing standard attack milestones; MMLU uses 0/128/512/1000. Defense checkpoints receive native CE and landscape measurements, without repeated full benchmark groups.
3. Run the two native landscape groups. Measure the shared profile reference R first, then I/O and RIO. This avoids accepting another case before common profile source/input identities are established.
4. Run final checkpoint and consolidated dynamic reports only after relevant numerical data, benchmark results, profile caches, and stage statuses are complete. The parent owns the separate consolidated dynamic comparison profile.
5. Run each original static measurement after its defense is validated; attacks are not required for this stage. Render the final static comparison only after all four static statuses are complete.

The evaluation scope is declared in root `study.json` as `evaluation_cases`. `evaluation-scope-amendment.json` records the metadata-only amendment; canonical training configs and frozen sources were not changed. The new evaluator is isolated in evaluation-v1 rather than patched into the already frozen canonical source.

## Original static series

The static worker takes `case RUN DEFENSE_CASE`; the report takes `report RUN`. The four measurement outputs are `static/cases/hparam-{r,i,o,rio}-defense`. They use radii 1000/500/200/125/100/10/1/0.1, a 9x9 grid per radius (648 points), seed 42, fixed10 documents, truncation 512, batch size 2, original BF16 random normalization and sequential BF16 perturbation arithmetic, and default attention. The worker validates that the default dataset selection matches pinned fixed10 texts and that collation/tokenizer inputs equal the original baseline. It binds grids to source, settings, checkpoint file identities, data, tokenizer-derived batch hashes, and versions. A stale or unbound cache is rejected. No extra trajectory checkpoint evaluations are performed.

This static convention is separate from the newer saved FP32 orthonormal dynamic bases and eager attention. The final `static/comparison/report.md` compares B/R/I/O/RIO plus the uploaded HF checkpoint. Every radius has common absolute CE color/z limits and camera across all six checkpoints, and a separate set of center-relative contours with common symmetric color limits. Radii have separate scales. The uploaded checkpoint is a multifactor context comparison, including different sampling/batching; it is not another controlled hparam cell. Final reporting rejects pending, incomplete, or incompatible inputs.

## Verification records

- `logs/evaluation-scope-tests.log`: Black and diff checks; 19 evaluation, 13 hparam preparation/order-guard, and 10 compute tests passed.
- `logs/operational-freeze.log`: scientific source/config/cache validation and immutable profile creation.
- `logs/operational-dryruns.log`: ten dry runs only; individual launcher logs are named `logs/dryrun-JOB.log`.
- `logs/static-profile-tests.log`: four synthetic CPU fixtures for grid completeness/centers, stale protocols, and shared limits; operational manifest and Bash checks.
- `logs/static-profile-dryruns.log`: five static dry runs only and the tiny diagnostic wrapper integration check. No models were loaded for these tests.
- `logs/static-profile-dryruns-v2.log`: scheduling-only replacement of five unsubmitted static dry runs; scientific sources and settings are unchanged.

Preparation and tests used interactive Bash with absolute-prefix `mm_activate`. Training uses `/usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4`; evaluation/analysis uses `/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2`. On this system the latter resolves to `/usr/WS2/kirchenb/lilpipe-tar-eval-rocm72-v2`; both names identify the same existing prefix.
