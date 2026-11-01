# Tamper Resistance: Agent Guidelines

Run every command from `examples/tamper-resistance`; all paths below are
relative to it. Training uses `.venv-train`, evaluation uses `.venv-eval`
(see `README.md`).

## Utility Budgets

Lineage names carry a utility budget tag, e.g. `z7b-wmdp-bio--5pp--unlearn-fft-rmu`.
`<N>pp` is the maximum allowed MMLU-no-bio drop, in percentage points, from the
base model. For Zephyr-7B the base MMLU-no-bio score is `58.59`:

| Tag | Max MMLU drop | Min MMLU-no-bio |
| --- | ---: | ---: |
| `--5pp--` | 5 | 53.59 |
| `--15pp--` | 15 | 43.59 |

Evaluations are written to `artifacts/tamia/evals/<model-id>/`, mirroring
`artifacts/tamia/models/`; the budget is part of each model id. Retired runs live in
`artifacts/tamia/evals/archive/`.

## Hyperparameter Tuning Procedure

1. **Define the sweep.** Each lineage lives at
   `configs/stages-wmdp/0_unlearn/<lineage>/` with a runnable `base.yml` and a
   `sweep.yml`. Each named stage in `sweep.yml` is an independent grid; only
   parameters inside one stage form a Cartesian product.
2. **Generate and submit.**

   ```bash
   python scripts/hpo.py generate configs/stages-wmdp/0_unlearn/<lineage>/sweep.yml
   lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml --dry-run
   lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml
   ```

   Generated configs, the model registry, and the manifest go to
   `artifacts/generated/hpo/<sweep-id>/`. Candidate `<sweep-id>--<stage>--NNN`
   maps to values through `artifacts/generated/hpo/<sweep-id>/configs/`. HPO runs keep
   only the final checkpoint.
3. **Monitor.** Jobs run on Tamia (`ssh tamia`; the checkout is
   `~/projects/lilpipe`). Check them with `squeue -u $USER` and `sacct -u $USER -X`.
   Each WMDP candidate gets one training job and one paired `tamia-h100-z7b--wmdp-and-mmlu-final`
   evaluation job. Budgets belong to model identities, not evaluation IDs.
4. **Collect.** For every candidate, read `acc,none` from the lm-eval
   `results_*.json` under
   `artifacts/tamia/evals/<sweep-id>--<stage>--NNN/checkpoint-*/{wmdp-bio-robust,mmlu-no-bio}/`.
5. **Select.** A candidate is *eligible* if its MMLU-no-bio drop stays within
   the lineage's budget (e.g. under 5 points for `--5pp--`). Among eligible
   candidates, select the one with the **lowest WMDP-Bio robust accuracy**,
   i.e. the strongest unlearning. Candidates near chance on WMDP (~25%) usually
   fail the budget because MMLU collapses too; never select them.
   **Tie-break:** when eligible candidates' WMDP scores are within noise of each
   other (about one standard error, ~1.6 points), prefer the one with the
   smaller MMLU drop. This matters most when several candidates reach chance.
   A candidate that satisfies several budgets is promoted only to the tightest
   one (e.g. `--5pp--`); do not keep the same model under two budget names.
6. **Refine if needed.** If the best eligible point sits on the edge of the grid,
   or the jump between neighbouring points is large, e.g. from eligible to
   collapsed, add a finer sweep stage around it and repeat from step 2.
7. **Record.** Add the sweep (grid, run count, promoted setting) to the
   *Hyperparameter sweeps* table in `docs/experiments.md`. When the choice is
   close, also add an eligibility table like the RMU + AO one.

## Running Generated Sweeps

Generate a sweep once, then use the generated manifest for both the dry run and
submission:

```bash
python scripts/hpo.py generate configs/stages-*/<stage>/<lineage>/sweep.yml
lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml --dry-run
lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml
```

For checkpoint trajectories, run at most two candidates at a time. Select each
candidate by generated model ID and reuse every upstream model that already
exists:

```bash
lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml \
  --models <candidate-id> \
  --existing-models <upstream-model-id> \
  --dry-run
lilpipe artifacts/generated/hpo/<sweep-id>/experiment.yml \
  --models <candidate-id> \
  --existing-models <upstream-model-id>
```

The dry run must contain only the selected attack and its dependent trajectory
evaluation. Do not submit a third candidate until one active candidate has
completed both jobs, written every expected checkpoint summary, and passed the
guarded cleanup (when `--delete-trajectory` is supplied) that removes its evaluated
trajectory. Each trajectory runs in one allocation. For a 1,000-step
trajectory saved every 50 steps, verify all 21 summaries from `checkpoint-0`
through `checkpoint-1000`. If training fails, fix or remove its partial output
before retrying. If only evaluation fails, keep the trajectory and resubmit just
the evaluation by marking both the upstream model and candidate as existing;
cleanup will run only after all expected results are present and valid. Include
the terminal step even when it is not divisible by the checkpoint interval.
Generated models expose `evaluation_checkpoint_steps` and `evaluation_final_step`;
an output-root terminal model requires an explicit `evaluation_final_model`.

Tamia has six supported Slurm entry points; see `docs/tamia-refactor.md` for
interfaces and the dependency inventory of retired wrappers. Regenerate old HPO
manifests before submission. Specialized Tamia perturbation, Optuna, authors'
reproduction, segmented training, and extra analysis wrappers are retired; their
research code and settings remain available.

Mila has four supported scripts: training, BeaverTails single-model/trajectory
evaluation, and parameter-norm analysis. Its trajectory script discovers `checkpoint-*` directories only;
baseline and final-root evaluation points are not synthesized. Both positional
removal flags default to `false`. See `docs/mila-refactor.md` for interfaces and
the dependency inventory of retired Mila workflows.

Active evaluator code lives under `scripts/evals/`. The Tamia trajectory helper
was relocated without a behavior change. TOFU, the old alignment evaluators,
weight-steering, and Optuna entry points are retired in this example; analysis
scripts and the notebook remain available.

### Promoting a BeaverTails SEAM defense

Promote a completed defense candidate without retraining it:

1. Copy its concrete hyperparameters into the lineage's `selected.yml`; keep
   `base.yml` as the reusable sweep template.
2. On Tamia, rename any current canonical model into
   `artifacts/tamia/models/archive/` so the change is reversible.
3. Move the selected candidate from
   `artifacts/generated/hpo/<sweep-id>/models/<candidate-id>` to its canonical
   `artifacts/tamia/models/<lineage>` path.
4. Verify that the promoted config and model agree before passing the canonical
   model as `--existing-models` to downstream relearning sweeps.

Do not copy a full model merely to promote it: these artifacts are large, and a
same-filesystem rename preserves the selected artifact without temporarily
doubling scratch usage.

## Promoting a WMDP Candidate

A candidate is promoted into a budget `<N>pp` that it satisfies. That budget
may differ from the one the sweep ran under: a `--15pp--` sweep candidate within
5 points can be promoted to `--5pp--`. Promotion is a config copy, an eval
copy, and a model move; nothing is retrained.

1. **Copy the config.** Write
   `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--<N>pp--<method>/selected.yml`
   with the candidate's concrete values. It is the lineage's `base.yml` with the
   chosen parameters; only `output_dir`, `dataset_prepared_path` and
   `wandb_name` are renamed to the `--<N>pp--` lineage. Do not add a registry or
   manifest entry.
2. **Copy the evals on Tamia.** The trajectory plot reads each method's
   unlearned model from `checkpoint-0` of its `--relearn-fft` eval folder:

   ```bash
   cd artifacts/tamia/evals
   SRC=<sweep-id>--<stage>--NNN/checkpoint-<final>
   DST=z7b-wmdp-bio--<N>pp--<method>--relearn-fft/checkpoint-0
   mkdir -p $DST && cp -r $SRC/wmdp-bio-robust $SRC/mmlu-no-bio $DST/
   ```

3. **Move the model on Tamia.** Move the candidate's sweep output to the
   lineage's canonical path, so it survives clearing `artifacts/generated/hpo`
   and later stages such as relearning can use it:

   ```bash
   cd artifacts/tamia/models
   mv -T ../../generated/hpo/<sweep-id>/models/<sweep-id>--<stage>--NNN \
     z7b-wmdp-bio--<N>pp--<method>
   ```

4. **Sync the evals locally.** `artifacts/tamia` is a scratch copy of Tamia's
   artifacts; plots are made from it:

   ```bash
   D=artifacts/tamia/evals/z7b-wmdp-bio--<N>pp--<method>--relearn-fft
   rsync -a tamia:projects/lilpipe/examples/tamper-resistance/$D/ $D/
   ```

5. **Add it to the plot.** Append
   `{"model": "z7b-wmdp-bio--<N>pp--<method>--relearn-fft", "method": "<Method>", "defence": "None"}`
   to the `configs_main` list in
   `scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py`.
   Until relearning runs exist, the method is plotted only at step 0.
6. **Rerun the plot** to the tracked figure names:

   ```bash
   .venv-eval/bin/python scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py \
     --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--<N>pp.pdf
   ```

7. **Test and document.** Run `python -m pytest tests/test_tamper_refactor.py`;
   its lineage test checks every `base.yml`, `selected.yml` and `sweep.yml`
   against the naming and sweep conventions, so no per-value test is needed.
   Record the promoted setting in `docs/experiments.md`.
