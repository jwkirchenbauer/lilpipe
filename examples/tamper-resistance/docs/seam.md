# SEAM experiments on Tamia

Run commands from `examples/tamper-resistance`. The authors' SEAM implementation
remains in `../seam/src`; our Python entry points live in `../seam/scripts`:

- `train.py`: the authors' relearning attack without evaluation, with final saving.
- `eval_harmfulness.py`: distributed DeBERTa evaluation on their 100 test prompts.
- `eval_defence.py`: the original baseline evaluation wrapper, with GPT judging skipped.
- `verify_checkpoint.py`: validate the saved step and model shards.

Reusable Slurm launchers live in `scripts/slurm/tamia`:

- `train_seam_defence.sbatch CONFIG MODEL RUN [STEPS]` trains the defence; defaults
  to 500 steps and saves the final weights directly under `MODEL`.
- `eval_seam_defence.sbatch MODEL CONFIG OUTPUT RUN` reproduces the older baseline
  evaluation with the authors' environment and utility evaluator.
- `train_seam_relearning_sweep.sbatch MODEL MODELS_ROOT RUN [SMOKE]` runs the five
  learning rates on four independent GPU workers, saving every 50 steps.
- `eval_harmfulness_and_capabilities.sbatch [--checkpoints] MODEL OUTPUT [MODEL OUTPUT ...]`
  runs our faster four-GPU evaluations. `--checkpoints` evaluates steps 50–250 and
  reuses step 250 for the final model after checking identical weights and configs.

The SEAM environment defaults to `$SCRATCH/lilpipe/seam-official/venv`; override
`SEAM_ENV` when needed. Capability evaluation uses the tamper-resistance uv eval
environment. Named defence and baseline-evaluation configs live in `../seam/config`.

## Artifact layout

```text
artifacts/tamia/models/<model>/                  final weights and tokenizer
artifacts/tamia/models/<model>/checkpoint-50/    intermediate model weights
artifacts/tamia/evals/<model>/                   final evaluation results
artifacts/tamia/evals/<model>/checkpoint-50/     intermediate evaluation results
artifacts/tamia/runs/<run>/                      job records, logs, config snapshots, W&B
artifacts/tamia/runs/<run>/submitted/            historical code and submitted scripts
artifacts/tamia/runs/<run>/archive/              configs and wrappers before relocation
artifacts/figures/                              generated PDFs
```

Models and evaluations mirror each other. Runs describe an execution rather than
an evaluated checkpoint; the relearning sweep has one run directory covering all
five learning rates. One-step smoke models also live under `models/`.

Historical records keep the paths that were used at execution time. The
`layout-cleanup-*.json` records map the relocated files and directories; use the
current reusable scripts for new jobs rather than editing submitted snapshots.
On Tamia, run directories also contain the original logs and offline W&B payloads.
The local checkout retains the smaller provenance and configuration snapshots.

The defended baseline's faster evaluations remain in its existing
`capabilities-4gpu/` and `harmfulness-4gpu-bs16/` subdirectories, while their job
metadata is under the corresponding run directories. The trajectory plot uses
these existing results at step zero.

## Additional capability tasks

Both capability launchers accept `CAPABILITY_TASKS`, `NUM_FEWSHOT`, and
`MAX_BATCH_SIZE` environment overrides. Their defaults remain the original four
capability tasks, zero-shot, with a maximum batch size of 128 per GPU.

For standard five-shot GSM8K, submit the existing capability-only launcher:

```sh
CAPABILITY_TASKS=gsm8k NUM_FEWSHOT=5 MAX_BATCH_SIZE=32 \
  sbatch scripts/slurm/tamia/eval_capabilities.sbatch \
  MODEL artifacts/tamia/evals/MODEL_SLUG/gsm8k-5shot
```

Use the same GSM8K settings for the original `meta-llama/Llama-2-7b-chat-hf` starting
model and its SEAM-defended checkpoint. Results contain both strict-match and
flexible-extraction exact match. Keep this task separate from the original
four-task capability mean, and keep the existing zero-shot results unchanged.

Collect the starting-model versus SEAM comparison using the existing results CLI:

```sh
lilpipe results configs/results/llama2-7b-seam-capabilities.yml
```

The report uses the saved four-task results and the separate five-shot GSM8K
results, with strict and flexible answer extraction shown separately.

## Parameter movement during defence

The existing `scripts/analysis/parameter-norms/get_param_norm.py` accepts an
optional `--reference` starting model. It reports the global L2 norms of the
model, reference, and difference, plus `delta_l2_norm / reference_l2_norm` and
layerwise values. Parameters are aligned by name and shape; buffers are excluded.
Subtraction and squared-norm accumulation use float64. The default BF16 loading
matches the SEAM defence's starting parameters, excluding the initial checkpoint's
conversion to BF16 from the measured training movement.

Run it on a Tamia CPU partition with
`scripts/slurm/tamia/analysis_get_param_norm.sbatch --model MODEL --reference BASE
--output artifacts/tamia/runs/RUN/norms.json`. Use the exact cached starting-model
revision as `BASE`. The Slurm script selects CPU execution and requires 96 GiB RAM;
check the available CPU partition with `sinfo` before submitting.


## Globally normalized Gaussian parameter noise

`scripts/analysis/parameter-noise/perturb_model.py` samples one Gaussian vector
across all learned parameters and normalizes it to the requested fraction of the
starting model's global L2 norm. The CLI accepts explicit `--percentages`: for
example, `0.1` means 0.1% (a fraction of 0.001). Each fraction starts from the same original
weights and uses the same noise direction. No layerwise rescaling is applied.
The generator's seed and PyTorch version are recorded, along with the target and
realized delta norm after casting back to BF16. Only the Gaussian-vector norm is
retained; the direction is regenerated from its seed to bound memory use.

Submit model generation with `scripts/slurm/tamia/analysis_perturb_model.sbatch`,
passing `--model`, `--output-prefix`, and `--output` (the run's metadata JSON).
Outputs are named `PREFIX-l2r<FRACTION>-seed<SEED>`. Existing outputs are rejected.

`scripts/slurm/tamia/eval_capability_models.sbatch EVAL_ROOT MODEL [MODEL ...]`
evaluates several saved models in one four-GPU allocation, using the existing
capability launcher for four zero-shot tasks and five-shot GSM8K on each model.
Evaluation directories use each model's basename to mirror `models/`.

The starting-model noise sweep uses **0.1%, 0.25%, 0.5%, and 0.75%** with seed 42.
The obsolete 10–75% attempt was cancelled and its artifacts removed.
Its report reuses the already-completed unperturbed baseline:

```sh
lilpipe results configs/results/llama2-7b-gaussian-noise.yml
```

## Harmful fine-tuning control on the starting model

The relearning sweep launcher also accepts the unmodified cached starting model.
Set `MODEL_SLUG=llama2-7b-beavertail` to keep output names stable when `MODEL`
points to a Hugging Face snapshot directory whose basename is a revision hash.
The authors' training logic, harmful examples, five learning rates, seed 42,
per-GPU batch size 4, one epoch, and 50-step checkpoint interval are unchanged.

Set `INCLUDE_GSM8K=1` for the combined evaluator to add five-shot GSM8K alongside
the four zero-shot capability tasks and DeBERTa harmfulness. GSM8K is saved in
`gsm8k-5shot/` under every checkpoint's evaluation directory. Step 250 is reused
for the final model only after the existing weight/config hash check passes.
With GSM8K enabled, split the five learning rates across evaluation jobs so each
allocation remains below Tamia's three-hour limit.

Set `HARMFULNESS_ONLY=1` on the combined evaluator to run just the existing
four-GPU DeBERTa wrapper; capability environment setup and benchmarks are skipped.
The Base step-zero score is stored at
`artifacts/tamia/evals/llama2-7b-beavertail/harmfulness.json` and is shared by all
five Base learning-rate trajectories. The SEAM baseline keeps its earlier
`harmfulness-4gpu-bs16/metrics.json` location. Both use the same 100 test prompts,
seed 42, and batch size 16 per GPU.

## Noisy SEAM baseline and relearning

The 0.75% SEAM-noise variant starts from the defended checkpoint and uses that
checkpoint's parameter norm. `INCLUDE_HARMFULNESS=1` on
`eval_capability_models.sbatch` adds the shared DeBERTa wrapper after capability
and GSM8K scoring, providing the trajectory's step-zero harmfulness point.

Its dependent harmful fine-tuning sweep uses the same five learning rates,
250 steps, seed 42, and checkpoints every 50 steps. The 25 checkpoint evaluations
run in three batches, followed by a CPU plotting job. The plot adds
`SEAM + 0.75% noise` alongside SEAM and Base; method controls hue and numeric
learning rate controls line width. Long pretrained checkpoint paths are passed
relative to the example root to keep lm-eval result-directory names below the
filesystem's component-length limit.

The four saved Base-noise weight directories were removed after their evaluations
completed, as requested. Their evaluation files, normalization metadata, and
comparison tables remain available.

## Canonical 1% and 2% SEAM-noise workflow

Run commands from `examples/tamper-resistance`. The existing defended checkpoint
`artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config`
is the input. Both perturbations use seed 42 and the same global Gaussian
vector `v`, with `theta_noised = theta + (percentage / 100) * ||theta|| * v / ||v||`.
The source is always the SEAM-defended model. BF16 rounding means the realized
noise radius can differ slightly from its target; `perturbations.json` records both.

```sh
uv run --frozen --group eval python scripts/run_seam_noise.py --dry-run
```

The manifest is `configs/experiments/llama2-7b-seam-noise-relearning.yml`, with
models and evaluation definitions in the existing registries. The runner uses
lilpipe's public `Plan` API to add the plotting stage after every evaluation.
The complete plan contains 15 jobs:

- Two CPU jobs generate the 1% and 2% models independently using the same seed.
- Two four-GPU jobs evaluate their baselines and train the five-LR sweeps.
  `EVALUATE_BASELINE=1` requests four zero-shot capability tasks, five-shot GSM8K,
  and DeBERTa harmfulness before the training smoke test and full attacks.
- Ten four-GPU jobs each evaluate one LR at steps 50, 100, 150, 200, and 250.
  The final checkpoint is reused only after the existing SHA-256 check.
- One CPU job produces both comparison PDFs after all ten evaluation jobs succeed.

Each LR-`2e-5` model entry owns its noise level's shared five-LR training producer.
The other four model entries reference outputs of that producer. Evaluation
stages depend on both training producers, so a sweep is never submitted twice.
The attack settings remain 1,000 examples, batch size 4, one epoch/250 steps,
seed 42, and full parameter fine-tuning with the authors' code.

Remove `--dry-run` to submit. Successful submissions are recorded immediately
under `artifacts/tamia/runs/llama2-7b-seam-noise-relearning/jobs.json`, including
partial submissions if a later `sbatch` fails. An existing run directory is
rejected; use `--run-dir` for a new submission record. Existing model outputs are
also rejected by the generation/training scripts. `--existing-models MODEL_ID ...`
can explicitly skip producers for artifacts already available.

The final figures also use the retained Base, SEAM, and 0.75%-noise evaluation
artifacts as references; their deleted relearning weights are not required.
The earlier `runs/<run>/submitted/` snapshots and recovery scripts are immutable
historical records. They are excluded from formatting and are not the reusable
submission interface.

## Plot variants

Both PDFs use a colorblind method palette, black for Base, solid lines, and LR
markers plus numeric line widths. Configure their model/LR lists in
`scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py`.

```sh
python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py --variant reduced
python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py --variant comparison
```

`reduced` excludes the 0.75%-noise method and uses Base at `1e-4`. `comparison`
includes it and uses Base at `8e-5`. The larger `1e-4` and `2e-4` LRs are currently
enabled for the SEAM methods in both variants, as requested. The existing
`--lr-markers.pdf` filename is retained for the comparison variant; its name no
longer controls the plotting interface. The old `--lr-markers` CLI flag has been
replaced by `--variant comparison`.

Add `--completed-only` while evaluations are in progress. A method/LR is included
only when its baseline and all five plotted checkpoints have results; incomplete
trajectories and their legend entries are omitted. Scores and checkpoint positions
are never jittered. The right-hand utility panel remains the original four-task
mean; GSM8K is reported separately by the results configurations.

`lilpipe results configs/results/llama2-7b-gaussian-noise.yml` includes the Base
noise controls and the SEAM + 0.75%, 1%, and 2% starting models, with a separate
column for each benchmark.
