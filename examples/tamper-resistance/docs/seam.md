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
