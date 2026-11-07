# SEAM experiments on Tamia

Run commands from `examples/tamper-resistance`. The defended starting checkpoint
is `artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config`.
The authors' implementation remains under `../seam/src`; their training-only
attack entry point is `../seam/scripts/train.py`.

## Training sources

Each Axolotl lineage has one `base.yml` and one `sweep.yml` under
`configs/stages-beavertails/1_relearn/`:

- `llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--data-overlapping--relearn-fft/`
  uses our SEAMRelearnTrainer. It defines the two 250-step LRs, four 1,000-step
  LRs, and the 250-step `2e-4` run with 50% warmup.

The sweep owns parameter combinations, naming, resources, source dependencies,
and evaluations. No manually maintained per-LR entries remain in the canonical
model registry. That registry contains reusable starting checkpoints, noise
producers, and fixed models.

## Generate, inspect, submit

For either lineage:

```bash
python scripts/hpo.py generate configs/stages-beavertails/1_relearn/LINEAGE/sweep.yml
lilpipe artifacts/generated/SWEEP_ID/experiment.yml --dry-run
lilpipe artifacts/generated/SWEEP_ID/experiment.yml --models MODEL_ID \
  --record-dir artifacts/tamia/runs/EXPERIMENT
```

Generation does not submit jobs. `--record-dir` archives the manifest, registries,
referenced configs/launchers, Git version/diff, and job IDs. Partial submissions
are recorded immediately, and existing record directories are rejected. `config_mode: overrides` writes four files:

```text
artifacts/generated/SWEEP_ID/
  base.yml        Exact shared source snapshot
  models.yml      Generated model paths, commands, and producers
  experiment.yml  Sweep-independent pipeline definition
  runs.yml        Per-run overrides and effective configuration records
```

Each producer invokes one training run using the base YAML plus CLI overrides.
The generated model ID is used for the output path and W&B name. `name_template`
formats scalar fields, with a group-specific template for the 50%-warmup suffix.
`model_root: artifacts/tamia/models` keeps weights outside the generation folder.
Scientific LR names normalize `e-05` to `e-5`; invalid placeholders and duplicate
IDs are rejected. Re-generation rejects a changed shared base snapshot; use a
fresh generation directory rather than silently changing an existing base.

Older sweeps retain `config_mode: resolved` by default, including indexed IDs,
per-run concrete configs, and their previous output paths. This mode supports
nested dataset/custom-trainer overrides that cannot be expressed as scalar CLI
arguments. The example-local generator handles these details; the root lilpipe
compiler still only sees concrete models, producer commands, and evaluations.

## Execution and dependencies

Our `train.sbatch` forwards Axolotl overrides and defaults to `--launcher python`
for layer-sharded training. Our relearn trainer uses completion-only labels and
fixed balanced sampling. Only harmful CE updates the model; benign gradients are
used for `loss_sd` / `cosine_similarity` logging. Online W&B uses
`marawan-gamal/lp-tamper-resistance`.

Author-codebase training is intentionally outside the Axolotl stage configs.
`seam_codebase_train_seam_relearning.sbatch CONFIG MODEL OUTPUT LR [AUTHOR_ARGS...]`
passes the authors' native YAML directly to their training script. For example:

```bash
sbatch scripts/slurm/tamia/seam_codebase_train_seam_relearning.sbatch \
  ../seam/config/eval.yaml MODEL OUTPUT 2e-4 --max_steps 250
```

This preserves the authors' input format, full-sequence labels, random sampler,
and optimizer settings. The wrapper sets model/output/LR/run name and enables
model-only checkpoints every 50 steps. Other options are passed through unchanged.
The native config remains under `../seam/config/`, just like the defense configs;
there is no author base/sweep/selected config under our `configs/stages-*` tree.
Author attack scheduling or LR loops live under `scripts/` rather than the
Axolotl model registry/generator. The historical packed launcher remains under
`scripts/misc/seam_codebase_train_seam_relearning_sweep.sbatch`.

The author environment defaults to `$SCRATCH/lilpipe/seam-official/venv`, with
its original offline W&B default. No author-codebase handling is embedded in
`scripts/hpo.py`. Our Axolotl runs continue to use four-GPU layer sharding and
one generated producer per LR, followed by its evaluation. Allocation sharing
is a separate scheduling feature, not implicit sweep behavior.

The authors' defense and original evaluation remain in the separately named
`seam_codebase_train_seam_defence.sbatch` and
`seam_codebase_eval_seam_defence.sbatch` launchers.

## Models, evaluations, and provenance

```text
artifacts/tamia/models/MODEL_ID/checkpoint-N/
artifacts/tamia/evals/MODEL_ID/checkpoint-N/
artifacts/tamia/runs/EXPERIMENT/
```

Models and evaluations mirror one another. Generated definitions and effective
settings remain under `artifacts/generated/`; archive them with submitted source,
commands, job IDs, and W&B references under the run record when submitting.
Historical `submitted/` snapshots, results, and figures are immutable.

The standalone `eval_harmfulness_and_capabilities.sbatch` evaluates four zero-shot
capability tasks and distributed DeBERTa harmfulness on 100 prompts. Set
`INCLUDE_GSM8K=1` for five-shot GSM8K, `INCLUDE_HARMFULNESS=0` for capability-only
runs, or `HARMFULNESS_ONLY=1` for harmfulness only. `CAPABILITIES_SUBDIR=capabilities-4gpu`
retains the starting-model layout. GSM8K is stored separately under `gsm8k-5shot/`.

`--max-step N --checkpoints` evaluates every 50 steps through N; explicit `--steps`
and `--baseline` support other retained schedules. Range environment overrides
can split evaluation across allocations. `RESUME_EVALUATION=1` skips valid complete
stages. CUDA caches are node-local, and `EVAL_LOCAL_LOG=1` archives local logs on
exit. Final-model evaluations reuse the terminal checkpoint only after model,
tokenizer, and generation configuration hashes match. No weights are deleted.

## Noise and results

`noise.sbatch` calls `scripts/analysis/parameter-noise/perturb_model.py`. It scales
a seeded Gaussian vector to the requested percentage of the starting model's
parameter norm. BF16 rounding can change the realized radius; `perturbations.json`
records target and realized norms, seed, and PyTorch version.

The independent baseline manifests `llama2-7b-base-noise.yml` and
`llama2-7b-seam-noise.yml` select starting/noised models and evaluations. They do
not enumerate attack LRs or execute training sweeps. Results YAMLs collect saved
metrics independently of how those models were produced:

```bash
lilpipe results configs/results/llama2-7b-gaussian-noise.yml
```

Saved norm measurements remain under the corresponding parameter-norm run
folders. `docs/results.md` contains the recorded tables. Weight-removal records
remain under run provenance; deleting trajectories never removes their results.

## Plots

`plot_trajectory_beavertails.py` has a flat config list with commented blocks for
author 250-step attacks, our 250-step attacks, and our four 1,000-step attacks.
Uncomment the desired block and individual entries; set its steps and output.
Choose the color or row layout, optionally uncomment GSM8K flexible extraction
and the Base reference. Missing requested results are errors. Only `--rootdir`
and `--outpath` remain as location overrides.

`plot_gradient_cosine.py` reads archived Trainer histories. Repeat
`--input LR TRAINER_STATE_JSON` to select other runs. LR controls thickness and
markers, and the dashed horizontal line marks zero cosine.

Temporary submission and plotting orchestration lives under `scripts/misc/`.
