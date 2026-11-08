# Tamper Resistance

## Prerequisites

- Request access to [`cais/wmdp-bio-forget-corpus`](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
- Update setups in the runner scripts `scripts/slurm/tamia`

The original Tamia workflows use six supported Slurm entry points. WMDP-Bio robust and MMLU-no-bio
run in one paired job, and each trajectory runs in one allocation. See
[Tamia interfaces and retired-file inventory](docs/tamia-refactor.md) for arguments,
checkpoint selection, and opt-in trajectory deletion. Regenerate older HPO
manifests before using them with the consolidated scripts.

Mila uses four supported scripts for training, BeaverTails single-model or
checkpoint evaluation, and parameter-norm analysis. See [Mila interfaces and retired-file inventory](docs/mila-refactor.md).

For the authors' SEAM defence and relearning sweep, see the
[SEAM scripts and artifact layout](docs/seam.md).

## Quickstart

```sh
# 0. Setup
cd examples/tamper-resistance
# NOTE: Training and eval use diff uv envs due to incompatible dependencies
UV_PROJECT_ENVIRONMENT=.venv-train uv sync --locked --group train
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --group eval

# 2. Run a selected 5pp unlearning method
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu/selected.yml

# 3. Run relearning
source .venv-train/bin/activate
axolotl train configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/selected.yml

# 4. Run eval on final ckpt
source .venv-eval/bin/activate
CHECKPOINT=artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/checkpoint-16
OUTPUT=artifacts/tamia/evals/z7b-wmdp-bio--5pp--unlearn-fft-rmu--relearn-fft/checkpoint-16
accelerate launch --num_processes=1 -m lm_eval run \
  --model hf \
  --model_args "pretrained=$CHECKPOINT,dtype=bfloat16,trust_remote_code=True" \
  --include_path lm_eval_tasks \
  --tasks wmdp_bio_robust,mmlu_no_bio \
  --num_fewshot 0 \
  --batch_size 32 \
  --output_path $OUTPUT

# 5. Plot the Zephyr-7B relearning trajectory after the jobs finish
# Note: run the above eval for every ckpt in the relearning trajectory before
# running the plotting code
source .venv-eval/bin/activate
python scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py \
  --plot main \
  --output artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--5pp.pdf
```

## Reproduce Figure 1a using lilpipe (z7b experiments)

```sh
# 1. Run all unlearning and tamper resistance methods and evals
# for the Zephyr-7B model, on SLURM.
lilpipe configs/experiments/z7b-wmdp-bio-unlearn--5pp.yml

# 2. Plot the relearning trajectory 
.venv-eval/bin/python scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py
```

## Hyperparameter optimization for q0.6b

```sh
# 1. Generate and run all defenses and their evaluations.
python scripts/hpo.py generate configs/stages-beavertails/0_alignment/qwen3-0.6b-beavertail--align-fft-seam/sweep.yml
lilpipe artifacts/generated/hpo/qwen3-0.6b-beavertail--align-fft-seam--hpo/experiment.yml

# 2. Generate and run all relearning candidates (only after the defense sweep is done).
python scripts/hpo.py generate configs/stages-beavertails/1_relearn/qwen3-0.6b-beavertail--align-fft-seam/sweep.yml
lilpipe artifacts/generated/hpo/qwen3-0.6b-beavertail--align-fft-seam--relearn--hpo/experiment.yml

# 3. Uncomment the desired configs in the plot script, then plot after evaluations finish.
.venv-eval/bin/python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py
```


Active BeaverTails evaluation code lives in `scripts/evals/`; HPO generation remains
at `scripts/hpo.py`. Retired TOFU, alignment-evaluation, weight-steering, and Optuna
entry points have been removed from this example.

## Project structure
```text
configs/
├── experiments/z7b-wmdp-bio-unlearn--5pp.yml  # models and evaluations to run
├── registries/                               # model producers and evaluation jobs
└── stages-wmdp/0_unlearn/
    └── z7b-wmdp-bio--5pp--unlearn-fft-rmu/
        ├── base.yml                          # training config used by the sweep
        ├── sweep.yml                         # HPO parameter grid
        └── selected.yml                      # optimal training config found
```

Analysis scripts are grouped under `scripts/analysis/<experiment-name>/`;
see [the rename mapping](docs/analysis-scripts.md).

## Ghost gradient inner products

The GIP implementation lives in `src/trainers/tr_gip.py`: shared mathematical
functions, the finite-difference defense trainer `GIPTrainer`, and passive
tracking through `GhostInnerProductsMixin` and `GIPRelearnTrainer`. Functions
remain directly importable for numerical tests without Axolotl.

`trainers.tr_gip` contains the tensor contractions and `GhostCapture`,
which attaches temporary hooks to an existing model. It computes detached
per-example gradient inner products without materializing full parameter gradients:

```python
from trainers.tr_gip import GhostCapture, causal_lm_losses

with GhostCapture(model, chunk_size=32) as capture:
    outputs = model(input_ids=input_ids)
    losses = causal_lm_losses(outputs.logits, labels)
    gram = capture.gram(losses, retain_graph=True)
losses.mean().backward()
```

For Hugging Face Trainer integration:

```python
from transformers import Trainer
from trainers.tr_gip import GhostInnerProductsMixin

class GhostTrainer(GhostInnerProductsMixin, Trainer):
    pass
```

The mixin installs forward and output-gradient hooks in initialization. It leaves
`compute_loss`, `training_step`, backward, accumulation, and optimizer updates
unchanged. `GhostLoggingCallback`, registered automatically, collects each
microbatch's mean off-diagonal gradient cosine and logs `ghost_cosine_mean`
before the optimizer step.
This is the average of microbatch cosine means, not the cosine mean over the entire
accumulated batch. Diagonal pairs and pairs involving zero-norm gradients are
excluded; microbatches with no valid pairs produce no metric. Common loss, gradient
accumulation, and AMP scaling cancel during normalization. Evaluation under
no-grad is ignored. Hooks are removed at normal training completion; call `trainer.close()`
explicitly after an interrupted or failed run.

For Axolotl, add these fields to an existing training YAML:

```yaml
plugins:
  - axolotl_plugins.custom_trainer_args.CustomTrainerArgsPlugin
trainer_cls: trainers.tr_gip.GIPRelearnTrainer
sample_packing: false
gradient_checkpointing: false
```

Run with `PYTHONPATH=src .venv-train/bin/axolotl train path/to/config.yml`.

For passive tracking during unlearning, use `GIPGradDiffTrainer` or
`GIPNPOTrainer` from the same module. They capture only `compute_forget_loss`;
retain forwards and NPO's reference forward are excluded. Each captured output
is counted on its first gradient traversal, so GD's diagnostic gradient calls
and subsequent training backward do not double-count it. The defense objectives
and optimizer gradients are unchanged.

Tracking supports one process on one GPU or a model sharded across GPUs. Layer
Gram contributions are collected on the Trainer device; parameter gradients
stay on their owning devices. Tamia launchers accept `TRAIN_SOURCE_ROOT` to
run against a frozen copy of the source directory.

Supported trainable parameter owners are Linear, Embedding, LayerNorm, RMSNorm,
HF Conv1D, and HF RMSNorm modules. Reused layers and tied weights include cross
terms. Linear weights use tensorized `ztz` and `dtd` contractions; temporary
memory scales as `[B,B,T,S]`. Shared parameters retain factors until the
microbatch ends. Norm parameters use small local VJPs with capture suppressed.
Modules must use independent, batch-first examples. GPT-2 requires explicit
`position_ids = torch.arange(sequence_length, device=input_ids.device).expand_as(input_ids)`
in each training batch because its default positional embeddings broadcast across
examples. Batch-coupled operations, direct parameter uses outside supported modules,
gradient checkpointing, packed batches, distributed training, and combining
different backward objectives for the same captured outputs are unsupported.
The Gram becomes available after the first backward traversal;
it is detached and cannot directly define a differentiable orthogonality penalty.

Run the numerical and Trainer regression tests from this directory with:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest ../../tests/test_ghost_inner_products.py
```

## Finite-difference gradient cosine defense

`trainers.tr_gip.GIPTrainer` minimizes refusal CE plus
`cosine_coefficient` times the stabilized cosine between two randomly split,
equal harmful-completion groups. Harmful losses are per-example mean completion
CE, averaged over each group. The ordinary harmful-loss gradient is never added
to the optimizer update. Refusal CE uses the model's ordinary mean loss.

The initial implementation requires FP32 parameters, one device/process,
unpacked batches, and no AMP or gradient checkpointing. Each batch contains
half harmful completions and half refusals, with batch size divisible by four.
Use `data.beavertails_hh` with roles `hh` and `hb`, `include_role: "true"`, and
`remove_unused_columns: false` to retain the metadata used for balanced sampling.

`custom_trainer_args` exposes `cosine_coefficient` (lambda), `cosine_tau`,
`cosine_rho`, and `cosine_skip_threshold`. The detached finite-difference helper
uses up to four harmful-group evaluations plus one retain evaluation per step;
zero directions skip their perturbation and small base norms skip regularization.
Setting lambda to zero runs only ordinary refusal training. Perturbations always
start at the original weights, replay each group's torch dropout randomness,
and restore exact parameter/buffer copies even on exceptions. Optimizer state
is never touched by these evaluations. This implementation does not use ghost
Gram matrices or individual-example parameter gradients.

W&B metrics include `train/gradient_cosine`, `train/gradient_norm_a`,
`train/gradient_norm_b`, `train/regularization_gradient_norm`, `train/loss_retain`,
and `train/regularization_skip_frequency`. The cosine compares group-average
gradients, unlike `train/ghost_cosine_mean`, which averages individual pairs.

`configs/experiments/qwen3-06b-gradient-cosine-defense.yml` compares the defense
against a matched lambda-zero refusal control, then applies the same 250-step
harmful attack to both and evaluates harmfulness before and every 50 attack steps.
The initial coefficients are experiment settings, not tuned hyperparameters.
