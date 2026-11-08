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

`trainers.ghost_inner_products` contains the tensor contractions and `GhostCapture`,
which attaches temporary hooks to an existing model. It computes detached
per-example gradient inner products without materializing full parameter gradients:

```python
from trainers.ghost_inner_products import GhostCapture, causal_lm_losses

with GhostCapture(model, chunk_size=32) as capture:
    outputs = model(input_ids=input_ids)
    losses = causal_lm_losses(outputs.logits, labels)
    gram = capture.gram(losses, retain_graph=True)
losses.mean().backward()
```

For Hugging Face Trainer integration:

```python
from transformers import Trainer
from trainers.ghost_trainer import GhostInnerProductsMixin

class GhostTrainer(GhostInnerProductsMixin, Trainer):
    pass
```

The mixin installs forward and output-gradient hooks in initialization. It leaves
`compute_loss`, `training_step`, backward, accumulation, and optimizer updates
unchanged. `GhostLoggingCallback`, registered automatically, collects each
microbatch's Gram mean and logs `ghost_gram_mean` before the optimizer step.
This is the average of microbatch means, not the Gram of the entire accumulated
batch. The statistic retains the existing loss normalization and gradient
accumulation scaling; AMP loss scaling is removed. Evaluation under no-grad is
ignored. Hooks are removed at normal training completion; call `trainer.close()`
explicitly after an interrupted or failed run.

For Axolotl, add these fields to an existing training YAML:

```yaml
plugins:
  - axolotl_plugins.custom_trainer_args.CustomTrainerArgsPlugin
trainer_cls: trainers.ghost_axolotl.GhostAxolotlTrainer
sample_packing: false
gradient_checkpointing: false
```

Run with `PYTHONPATH=src .venv-train/bin/axolotl train path/to/config.yml`.

Supported trainable parameter owners are Linear, Embedding, LayerNorm, RMSNorm,
HF Conv1D, and HF RMSNorm modules. Reused layers and tied weights include cross
terms. Linear weights use tensorized `ztz` and `dtd` contractions; temporary
memory scales as `[B,B,T,S]`. Shared parameters retain factors until the
microbatch ends. Norm parameters use small local VJPs with capture suppressed.
Modules must use independent, batch-first examples. GPT-2 requires explicit
`position_ids = torch.arange(sequence_length, device=input_ids.device).expand_as(input_ids)`
in each training batch because its default positional embeddings broadcast across
examples. Batch-coupled operations, direct parameter uses outside supported modules,
gradient checkpointing, packed batches, distributed training, and multiple backward
passes per microbatch are unsupported. The Gram becomes available after backward;
it is detached and cannot directly define a differentiable orthogonality penalty.

Run the numerical and Trainer regression tests from this directory with:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest ../../tests/test_ghost_inner_products.py
```
