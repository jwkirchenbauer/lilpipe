# Analysis scripts

Scripts are grouped by experiment under `scripts/analysis/<experiment-name>/`,
with underscore filenames beginning with `get_` or `plot_`. Separate 2D and 3D
plotters remain available. Arguments, defaults, and output paths are unchanged.
Slurm wrapper filenames remain unchanged.

Historical retirement inventories use the former names listed below.
All paths in this table are relative to `scripts/analysis/`.

| Former filename | Current path |
| --- | --- |
| `gen_ll.py` | `loss-landscapes/get_ll.py` |
| `plot_ll_2d.py` | `loss-landscapes/plot_ll_2d.py` |
| `plot_ll_3d.py` | `loss-landscapes/plot_ll_3d.py` |
| `get_param_norm.py` | `parameter-norms/get_param_norm.py` |
| `plot_layerwise_norms.py` | `parameter-norms/plot_layerwise_norms.py` |
| `gen_seam_toy.py` | `toy-experiments/get_seam_toy.py` |
| `plot_seam_toy.py` | `toy-experiments/plot_seam_toy.py` |
| `gen_locking_toy.py` | `toy-experiments/get_locking_toy.py` |
| `plot_locking_toy.py` | `toy-experiments/plot_locking_toy.py` |
| `gen_mnist_basin.py` | `toy-experiments/get_mnist_basin.py` |
| `plot_mnist_basin.py` | `toy-experiments/plot_mnist_basin.py` |
| `train_mnist.py` | `toy-experiments/get_mnist.py` |
| `plot_trajectory_wmdp.py` | `relearning-trajectory/plot_trajectory_wmdp.py` |
| `plot_trajectory_beavertails.py` | `relearning-trajectory/plot_trajectory_beavertails.py` |

Model-distance, smoothness, and perturbed-unlearning analysis implementations
were removed along with their dedicated model-distance and smoothness tests.
Historical configurations and results remain preserved.

## CIFAR-10 defense and attack toy

Four scripts under `scripts/analysis/toy-experiments/` run and plot the comparison:

- `run_cifar_defence_and_attack.py`: random initialization, BaSFT, TAR, or SAM,
  followed by an ordinary cross-entropy attack. Saves `train.pt`, `relearn.pt`,
  and `results.json`. Checkpoints contain model weights, not optimizer state.
- `get_cifar_landscapes.py`: evaluates shared orthonormal random weight slices
  around the initial and defended models, writing JSON grids.
- `plot_cifar_relearn_trajectory.py`: plots attack LR as rows and metrics as columns.
- `plot_cifar_landscapes.py`: plots radius as rows and method as columns.

Run from this example directory using an environment with torch, torchvision,
numpy, pandas, seaborn, and matplotlib. Set `run_root` to a scratch directory.
The plotters produce PDF only.

```bash
run_root="$SCRATCH/lilpipe/cifar10-tar-sweep"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method random --output "$run_root/random"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method tar --attack-steps 16 --output "$run_root/tar"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method sam-nll --rho 0.01 --output "$run_root/sam-nll-rho0.01"
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method sam-ent --rho 0.01 --output "$run_root/sam-ent-rho0.01"
```

Use TAR folders `tar-k1`, `tar-k8`, and `tar-k32` with the corresponding
`--attack-steps` values. BaSFT uses `--method basft --beta VALUE`.
Every run defaults to seed 0, 600 defense steps, 600 attack steps, batch size 128,
Adam LR 0.001, and a fixed 1,024-image test subset. Attack loss and gradient norm
are measured on the current batch every 10 steps. SAM (nll) minimizes negative
cross-entropy; SAM (ent) and TAR minimize negative entropy. TAR uses a fresh Adam
attacker and a first-order gradient through its endpoint. SAM uses global-L2
perturbations and a detached perturbation for its outer gradient.

To attack an existing defense with a larger LR, skip defense with `--checkpoint`:

```bash
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method tar --checkpoint "$run_root/tar/train.pt" \
  --relearn-learning-rate 0.01 --output "$run_root/attack-lr0.01/tar"
```

Repeat for each method, using the same folder names under `attack-lr0.01`.
For the random baseline omit `--checkpoint` and use `--method random`.

```bash
python scripts/analysis/toy-experiments/plot_cifar_relearn_trajectory.py --root "$run_root"
python scripts/analysis/toy-experiments/get_cifar_landscapes.py \
  --root "$run_root" --radii 1000 5000 --points 9
python scripts/analysis/toy-experiments/plot_cifar_landscapes.py \
  --input artifacts/figures/resnet18-cifar10-tar-loss-landscapes.json
```

Landscapes use a fixed 1,024-image training subset and seed-0 random initialization.
`--resume` reuses saved grids with matching settings and checkpoint paths; use
immutable checkpoint paths or omit `--resume` after replacing checkpoint contents.

### Bad-minimum candidates on a fixed subset

The runner additionally supports `bad-target`, `bad-stationary`, `bad-basin`, and
`bad-dead-relu`. The first targets NLL at the subset's empirical class entropy;
`bad-stationary` adds `--gradient-weight` times the squared full-batch NLL gradient
norm, differentiating through that gradient. `bad-basin` also adds `--beta` times
an escape penalty comparing the center with random and projected NLL-descent
perturbations (`--rho`, `--probe-steps`, and `--attack-learning-rate`). Both
endpoint losses remain differentiable; the selected perturbation is detached.
`--curvature 0` asks for no nearby descent, without imposing a strict quadratic
margin in symmetry directions.

`bad-dead-relu` is a constructive control, not a learned defense: GroupNorm scales
are zero and offsets are negative, so ReLU features are locally inactive; the
classifier predicts empirical class frequencies. Parameters remain trainable.
This demonstrates a degenerate bad minimum with flat body directions and
nonnegative head curvature, but destroys all classification utility.

```bash
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method bad-basin --train-samples 512 --batch-size 512 --steps 100 \
  --relearn-steps 100 --log-every 1 --gradient-weight 0.1 --beta 10 \
  --rho 1 --curvature 0 --probe-steps 8 --attack-learning-rate 0.01 \
  --output "$SCRATCH/lilpipe/cifar10-bad-minima/basin10"
```

For local geometry, the evaluator supports `--directions curvature` (the two
lowest Lanczos Ritz directions) and `--directions descent` (the NLL gradient and
an orthogonal curvature direction). Hessian-vector products are exact autograd
products accumulated across the full subset; `--batch-size` controls memory.
Each saved model includes NLL, training accuracy, gradient norm, estimated minimum
Ritz eigenvalue, and its eigenvector residual. A negative Rayleigh quotient gives
an escape direction; nonnegative estimates do not certify a local minimum.

```bash
python scripts/analysis/toy-experiments/get_cifar_landscapes.py \
  --root "$SCRATCH/lilpipe" \
  --indices "$SCRATCH/lilpipe/cifar10-bad-minima/basin10/indices.json" \
  --model "Escape 10" "$SCRATCH/lilpipe/cifar10-bad-minima/basin10/train.pt" \
  --directions curvature --lanczos-steps 20 --batch-size 32 \
  --radii 0.1 1 10 --points 9 \
  --output "$SCRATCH/lilpipe/cifar10-bad-minima/basin10/landscape"
```

An activation-margin candidate (`--method bad-margin`) adds the mean squared
positive part of the final BasicBlock's pre-ReLU input plus `--feature-margin`.
The input is cloned before torchvision's in-place ReLU mutates it. The stronger
run used feature weight 10, margin 1, gradient weight 0.1, and outer Adam LR 0.01:
100 updates from random initialization, then 400 updates from that defended
checkpoint via `--defend-checkpoint` with a fresh optimizer. All attack evaluations
still used 100 full-batch Adam updates at LR 0.001.

`--calibrate-constant-head` sets the classifier bias to log class frequencies,
but rejects checkpoints unless every final ReLU input is strictly negative on the
specified training set. Within an open neighborhood of such a checkpoint, those
features remain zero; bias-only cross-entropy is convex and minimized by those
frequencies. This is a degenerate local minimum with flat body directions.
No parameters are frozen. `--project-feature-margin 0.1` first shifts the last
normalization's per-channel bias by the observed pre-ReLU maximum plus a margin,
which directly constructs the same condition from any initial checkpoint.
These constructions suppress all classification utility on the subset.

```bash
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method random --checkpoint "$SCRATCH/lilpipe/cifar10-fullbatch512/train.pt" \
  --train-samples 512 --batch-size 512 --relearn-steps 100 --log-every 1 \
  --project-feature-margin 0.1 --calibrate-constant-head \
  --output "$SCRATCH/lilpipe/cifar10-bad-minima/projected"
```

`--directions head` evaluates a shared orthonormal classifier-bias plane, useful
for displaying upward curvature where the body-plane landscape is flat.

### Group-gradient cosine without a retain objective

`--method gradient-cosine` minimizes only the stabilized cosine between the
cross-entropy gradients of two random, disjoint halves of each batch. There is
no cross-entropy term in the defense update and no retain set. This reuses the
finite-difference implementation in `src/trainers/tr_gip.py`, including
FP32 perturbations, RNG replay, parameter/buffer restoration, and small-gradient
skipping. A skipped defense update also skips Adam, preventing momentum drift.
The signed objective encourages orthogonal or opposing group gradients; it is
not an all-pairs individual-example orthogonality penalty.

Use the saved random-initialization baseline for an identical starting point:

```bash
PYTHONPATH=src python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method gradient-cosine \
  --defend-checkpoint "$SCRATCH/lilpipe/cifar10-fullbatch512/train.pt" \
  --train-samples 512 --batch-size 512 --steps 100 --learning-rate 0.001 \
  --rho 0.001 --cosine-tau 0.001 --cosine-skip-threshold 1e-8 \
  --optimizer adam --relearn-learning-rate 0.001 --relearn-steps 100 --log-every 1 \
  --output "$SCRATCH/lilpipe/cifar10-fullbatch512-gradient-cosine-rho1e-3"

python scripts/analysis/toy-experiments/plot_cifar_relearn_trajectory.py \
  --fullbatch-root "$SCRATCH/lilpipe" --fullbatch-tar \
  --run 'Group gradient cosine' \
    "$SCRATCH/lilpipe/cifar10-fullbatch512-gradient-cosine-rho1e-3/results.json" \
  --output artifacts/figures/resnet18-cifar10-fullbatch-gradient-cosine-relearning.pdf
```

The defense history records the group cosine, both gradient norms, regularization
gradient norm, cross-entropy as a diagnostic, and cumulative skipped frequency.
The subsequent attack uses ordinary mean cross-entropy on the same fixed images.

`--method gradient-cosine-exact` is a separate method using second-order autograd
for the same group objective. It does not perturb parameters. The finite-difference
method remains available as `gradient-cosine`; neither is silently substituted for
the other. GPU allocation and peak reserved memory are recorded in the defense
history, along with gradient computation time.

The additional individual-gradient methods are:

- `frozen-normalization`: compute individual stabilized inverse gradient norms,
  detach them, and minimize the raw inner product of weighted group gradients.
- `sampled-individual`: differentiate normalization and estimate the off-diagonal
  individual cosine derivative using `--cosine-contributions K` sampled examples.
- `full-individual`: use every example's finite-difference contribution.

`positive-pair-cosine` is a separate follow-up objective:
`mean(relu(cos(g_i, g_j)))`, excluding self-pairs. It samples
`--cosine-contributions K` pairs uniformly from the batch and uses exact
second-order autograd for each selected positive contribution. Opposing pairs
contribute zero, rather than canceling aligned pairs in the objective. Pair
graphs are processed sequentially. The sampled objective is logged separately
from the full-batch off-diagonal cosine measured during relearning.

Individual gradients are streamed rather than stored in a batch-by-parameter
matrix. Individual methods clone buffers and use functional parameter shifts,
replaying each matching base/perturbed RNG state. They record their actual count
of gradient evaluations and time spent computing per-example norms. These CIFAR
experiments use evaluation mode and GroupNorm, so examples remain independent.

With stabilized normalized gradients `u_i`, the off-diagonal objective is
`(N * ||mean(u)||^2 - mean(||u_i||^2)) / (N - 1)`. The second term is differentiated
as well: stabilized `u_i` do not have constant unit norm. The full individual
estimator is checked against this exact higher-order objective in FP64; the
frozen method is checked with its normalization weights explicitly detached.

Add `--cosine-diagnostic-samples 32 --cosine-diagnostic-every 10` to any run,
including a baseline or an attack loaded with `--checkpoint`, to record cosine
throughout relearning. Diagnostics always use the same cohort and leave optimizer
gradients unchanged. Individual off-diagonal cosine and group cosine are logged
separately. They use an exact streaming identity, with no contribution sampling.
Use `--disable-tf32` for the FP32 comparison, especially for finite differences.

Named runs can be plotted without the historical defaults:

```bash
python scripts/analysis/toy-experiments/plot_cifar_relearn_trajectory.py \
  --run Base /path/to/base/results.json \
  --run 'Exact group cosine' /path/to/group-exact/results.json \
  --metric individual_pairwise_cosine \
  --output artifacts/figures/cifar-gradient-cosine-during-relearning.pdf
```

### Fully differentiated TAR

`--method tar-full` differentiates the outer negative-entropy objective through
every inner Adam update, including first and second moments and bias correction.
It uses out-of-place functional updates, `create_graph=True`, and non-reentrant
checkpointing of each full inner step. No stop-gradient approximation is applied.
Adam's square-root computation is guarded at exactly zero moments to avoid
spurious nonfinite meta-gradients. The composite Adam update remains unchanged.

```bash
python scripts/analysis/toy-experiments/run_cifar_defence_and_attack.py \
  --method tar-full --attack-steps 16 --steps 100 --train-samples 512 \
  --batch-size 512 --learning-rate 0.001 --attack-learning-rate 0.001 \
  --relearn-learning-rate 0.001 --relearn-steps 100 --log-every 1 \
  --output "$SCRATCH/lilpipe/cifar10-fullbatch512-tar-full-k16"
```

The 16-step outer gradient is checked against finite differences on a small model,
with checkpointing enabled and disabled; functional Adam is checked against
PyTorch Adam. This mode needs substantially more memory and compute than the
first-order `tar` mode. A one-update GPU smoke test should precede a full run.

A GPU audit of the full 16-step derivative on GroupNorm ResNet-18 (four fixed
images, float64) found identical checkpointed/uncheckpointed gradients. A tested
classifier-bias directional derivative matched finite differences at epsilon
1e-9 (absolute error 4.4e-7), and functional Adam's endpoint matched PyTorch
Adam within 4.7e-13 per parameter. Coarser finite differences did not converge;
the unrolled attack is highly sensitive to finite perturbations. A smooth
Softplus control also converged as epsilon decreased. These checks support the
implementation but do not establish that a full hypergradient improves a longer
attack or that the outer learning rate is appropriate.
