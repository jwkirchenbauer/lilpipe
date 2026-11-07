# Experiment notes

Run commands from `examples/tamper-resistance`.

On Tamia, `artifacts` is a symlink to
`/project/aip-grabus/mgamal/lilpipe/examples/tamper-resistance/artifacts`.
Use `rsync --keep-dirlinks` when transferring directories into this checkout so
that transfers preserve the configured artifact root.

## TAR and SAM relearning trajectories

The existing `configs/experiments/z7b-wmdp-bio-relearn-128--5pp.yml` includes
matching attacks on MaxEnt + TAR K=16 and MaxEnt + SAM (`rho=1e-2`). Select
only their `--relearn-fft-lr5e-6-ms128` model IDs for this comparison. Both
use 128 optimizer steps, an initial learning rate of `5e-6` with linear decay,
microbatch size 1, gradient accumulation 1, training seed 1, and data seed 42.

Save model-only checkpoints every 16 steps and keep all eight checkpoints
(16 through 128); use the original defended model for step 0. Preserve these
weights for later projection onto the seeded loss-landscape directions and
contour trajectory plots. The manifest evaluates WMDP-Bio robust and
MMLU-no-bio at steps 0, 16, ..., 128 after each attack succeeds. No cleanup
is scheduled.

From the example root, inspect the planned commands before submission:

```bash
tar_model=z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr5e-6-ms128
sam_model=z7b-wmdp-bio--5pp--unlearn-fft-me-sam-rho1e-2--relearn-fft-lr5e-6-ms128

lilpipe configs/experiments/z7b-wmdp-bio-relearn-128--5pp.yml \
  --models "$tar_model" "$sam_model" \
  --sbatch-args='--partition=gpubase_bynode_b1' \
  --dry-run
```

Remove `--dry-run` to submit. Before submission, check that the canonical
output paths are absent or already contain the intended complete trajectories;
reuse complete outputs rather than overwriting them. If rerunning an attack
whose evaluation directory already exists, move that directory into
`artifacts/tamia/evals/archive/` before submission so old and new results
are never mixed.

## TODO: Gaussian pre-relearning attack

After the queued 1,000-step SEAM trajectories finish, run a separate 250-step
attack sweep that adds an independent parameter delta
$\delta_i \sim \mathcal{N}(0, \sigma^2)$ before ordinary relearning. Use
`trainers.rl_perturb.GaussianBF16Trainer`, a fixed `gaussian_seed`, and sweep
`gaussian_sigma`. The trainer writes the perturbed model to `checkpoint-0`, so
evaluate that checkpoint before checkpoints 10 through 250. Keep attacks
serialized as attack, evaluate every retained checkpoint, cleanup, then start
the next setting. Retain 26 checkpoints so step 0 and all 25 ten-step attack
checkpoints survive rotation. Record `gaussian_sigma` and `gaussian_seed` in
every evaluation summary. Do not include this sweep in the 1,000-step
experiment.

## Results

### BeaverTails legacy SEAM reproducibility

The legacy trainer computes global BF16 gradient norms by iterating over a
Python set in `gradient_dict_dot`. Consequently, `seed: 42` does not fully
specify the defense: the process-level `PYTHONHASHSEED` changes the reduction
order and can send training into a different basin. Pin `PYTHONHASHSEED=0`
when launching the legacy defense. A follow-up that retained this order but
promoted the accumulation to FP32 did not reproduce self-destruction, so the
canonical reproduction retains the authors' BF16 accumulation.

Under the disjoint relearning attack with learning rate `1e-4`, the newly
trained hash-0 checkpoint reproduces the historical self-destruction:

| Defense | Step | Harmfulness | Capability |
| --- | ---: | ---: | ---: |
| Historical legacy | 0 | 4.30 | 55.84 |
| Historical legacy | 100 | 0.28 | 25.74 |
| Historical legacy | 200 | 0.19 | 25.68 |
| Clean canonical rerun | 0 | 4.62 | 55.78 |
| Clean canonical rerun | 100 | 0.34 | 25.70 |
| Clean canonical rerun | 200 | 0.34 | 25.63 |
| FP32 accumulation | 0 | 4.62 | 55.78 |
| FP32 accumulation | 100 | 72.90 | 54.23 |
| FP32 accumulation | 200 | 74.89 | 53.82 |

Generate the regular and alignment attack trajectories from the example root:

```bash
python scripts/analysis/relearning-trajectory/plot_trajectory_wmdp.py --plot main
python scripts/analysis/relearning-trajectory/plot_trajectory_beavertails.py --plot main
```

The scripts write PDF and PNG figures to `artifacts/figures`.

### Hyperparameter sweeps

The TAR-k16 `--ms250` initial candidate (`--hpo--grid-001--001`) ran 250 steps
with learning rate `1e-5`, retain coefficient `4`, and the existing 16-step
inner attack. Training `659169` and paired evaluation `659170` both completed
with exit code `0:0` on 2026-11-01. Only `checkpoint-250` was retained; the
paired evaluator loaded the final saved model and wrote its summary under
`checkpoint-250`.

| Candidate | WMDP-Bio robust | MMLU-no-bio | MMLU drop from 58.59% | 5pp eligible |
| --- | ---: | ---: | ---: | --- |
| TAR-k16 `--ms250`, LR `1e-5`, retain `4` | 43.66% | 53.33% | 5.26pp | No |

The minimum eligible MMLU score is 53.59%; this candidate misses it by 0.26pp.
It was not promoted. The 125-step lineage remains unchanged.

The `--ms250` sweep now also contains a `grid-002` stage over learning rates
`{1e-6, 5e-6, 1e-5}` and retain coefficients `{0.1, 1, 5}`. The nine new
candidates keep all other training settings, including the final-only checkpoint
at step 250. The completed `grid-001--001` initial candidate is preserved and excluded
from this submission. The source sweep remains at
`configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--ms250/sweep.yml`.
The generated `grid-002.yml` manifest selects only these nine candidates; its dry run
contains nine training jobs and nine dependent paired evaluations. Submitted
2026-11-01; select the lowest WMDP score among candidates with MMLU at least 53.59%.

The stages, generated configs, model/evaluation paths, and W&B display names
use `grid-001` and `grid-002`. Slurm job IDs are unchanged. See the stage tables in
[results.md](../configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--ms250/results.md).

| Candidate | Learning rate | Retain coefficient | Training job | Paired evaluation job | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| grid-002--001 | 1e-06 | 0.1 | 659654 | 659663 | Completed |
| grid-002--002 | 5e-06 | 0.1 | 659655 | 659664 | Completed |
| grid-002--003 | 1e-05 | 0.1 | 659656 | 659665 | Completed |
| grid-002--004 | 1e-06 | 1 | 659657 | 659666 | Completed |
| grid-002--005 | 5e-06 | 1 | 659658 | 659667 | Completed |
| grid-002--006 | 1e-05 | 1 | 659659 | 659668 | Completed |
| grid-002--007 | 1e-06 | 5 | 659660 | 659669 | Completed |
| grid-002--008 | 5e-06 | 5 | 659661 | 659670 | Completed |
| grid-002--009 | 1e-05 | 5 | 659662 | 659671 | Completed |

All nine training jobs and paired evaluations completed with exit code `0:0`.
Best feasible candidate: `--hpo--grid-002--009`, LR `1e-5`, retain `5`;
WMDP 45.05%, MMLU 54.54%, MMLU drop 4.05pp. No promotion has been performed.
Detailed scores are in the linked `results.md`.

The table records the Zephyr-7B sweeps by unlearning-method family. Each named
stage in `sweep.yml` is an independent grid; only parameters within the same
stage form a Cartesian product. Other settings were inherited from the method's
`base.yml`; promoted values are recorded in `selected.yml`.

| Method | Learning rate | $\lambda_{\mathrm{ret}}$ | $\lambda_{\mathrm{steer}}$ | $\rho_{\mathrm{SAM}}$ | $\lambda_{\text{weight-dist}}$ | $\lambda_{\mathrm{AO}}$ | Runs | Promoted setting |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| GD | `{5e-7, 1e-6, 2e-6, 2.5e-6, 3.28e-6}` | `{1, 3, 4, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.75, 5, 10}` | - | - | - | - | 23 | learning rate `2.5e-6`; retain `4` |
| MaxEnt | `{1e-6, 5e-6, 1e-5}` | `{0.1, 0.25, 0.5, 1, 2, 4, 8}` | - | - | - | - | 13 | learning rate `5e-6`; retain `0.25` (promoted to 5pp) |
| MaxEnt + SAM | `{1e-6, 5e-6, 1e-5}` | `{0.1, 0.25, 0.5, 1, 2, 4, 8}` | - | `0.01` | - | - | 13 | learning rate `5e-6`; retain `4` (promoted to 5pp) |
| MaxEnt + WD (5pp) | from 5pp MaxEnt: `5e-6` | from 5pp MaxEnt: `0.25` | - | - | `{0, 1e2, 1e3, 1e4, 1e5, 1e6}`; refine `{2e4, 3e4, 5e4, 7e4}` | - | 10 | weight distance `5e4` (promoted to 5pp) |
| MaxEnt + TAR (5pp) | `{5e-6, 1e-5}` | `{0.1, 0.25, 0.5, 0.75, 1, 2, 4}` | - | - | - | - | 13 | inner steps `8`; learning rate `1e-5`; retain `2` (promoted to 5pp); retain `0.1` promoted to 15pp as an exception (MMLU drop 15.59, 0.59 over budget) |
| GD + SEAM (5pp) | `{1e-6, 5e-6, 1e-5}`; eps stage `{5e-6, 1e-5}` | `4` | - | - | - | - | 36 | learning rate `1e-5`; SEAM coefficient `10` (of `{0.01, 1}`, then `{3, 10, 30}`); epsilon `1` (of `{1e-4, 1e-3, 1e-2}`, then `{1, 3, 10}`) (eps--008, promoted to 5pp; WMDP-Bio robust 25.58, MMLU drop 3.19). The first grid's pick (grid--018) was retired: epsilon <= 1e-2 rounds away in bf16, so its SEAM term was noise and the model was effectively GD |
| MaxEnt + TAR, 16 inner steps (5pp) | `{5e-6, 1e-5}` | `{0.1, 0.25, 0.5, 0.75, 1, 2, 4}` | - | - | - | - | 10 | learning rate `1e-5`; retain `4` (promoted to 5pp) |
| MaxEnt + TAR, 16 inner steps, 250 outer steps (5pp) | search `{1e-6, 5e-6, 1e-5}`; initial `1e-5` | search `{0.1, 1, 5}`; initial `4` | - | - | - | - | 10 | Initial candidate: WMDP 43.66%, MMLU 53.33% (drop 5.26pp), ineligible. Nine-point search completed; best feasible LR `1e-5`, retain `5`: WMDP 45.05%, MMLU 54.54% (drop 4.05pp); not promoted |
| GD + SAM | `{2.5e-6, 3.28e-6, 5e-6}` | from GD: `1` | - | `{0.001, 0.01, 0.1}` | - | - | 6 | learning rate `2e-6`; retain `1`; rho `0.01` |
| GD + WD | - | - | - | - | `{1e5, 3e5, 1e6, 3e6, 5e6}` | - | 5 | weight distance `3e5` |
| GD + AO | from GD: `2.5e-6` | `{1.5, 2, 3, 4, 5, 7, 10}` | - | - | - | `{0, 0.001, 0.003, 0.005, 0.01, 0.1, 1, 10, 20, 25, 30, 35, 40, 60, 80, 100}` | 25 | activation orthogonality `40`; retain `4` |
| NPO | `{5e-7, 1e-6, 3e-6}` | `{1, 3, 10}` | - | - | - | - | 9 | learning rate `1e-6`; retain `1` |
| NPO + SAM | - | `{0.5, 1, 2}` | - | `{0.001, 0.01, 0.1}` | - | - | 9 | retain `1`; rho `0.001` |
| NPO + WD | - | `{0.5, 1, 2}` | - | - | `{100, 1000, 10000}` | - | 9 | retain `1`; weight distance `1000` |
| NPO + WD (5pp) | from 5pp NPO: `5e-6` | from 5pp NPO: `1` | - | - | `{0, 1e2, 1e3, 1e4, 1e5, 1e6}` | - | 6 | weight distance `1e4` (promoted to 5pp) |
| RMU | - | `{300, 600, 1200}` | `{3.25, 6.5, 13}` | - | - | - | 9 | retain `600`; steering `6.5` |
| RMU + AO | from RMU: `5e-5` | from RMU: `600` | from RMU: `6.5` | - | - | `{0, 0.01, 0.03, 0.05, 0.07, 0.1, 1, 10, 100}` | 9 | activation orthogonality `0.01` |
| RMU + SAM | - | `{600, 1200, 2400}` | - | `{0.001, 0.01, 0.1}` | - | - | 9 | retain `600`; rho `0.001` |
| RMU + WD | - | `{600, 1200, 2400}` | - | - | `{100, 1000, 10000}` | - | 9 | retain `1200`; weight distance `100` |

GD + AO inherited the promoted GD learning rate and forget coefficient, then
used six independent sweep stages in one manifest. The canonical relearning learning-rate sweep
tested `{2.5e-6, 5e-6, 7.5e-6, 1e-5}` before promoting `1e-5` for the 125-step
long attack. RMU + AO keeps the promoted RMU settings and trainable layers
fixed, targets AO at layers 5--7, and varies only the AO coefficient.

RMU + AO selected the strongest positive AO candidate satisfying a maximum
2-point MMLU drop from the 58.59 base-model score:

| $\lambda_{\mathrm{AO}}$ | WMDP-Bio robust $\downarrow$ | MMLU-no-bio $\uparrow$ | MMLU drop | Eligible |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 27.65 | 56.72 | 1.87 | yes |
| **0.01** | **28.23** | **57.03** | **1.55** | **yes; selected** |
| 0.03 | 34.22 | 57.05 | 1.54 | yes |
| 0.05 | 41.01 | 56.97 | 1.61 | yes |
| 0.07 | 39.98 | 56.73 | 1.85 | yes |
| 0.1 | 37.90 | 56.53 | 2.06 | no |
| 1 | 29.72 | 55.75 | 2.84 | no |
| 10 | 27.88 | 52.06 | 6.53 | no |
| 100 | 25.81 | 44.12 | 14.46 | no |

#### MiniCPM5-1B

MiniCPM configs and registry entries have been removed. The search settings and
measurements below are historical records; this workflow is no longer runnable
from the checked-in configs.

| Method | Learning rate | $\lambda_{\mathrm{ret}}$ | $\lambda_{\mathrm{steer}}$ | $\rho_{\mathrm{SAM}}$ | $\lambda_{\text{weight-dist}}$ | $\lambda_{\mathrm{AO}}$ | Target layer | Updates | Runs | Promoted setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| GD | `{5e-7, 1e-6, 2.5e-6, 5e-6, 1e-5}` | `[0.25, 16]` (log) | - | - | - | - | - | `{50, 100, 150, 250}` | 24 | learning rate `1e-6`; retain `0.630454`; 100 updates |
| GD + SAM | - | - | - | - | - | - | - | - | - | - |
| GD + WD | - | - | - | - | - | - | - | - | - | - |
| GD + AO | - | - | - | - | - | - | - | - | - | - |
| NPO | - | - | - | - | - | - | - | - | - | - |
| NPO + SAM | - | - | - | - | - | - | - | - | - | - |
| NPO + WD | - | - | - | - | - | - | - | - | - | - |
| RMU | `{2.5e-5, 5e-5, 1e-4}` | `{300, 600, 1200, 2400}` | `{3.25, 6.5, 13, 30, 100, 300}` | - | - | - | `{3..23}` | `{150, 300, 500}` | - | - |
| RMU + SAM | - | - | - | - | - | - | - | - | - | - |
| RMU + WD | - | - | - | - | - | - | - | - | - | - |
| RMU + WD (5pp) | from 5pp RMU: `1e-5` | from 5pp RMU: `300` | from 5pp RMU: `6.5` | - | `{0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6}` | - | 7 | weight distance `1e2` (promoted to 5pp) |

The MiniCPM5-1B GD study minimizes pre-attack WMDP-Bio robust accuracy subject
to MMLU-no-bio accuracy remaining at least `0.5062` (at most a two-point drop
from the base model). Trial 11 was promoted with WMDP-Bio `0.47350` and MMLU
`0.52920`. Trials reaching near-random WMDP accuracy were ineligible because
their MMLU accuracy collapsed to approximately `0.23`.

### Authors OPTML NPO + 5% global Gaussian noise

`configs/experiments/z7b-npo-author-noise.yml` compares the pinned
`OPTML-Group/NPO-WMDP` snapshot `fec7960fb3925037a6befc5819036cbc55664b18`
with a 5% global L2 Gaussian perturbation (seed 42, BF16). The existing noise
generator creates the perturbed model; the standalone WMDP/MMLU launcher evaluates
both models at zero shot on full WMDP-Bio robust and MMLU-no-bio, with four
GPU replicas. Each task has its own evaluation job and mirrored output folder.
There is no relearning stage. Collect the paired scores with
`lilpipe results configs/results/z7b-npo-author-noise.yml`.

The 1% follow-up uses `configs/experiments/z7b-npo-author-noise.yml`, the same
authors checkpoint and Gaussian seed 42, and reuses the completed unperturbed
baseline. The existing results configuration includes the baseline, 1%, and 5%.
