Earlier noise, norm, and ablation reports: [results history](results-history.md).

| Stream | Authors | Ours before | Ours, refusal space | Ours, harmful + refusal spaces |
|---|---|---|---|---|
| Harmful (HH) | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer: {harmful answer}` |
| Refusal (HB) | `Answer: {refusal}` | `Answer:\n{refusal}` | `Answer: {refusal}` | `Answer: {refusal}` |
| Benign (BB) | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` |

The refusal-space variant changes only the refusal separator. The new variant
also replaces the harmful separator with a space. Benign formatting is unchanged.
`\n` means a newline.

| Defense variant | Before attack | After LR `1e-4` | After LR `2e-4` |
|---|---:|---:|---:|
| SEAM — our original implementation | 48.42% | 51.40% | 50.13% |
| SEAM — ours, refusal formatting corrected | 51.87% | 29.48% | 24.90% |
| SEAM — ours, harmful + refusal spaces | 50.55% | 46.42% | 45.88% |
| SEAM — authors’ implementation (reference) | 52.42% | 35.15% | 24.79% |

Capability is the same four-task average (MMLU, TruthfulQA MC1, normalized
ARC-Easy, and normalized HellaSwag), excluding GSM8K. Attacks use the authors’
code for 250 steps. The refusal-corrected variant changes only the refusal
separator from newline to space relative to our original implementation. The
harmful + refusal variant additionally changes the harmful separator to a space;
completion masking, EOS supervision, Alpaca filtering, and optimization settings
remain unchanged. Results use seed 42.

| Harmful | Refusal | Attack | Self-destruct | Before attack | After LR `1e-4` | After LR `2e-4` | W&B |
|---|---|---|---|---:|---:|---:|---|
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | ✗ | 48.42% | 51.40% | 50.13% | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | 48.42% | — | — | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✓ | 51.87% | 29.48% | 24.90% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer: {answer}` | ✗ / ✓ | 51.87% | 51.23% | 24.82% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✗ | 50.55% | 46.42% | 45.88% | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer: {answer}` | — | 50.55% | — | — | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |

Harmful and refusal describe defense training; attack describes attacker training.
Benign formatting stays `Answer: {helpful answer}`. ✓ = substantial capability
collapse; ✗ = no comparable collapse; — = not tried. A single symbol applies to
both tested LRs; `✗ / ✓` means no collapse at `1e-4`, but collapse at `2e-4`. W&B links are defense-training runs. These outcomes refer to our
implementation with seed 42.

Capability uses the same four-task average above. Before-attack scores and W&B
links repeat when attack variants share a defended checkpoint.

## CIFAR-10 gradient alignment regularization (2026-11-08)

The positive-pair penalty produced a modest relearning delay across three
pair-sampling seeds in this fixed-initialization experiment. Signed mean-cosine
penalties did not consistently slow relearning.

GroupNorm ResNet-18 starts from a saved random initialization. Defense uses only
an alignment penalty for 100 Adam updates on 512 fixed training images, without
a retain or cross-entropy training term. The attack uses ordinary full-batch
cross-entropy Adam updates (LR `1e-3`, 100 updates) on the same 512 images.
Test accuracy uses 1,024 fixed held-out images. Attack example order is fixed by
an independent DataLoader generator at seed 0. Only defense sampling seeds vary.

The successful follow-up objective is `mean(relu(cos(g_i, g_j)))`, uniformly
sampling distinct example pairs. Opposing pairs contribute zero. Each sampled
positive contribution uses exact second-order autograd. Defense LR is `1e-4`;
K counts sampled pairs per defense update. Means and sample SDs below cover all
three sampling seeds for each setting.

| Method | Initial individual mean cosine | Updates to 20% test accuracy | Updates to CE below 2.0 | Mean test accuracy during 100 attack updates |
|---|---:|---:|---:|---:|
| Random initialization | 0.02483 | 38 | 46 | 22.33% |
| Positive-pair penalty, K=32 | 0.00187 ± 0.00056 | 51.7 ± 7.4 | 60.3 ± 3.5 | 19.92 ± 1.14% |
| Positive-pair penalty, K=128 | 0.00036 ± 0.00024 | 49.7 ± 0.6 | 59.3 ± 1.2 | 20.07 ± 0.53% |

All six defended runs start with lower mean cosine, take longer to reach both
thresholds, and have lower mean attack test accuracy than the baseline.
Cosine alignment recovers during relearning. These results establish a delay
for this initialization, data subset, optimizer, and attack LR; they do not
establish robustness across model initializations or other attacks.

The signed objective can hide class structure: baseline within-class cosine
was `0.898`, versus between-class `-0.0715`. Frozen individual normalization
lowered overall initial cosine to `-0.00191` but raised within-class cosine to
`0.994` and reached 20% accuracy in only 17 updates. Thus reducing the signed
mean alone was insufficient. The positive-pair objective prevents negative
pairs from canceling positive contributions in its training objective.

Authoritative artifacts on Mila:

- `/network/scratch/m/marawan.gamal/lilpipe/cifar10-gradient-regularizers-20261108`:
  signed objectives, TAR references, and initial positive-pair results.
- `/network/scratch/m/marawan.gamal/lilpipe/cifar10-positive-pair-cosine-20261108`:
  frozen training source snapshots, submitted commands, logs, and checkpoints.
- `/network/scratch/m/marawan.gamal/lilpipe/cifar10-positive-pair-repeats-20261108`:
  all repeat commands and job IDs; `repeat-statistics.json`, `comparison-audit.json`,
  `trajectories.csv`, and `relearning-summary.pdf` / `.png`. Individual-run PDFs
  show accuracy, loss, individual cosine, and group cosine separately.

The final audit verifies identical initialization checksums and data indices
for all seven runs, plus 101 attack rows and 11 cosine diagnostics per run.
Diagnostics stream exact individual gradient contractions over all 512 examples,
exclude the diagonal, and run at attack steps 0, 10, …, 100. The sampled defense
objective and the full-cohort diagnostic are distinct. All 32 CIFAR experiment
regression tests passed, including exact FP64 validation of the positive-pair
objective and validation of streaming cosine diagnostics.

## Zephyr-7B forget-gradient tracking (2026-11-08)

Completed passive tracking runs from the original Zephyr-7B-beta model using
the selected 15pp GD and 5pp NPO settings. Both trained for 125 updates with
batch size 8 (four forget rows), sequence length 512, BF16, and the selected
four-GPU model sharding. Only trainer class, output directory, and W&B name
changed in the selected configs. No cosine regularization was applied.

| Defense | WMDP-bio robust accuracy | MMLU-no-bio accuracy | First 10 batch cosine mean | Last 10 batch cosine mean | W&B |
|---|---:|---:|---:|---:|---|
| GD, 15pp selected | 31.68% | 50.40% | 0.0580 | 0.7906 | [h44oyeoi](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/h44oyeoi) |
| NPO, 5pp selected | 29.38% | 55.22% | 0.1916 | 0.5969 | [ueh1tiq2](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/ueh1tiq2) |

Each run has 125 finite `train/ghost_cosine_mean` measurements, verified online
in W&B, at optimizer steps 0–124. Measurements use the current forget batch,
exclude diagonal pairs and zero-norm gradients, and exclude retain/reference
forwards. The table averages the first and last ten logged batches; these are
not fixed-cohort or corpus-wide measurements. Both traces become more aligned.

Full benchmark evaluations ran on the saved final models with no example limit.
The selected capability budgets differ, so this is not a matched-budget
comparison of GD and NPO. Final models and `checkpoint-125` are retained under
`artifacts/tamia/models/<model-id>`, using the `--ghost-track` suffix.

Training/evaluation jobs were GD `678006`/`678008` and NPO `678007`/`678009`;
all completed with exit code `0:0`. A GPU preflight verified cross-device Gram
contractions against explicit gradients, two updates for each full model,
checkpoint saving, and the offline evaluation environment. Local tracking tests
also verify forget-only capture, unchanged training gradients, and no duplicate
counting from GD diagnostics.

Implementation commit: `dcf55deb`. Frozen source, configs, submission commands,
and final results are recorded on Tamia under
`artifacts/tamia/runs/z7b-gd15pp-npo5pp-ghost-track-20261108/`.
The local report at
`artifacts/reports/z7b-gd15pp-npo5pp-ghost-track-20261108/` contains
`final-results.json`, both Trainer states, `cosine-trajectory.csv`, and
`defense-cosine-trajectory.pdf` / `.png`.

The selected **NPO+SAM 5pp** control also completed with passive tracking
(commit `3ac4fa74`, training `678499`, evaluation `678500`). Its selected
settings include beta 0.025, retain coefficient 4, and SAM radius 0.01; these
are different from plain NPO above. Tracking captures the unperturbed forget
backward only. Tests check the Gram against explicit parameter gradients and
verify unchanged SAM updates.

| Defense | WMDP-bio robust | MMLU-no-bio | First 10 batch cosine mean | Last 10 batch cosine mean | W&B |
|---|---:|---:|---:|---:|---|
| NPO+SAM, 5pp selected | 28.00% | 54.47% | 0.0477 | 0.8357 | [sqf3flp5](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/sqf3flp5) |

Its final logged batch (step 124) has mean cosine `0.793833` and mean squared
cosine `0.642079`. The last ten batches have mean squared cosine `0.726033`.
The squared metric averages individual off-diagonal squared cosines; it does
not square the signed mean. As above, these are changing forget batches from
training. Full final-model benchmark evaluations used no example limit.
The model is saved as
`z7b-wmdp-bio--5pp--unlearn-fft-npo-sam-rho1e-2--ghost-track`.
Frozen source and launch receipts are under
`artifacts/tamia/runs/z7b-npo-sam5pp-ghost-track-20261108/`; the local report
contains `final-results.json` and `trainer-state.json` under
`artifacts/reports/z7b-npo-sam5pp-ghost-track-20261108/`.

## Zephyr squared-cosine defense calibration (2026-11-08)

GD and NPO now optionally add a penalty on the squared cosine between two
individual forget-example CE gradients, averaged over sampled distinct pairs.
Their original unlearning/retain objectives remain in the update. The estimator
uses ordinary gradients and forward finite differences along detached,
normalized directions, rescaled by their norms and the squared-cosine factor
`2*c`. Parameters, buffers, and matching Torch dropout states are restored.
FP32 gradients remain on each parameter's GPU; scalar reductions cross devices.
Optimizer gradients and the pair accumulator use CPU storage during probes to
bound memory. Coefficient zero preserves the ordinary trainer update.

The tiny-model tests compare the estimate to an independent exact FP64 derivative,
including negative cosine, dropout replay, restoration on exceptions, and GD/NPO
updates with gradient accumulation. A real-model audit additionally compared the
**full estimated gradient vector** to an exact higher-order FP32 reference for one
Zephyr pair, before and after an ordinary NPO update. The exact reference is used
only for validation; defense training continues to use finite differences.

| Parameter-space perturbation length | Relative gradient error, initial | Relative gradient error, after NPO step |
|---|---:|---:|
| 1e-3 | 18.55% | 18.55% |
| 3e-4 | 7.58% | 5.84% |
| 1e-4 | 2.97% | 1.95% |
| **3e-5** | **0.88%** | **0.77%** |
| 1e-5 | 1.60% | 2.03% |

At `rho=3e-5`, estimated/reference gradient alignment exceeds `0.99996` in both
states. Smaller perturbations begin losing accuracy to FP32 cancellation. Earlier
`rho=0.01` probes gave grossly inconsistent directional slopes, so those six
pending regularized jobs and their dependent evaluations were cancelled before
training. The two FP32 coefficient-zero controls continue unchanged.

Full GD and NPO sweeps use coefficients `0.1`, `1`, and `10`, two sampled
individual-example pairs, and 125 updates. Training uses FP32 without AMP and
four H100s with one process and explicit model sharding. A GPU gradient
accumulator reduced measured steps from roughly 90 seconds to 33–40 seconds,
allowing the final campaign to use `gpubase_bynode_b1` with three-hour limits.
Full WMDP-bio robust and MMLU-no-bio evaluations follow successful training.

Implementation: `dec76bef`; GPU accumulator: `7f77c5ab`; final LR selection:
`69fc3050`. Full-model preflights: `678516` and `678610`; exact Zephyr reference
check: `678574`. Run root on Tamia:
`artifacts/tamia/runs/z7b-squared-cosine-20261108/`. Authoritative completed
receipts: `jobs-selected-fp32.json`; immutable source/config snapshot:
`submitted/fp32-lr1e-6-rho3e-5/`. Earlier submission receipts are retained as
provenance and are superseded. Local report copies:
`artifacts/reports/z7b-squared-cosine-20261108/`.

### FP32 capability calibration

Full evaluations of the coefficient-zero FP32 controls showed that the selected
BF16 learning rate `5e-6` does not transfer safely to FP32 in this setup:

| FP32 control, LR 5e-6 | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---|
| GD, retain coefficient 4 | 24.77% | 36.35% | 0.8959 | 0.8080 | [lsj6y1ux](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/lsj6y1ux) |
| NPO, retain coefficient 1 | 27.30% | 23.01% | 0.9334 | 0.8718 | [rx6uvehe](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/rx6uvehe) |

Both controls completed all 125 updates and full benchmark evaluations
(GD `678531` / `678539`, NPO `678535` / `678543`). They fail the intended capability
budget, despite strong unlearning. The six regularized runs at this learning rate
were stopped in their first few updates; no final defense outcome is claimed.
Their receipts remain in `jobs-rho3e-5.json` for provenance.

The next controls sweep FP32 learning rates `2e-7`, `5e-7`, `1e-6`, and `2e-6`,
with regularization disabled and all other selected defense settings preserved.
Each still trains for 125 updates and logs changing-batch cosine and squared
cosine. The adjacent `calibration.yml` files generate these controls; commit
`72ab9199` records their configs. Training jobs `678622`–`678629` and dependent
full evaluations `678630`–`678637` are listed in `jobs-lr-calibration.json`.
The active source/config snapshot is `submitted/lr-calibration/`. Selection must
consider both unlearning and capabilities before rerunning the squared-cosine
penalty; the original capability/unlearning goal remains unmet.

A separate performance update (`7f77c5ab`) keeps the regularizer accumulator on
parameter devices while temporarily storing the ordinary optimizer gradient on
CPU. This avoids per-pair CPU transfers and preserves the update, including
unused parameters and exception recovery. Exact-update and tracking regression
tests pass. Three-step Zephyr validation job `678610` completed successfully
for both GD and NPO under `accumulator-preflight/`, including model saving.
Step time dropped to about 35–40 seconds, compared with about 90 seconds with
the CPU accumulator. Five-second GPU telemetry measured a maximum of 74,507 MiB
(72.8 GiB) across the four H100s. Future regularized runs therefore fit the
three-hour `gpubase_bynode_b1` partition at the unchanged 125-update budget.

The FP32 learning-rate calibration is complete. **LR 1e-6** restores a useful
baseline for both methods:

| FP32 control | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---|
| GD, LR 1e-6 | 32.14% | 50.87% | 0.7591 | 0.6329 | [8jw7anet](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/8jw7anet) |
| NPO, LR 1e-6 | 27.19% | 54.74% | 0.6305 | 0.4576 | [v376bmf2](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/v376bmf2) |

The selected controls are training/evaluation jobs `678624`/`678632` and
`678628`/`678636`. All eight calibration controls and evaluations completed
successfully with final models, checkpoint-125, and 125 geometry measurements.
The per-family `results.md` files record all learning rates, including the failed
5e-6 controls, in increasing order. Small-LR controls with lower cosine had
insufficient unlearning; selection therefore uses both benchmark metrics.

### Completed squared-cosine comparison

All six regularized defenses completed 125 updates, saved final weights and
checkpoint-125, and completed full benchmark evaluation without example limits.
All W&B runs are finished and contain 125 geometry measurements on changing
forget batches. NPO coefficient 0.1 required one retry after a filesystem failure
before the first optimizer update (678690 → 678714; evaluation 678715).

| Defense | Coefficient | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---:|---|
| GD | 0 | 32.14% | 50.87% | 0.7591 | 0.6329 | [8jw7anet](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/8jw7anet) |
| GD | 0.1 | 35.25% | 52.66% | 0.8131 | 0.6898 | [5obqzryb](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/5obqzryb) |
| GD | 1 | 45.85% | 56.42% | 0.4122 | 0.2243 | [fd9hc8aj](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/fd9hc8aj) |
| GD | 10 | 54.84% | 57.72% | 0.1838 | 0.0381 | [mnfcrd88](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/mnfcrd88) |
| NPO | 0 | 27.19% | 54.74% | 0.6305 | 0.4576 | [v376bmf2](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/v376bmf2) |
| NPO | 0.1 | 26.04% | 52.50% | 0.4669 | 0.2996 | [00as44ut](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/00as44ut) |
| NPO | **1** | **29.26%** | **53.06%** | **0.2623** | **0.0988** | [1fdg1s4r](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/1fdg1s4r) |
| NPO | 10 | 51.61% | 56.85% | 0.2046 | 0.0809 | [qh7vvmas](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/qh7vvmas) |

Lower WMDP accuracy means stronger unlearning. All cosine columns exclude the
diagonal and zero-norm examples. Mean squared cosine averages individual squares.
These training statistics precede optimizer updates. GD's strong penalties weaken
unlearning; **NPO coefficient 1** retains the desired approximately 30% WMDP and
53% MMLU while substantially reducing alignment.

Saved-model checks 678821 (GD) and 678827 (NPO) both exited successfully. Each
model/precision uses 32 fresh forget examples in eight batches of four, with
recorded independent seeds and indices. For NPO coefficient 1, final checkpoint
mean cosine is **0.3679 ± 0.0303 batch SEM in FP32**, versus 0.5534 ± 0.0712 for
its control; after BF16 reload it is **0.3806 ± 0.0442**, versus 0.8446 ± 0.0429.
Mean squared cosine is 0.1627 (FP32) / 0.1733 (BF16). Samples differ across models
and precisions; these are independent estimates, not paired precision tests.
Explicit parameter-gradient Gram checks pass on the first batch, and weights
remain unchanged. Full details are in the per-family `results.md` files.

The selected final model is
`artifacts/tamia/models/z7b-wmdp-bio--5pp--unlearn-fft-npo--gip-squared-fp32--lr1e-6--rho3e-5--hpo--lambda-1--001`.
No relearning attack has yet been run against these new defenses; lower gradient
alignment alone does not establish attack resistance.

Comparison figure: `artifacts/reports/z7b-squared-cosine-20261108/squared-cosine-defense-comparison.pdf`.
It shows full benchmark scores and the last-ten-training-batch mean squared
cosine. Raw results and trajectories: `summary.json`, `summary.csv`, and
`cosine-trajectory.csv`; final geometry: `checkpoint-geometry-{gd,npo}-{float32,bfloat16}.json`.

## NPO / SAM / GIP matched relearning trajectories (2026-11-08)

All three attacks use `GIPRelearnTrainer` for passive ghost tracking and ordinary
cross-entropy on the WMDP-bio forget corpus. There is no retain stream or defense
regularization during the attack. They use LR 1e-5 with linear decay, AdamW with
zero weight decay, BF16, sequence length 512, batch size 2, accumulation 1, training
seed 1, data seed 42, and 16 optimizer updates. Batch size 2 is the agreed change
from the saved batch-size-1 attack, allowing an off-diagonal cosine per update.
This processes 32 training examples rather than 16. All three use identical
prepared rows, tokenizers, and data order. Model-only checkpoints 4/8/12/16 remain
saved; full WMDP-bio robust and MMLU-no-bio evaluations cover 0/4/8/12/16 with no
example limits. Step 0 evaluates the actual starting defense.

| Starting defense | WMDP before | WMDP after 16 | MMLU before | MMLU after 16 | W&B |
|---|---:|---:|---:|---:|---|
| NPO | 29.38% | 26.73% | 55.22% | 22.92% | [ydc6nnmq](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/ydc6nnmq) |
| NPO + SAM | 28.00% | 45.51% | 54.47% | 55.13% | [vpsu5rxq](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/vpsu5rxq) |
| NPO + GIP | 29.26% | 54.61% | 53.06% | 57.84% | [djcw3fjs](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/djcw3fjs) |

Under this attack, GIP recovers more forget-domain accuracy than SAM by steps
8–16, so this run does not demonstrate improved relearning resistance from GIP.
Plain NPO loses MMLU capability almost immediately (22.75% at step 4), making its
low WMDP accuracy unsuitable as evidence of a useful robust defense. These are
single-seed results at one attack learning rate. The defenses themselves retain
their selected training settings; GIP was trained in FP32 at LR 1e-6, while NPO
and SAM used BF16 at LR 5e-6.

Figure generated with the existing basic trajectory plotter, whose old selection
block is commented out and whose active selection is `configs_npo_sam_gip`:
[WMDP / MMLU trajectory PDF](../artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--npo-sam-gip-ghost-ms16.pdf).
A PNG is saved beside it. The black reference line is the original Zephyr base
model evaluation already present locally, not an additional attacked model.

Manifest: `configs/experiments/z7b-wmdp-bio-npo-sam-gip-relearn-ghost-ms16.yml`.
Source/config commit: `ad69a856`. Preparation job 678860 completed successfully;
training jobs 678861/678862/678863 and evaluation jobs 678864/678865/678866 all
exited 0. Each attack saved four checkpoints of 15,008,318,632 weight bytes and
recorded 16 valid cosine measurements. W&B confirms all three runs are finished
with 16 geometry records at pre-update global steps 0–15.

Immutable submitted source/configs, resource dependencies, hashes, logs, and
receipts on Tamia:
`artifacts/tamia/runs/z7b-npo-sam-gip-relearn-ghost-ms16-20261108/`.
Local report: `artifacts/reports/z7b-npo-sam-gip-relearn-ghost-ms16-20261108/`.
`summary.json` and `trajectory.csv` record all 15 full benchmark evaluations and
their exact source files; `cosine-trajectory.csv` records all 48 training geometry
measurements; `prepared-cache-verification.json` records the identical-tokenizer
check and the three corresponding prepared cache hashes. Eight focused config,
evaluation-interface, and plotting regression tests passed.

### Relearning learning-rate comparison

The same three starting checkpoints were also attacked at initial LR 1e-6.
All training parameters except LR, output directory, and run name match the
1e-5 configs above. Training uses the same 16 steps, batch size 2, seed/data order,
BF16, and linear decay. All 15 full evaluations completed successfully:

| Defense | Initial attack LR | WMDP at step 16 | MMLU-no-bio at step 16 | W&B |
|---|---:|---:|---:|---|
| NPO | 1e-6 | 29.84% | 55.04% | [fpiej6cy](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/fpiej6cy) |
| NPO + SAM | 1e-6 | 31.57% | 54.87% | [8ml2nrzs](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/8ml2nrzs) |
| NPO + GIP | 1e-6 | 35.37% | 55.04% | [7s8sy04v](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/7s8sy04v) |

All three lower-LR attacks preserve MMLU near 55% over this short budget, while
WMDP recovery remains modest. These are results for one seed and 16 updates,
not evidence that a stronger or longer attack would also fail.

Training jobs 678912/678913/678914 and evaluations 678915/678916/678917 all exited
0. Each run saved checkpoints 4/8/12/16 and recorded 16 cosine measurements;
W&B confirms all three runs are finished with 16 valid geometry records.
Immutable configs/source commit: `92cf1e9f`. Run root on Tamia:
`artifacts/tamia/runs/z7b-npo-sam-gip-relearn-ghost-lr1e-6-ms16-20261108/`;
local report uses the same suffix under `artifacts/reports/`.
The initial three-hour allocation limits were reduced to 30 minutes before
completion; `allocation-updates.json` preserves the before/after Slurm state.

The intermediate LR 5e-6 comparison is complete with otherwise identical
settings. Source/config commit: `ecadb147`. Training jobs
678959/678960/678961 and evaluations 678962/678963/678964 all exited 0.
All three saved checkpoints 4/8/12/16 and logged 16 geometry measurements;
W&B confirms all three are finished and all 16 records are online.
Submitted source, exact 30-minute resource requests, hashes, and per-stage
submission receipts are under
`artifacts/tamia/runs/z7b-npo-sam-gip-relearn-ghost-lr5e-6-ms16-20261108/`.
The local report uses the same suffix under `artifacts/reports/`.

| Defense | Initial attack LR | WMDP at step 16 | MMLU-no-bio at step 16 | W&B |
|---|---:|---:|---:|---|
| NPO | 5e-6 | 27.42% | 51.59% | [j76ivh87](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/j76ivh87) |
| NPO + SAM | 5e-6 | 45.39% | 55.79% | [yb2drrnh](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/yb2drrnh) |
| NPO + GIP | 5e-6 | 52.42% | 57.81% | [i58zeg4w](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/i58zeg4w) |

At 5e-6, GIP again recovers more WMDP accuracy than SAM. Plain NPO retains low
WMDP accuracy while losing about 3.6 MMLU percentage points from its starting
checkpoint, rather than the much larger collapse at 1e-5. The 1e-6 attack
produces modest recovery for all three over 16 updates. These results do not
establish improved relearning resistance for GIP at the tested rates/budget.

The existing basic plotter now selects `configs_npo_sam_gip_lrs`. Color denotes
defense; line thickness denotes initial attack LR: 1e-6 thin, 5e-6 medium, and
1e-5 thick. The previous selection blocks remain commented out.
[Combined three-LR trajectory PDF](../artifacts/figures/z7b-wmdp-mmlu-relearning-trajectory--npo-sam-gip-ghost-lrs-ms16.pdf)
(and PNG beside it) includes all nine trajectories at steps 0/4/8/12/16 for both
WMDP and MMLU-no-bio: 90 plotted observations from 45 full benchmark evaluations.
`combined-plotted-trajectories.csv` and `figure-verification.json` in the 5e-6
report record the exact plotted rows and line widths. All raw evaluation JSONs
and figures are committed. Six focused config/plot regression cases passed,
including exact equality of all training settings except LR/output/run name and
monotonically increasing line thickness with LR.
