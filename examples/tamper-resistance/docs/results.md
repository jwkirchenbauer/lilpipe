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
