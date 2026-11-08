Scores are percentages. noise01, noise02, and noise05 denote 1%, 2%, and 5%
global L2 noise. `llama2-7b-noise01` is perturbed directly from Base, without SEAM.

The `lr` suffix specifies the attack learning rate, including in the before-attack
table where it identifies the subsequent run.
The two unperturbed Base rows share the same before-attack evaluation.

Before the attack:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | GSM8K strict² | GSM8K flexible² | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| `llama2-7b-lr1e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| `llama2-7b-lr2e-04-noise01` | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | 22.21 | 22.52 | **48.66** |
| `llama2-7b-lr2e-04-seam` | 4.35 | 41.51 | 29.62 | 64.02 | 74.55 | 19.26 | 20.24 | **45.79** |
| `llama2-7b-lr2e-04-seam-noise02` | 5.00 | 41.88 | 29.74 | 64.65 | 74.44 | 17.13 | 18.12 | **45.57** |
| `llama2-7b-lr2e-04-seam-noise05` | 4.83 | 41.28 | 28.89 | 64.35 | 74.00 | 17.89 | 18.57 | **45.28** |

After 250 relearning attack steps:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | GSM8K strict² | GSM8K flexible² | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 78.91 | 42.16 | 25.83 | 58.80 | 71.56 | 0.99 | 12.66 | **39.87** |
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | 16.76 | 18.88 | **45.83** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | 0.38 | 11.68 | **37.79** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | N/A | N/A | **N/A** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | 0.00 | 0.61 | **34.94** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | 2.58 | 11.68 | **40.06** |

Average is the unweighted mean of MMLU, TruthfulQA MC1, ARC-E, HellaSwag, and
GSM8K strict, calculated before rounding. Flexible extraction is shown separately
and is not counted again in the average. GSM8K is missing for the attacked SEAM
checkpoint, so its average is N/A.

¹ Normalized accuracy. ² Five-shot exact match with the indicated extraction rule.

Without GSM8K, the average uses MMLU, TruthfulQA MC1, ARC-E, and HellaSwag,
calculated before rounding. This matches the original plots.

Before the attack, without GSM8K:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | **55.27** |
| `llama2-7b-lr1e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | **55.27** |
| `llama2-7b-lr2e-04-noise01` | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | **55.27** |
| `llama2-7b-lr2e-04-seam` | 4.35 | 41.51 | 29.62 | 64.02 | 74.55 | **52.42** |
| `llama2-7b-lr2e-04-seam-noise02` | 5.00 | 41.88 | 29.74 | 64.65 | 74.44 | **52.68** |
| `llama2-7b-lr2e-04-seam-noise05` | 4.83 | 41.28 | 28.89 | 64.35 | 74.00 | **52.13** |

After 250 relearning attack steps, without GSM8K:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 78.91 | 42.16 | 25.83 | 58.80 | 71.56 | **49.59** |
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | **47.14** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |


Final:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | **47.14** |
| `llama2-7b-lr1e-04-noise01` | 78.94 | 42.81 | 26.32 | 66.58 | 73.70 | **52.35** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-lr2e-04-seam-noise01` | 0.00 | 22.95 | 24.60 | 29.97 | 39.29 | **29.20** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |

Renamed

| Model | Harmfulness ↓ | MMLU ↑ | TruthfulQA MC1 ↑ | ARC-E ↑ | HellaSwag ↑ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-seam-noise01` | 0.00 | 22.95 | 24.60 | 29.97 | 39.29 | **29.20** |
| `llama2-7b-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |


| Method | Low capability drop | Robust against HPs |
|---|:---:|:---:|
| TAR | ❌ | ❌ |
| SAM | ❌ | ❌ |
| SEAM | ✅ | ❌ |

| Vector | Norm |
|---|---:|
| $\theta_{\mathrm{base}}$ | 1575.2859 |
| $\tau_{\mathrm{seam}}$ | 8.9118 |


Llama-2-7B-chat base and Gaussian-noise variants, before any relearning attack:

The base is `meta-llama/Llama-2-7b-chat-hf`. Noise is applied directly to this base model with seed 42, with its L2 norm scaled to the stated percentage of the base model’s parameter norm.

| Model | Harmfulness ↓ | MMLU ↑ | TruthfulQA MC1 ↑ | ARC-E¹ ↑ | HellaSwag¹ ↑ | GSM8K strict² ↑ | GSM8K flexible² ↑ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B-chat | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| Llama-2-7B-chat + 0.1% noise | N/A | 46.50 | 30.23 | 68.39 | 75.94 | 22.52 | 22.90 | **48.72** |
| Llama-2-7B-chat + 0.25% noise | N/A | 46.57 | 29.99 | 68.52 | 76.02 | 22.37 | 22.67 | **48.69** |
| Llama-2-7B-chat + 0.5% noise | N/A | 46.62 | 29.99 | 68.52 | 75.97 | 22.82 | 23.05 | **48.78** |
| Llama-2-7B-chat + 0.75% noise | N/A | 46.70 | 29.62 | 68.64 | 75.93 | 22.44 | 22.74 | **48.67** |
| Llama-2-7B-chat + 1% noise | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | 22.21 | 22.52 | **48.66** |
| Llama-2-7B-chat + 5% noise | N/A | 46.06 | 28.27 | 68.35 | 75.14 | 21.91 | 22.14 | **47.95** |

Scores are percentages. The average is the unweighted mean of the four capability tasks and GSM8K strict, computed before rounding; flexible extraction is shown separately. MMLU, TruthfulQA MC1, ARC-E, and HellaSwag are zero-shot; GSM8K is five-shot. N/A denotes a missing saved harmfulness evaluation.

Capability values come from `configs/results/llama2-7b-gaussian-noise.yml`; harmfulness comes from the corresponding model’s saved evaluation directory.

Average capability summary (same five-task average as above):

| Model | Average ↑ |
|---|---:|
| Llama-2-7B-chat | **48.60** |
| Llama-2-7B-chat + 0.1% noise | **48.72** |
| Llama-2-7B-chat + 0.25% noise | **48.69** |
| Llama-2-7B-chat + 0.5% noise | **48.78** |
| Llama-2-7B-chat + 0.75% noise | **48.67** |
| Llama-2-7B-chat + 1% noise | **48.66** |
| Llama-2-7B-chat + 5% noise | **47.95** |

SEAM implementation comparison, using 250-step author-codebase attacks with 10% warmup:

All models start from `meta-llama/Llama-2-7b-chat-hf`. Capability is the unweighted mean of MMLU, TruthfulQA MC1, normalized ARC-Easy, and normalized HellaSwag; GSM8K is excluded from this average. Scores are percentages, and drops are percentage points.

| Defense | Before attack | After LR `1e-4` | After LR `2e-4` | Drop at `2e-4` |
|---|---:|---:|---:|---:|
| Base | 55.27 | 53.09 | 49.59 | 5.68 |
| SEAM (authors) | 52.42 | 35.15 | 24.79 | 27.63 |
| SEAM (ours, completion mask) | 48.42 | 51.40 | 50.13 | -1.71 |
| SEAM (ours, paired mask) | 51.98 | 31.44 | 35.94 | 16.04 |

Our corrected defense uses `trainers.tr_seam.SEAMTrainer` with the authors' paired-token loss mask and exact data formatting. Its 24,000 prepared examples match the authors' token IDs, attention masks, and labels. After either attack, GSM8K flexible extraction falls from 17.66% to 0.00%, and harmfulness falls from 4.34% to 0.00%. All saved parameters have finite norms. The capability collapse is substantial but less severe than the author-trained model's result at this LR.

The corrected checkpoint is `artifacts/tamia/models/llama2-7b-beavertail--align-fft-seam--paired-token-mask`. Its recipe is [base.yml](../configs/stages-beavertails/0_alignment/llama2-7b-beavertail--align-fft-seam--paired-token-mask/base.yml), and its recorded endpoint metrics are in [endpoint-results.json](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paired-token-mask/endpoint-results.json). Both endpoints and all intermediate-checkpoint evaluations are complete for the corrected defense. The trajectory figures include checkpoints every 50 steps.

The corrected `tr_seam.py` defense appears as **SEAM (ours, paired)** in the [250-step author-attack comparison](../artifacts/figures/llama2-7b-beavertail--align-fft-seam-author-codebase-paper-config--data-overlapping--relearn-fft-author-codebase--ms250.pdf).

Controlled SEAM ablations (complete).

All defenses use our SEAM trainer for 500 steps with the paper hyperparameters. Attacks use the authors' native trainer for 250 steps, with 10% warmup and seed 42. Capability is the equal mean of MMLU accuracy, TruthfulQA MC1 accuracy, ARC-Easy normalized accuracy, and HellaSwag normalized accuracy; GSM8K is separate. Values below are percentages.

| Defense variant | Capability before | After LR `1e-4` | After LR `2e-4` |
|---|---:|---:|---:|
| Original completion-mask control | 48.42 | 51.40 | 50.13 |
| Alpaca selection/truncation only | 49.32 | 50.39 | 47.50 |
| Paired mask only | 50.75 | 48.61 | 47.51 |
| Ignore EOS labels only | 52.57 | 48.70 | 45.73 |
| Paired mask + ignore EOS labels | 48.51 | 49.27 | 47.78 |
| Paired mask + ignore EOS + refusal separator space | 52.61 | **34.57** | 49.34 |
| Fully author-matched preparation + paired mask (reference) | 51.98 | 31.44 | 35.94 |

The Alpaca-only, mask-only, and EOS-only variants independently change one factor relative to the completion-mask control. The mask/EOS variant adds only EOS suppression to the mask-only variant. The refusal-space variant then changes only the refusal prompt separator, retaining the original Alpaca filtering, separate tokenization, labels, and training settings. The fully author-matched reference changes multiple preparation factors and establishes that collapse is possible; it does not isolate its cause.

The refusal separator is a trigger on the paired-mask/EOS-ignore background at LR `1e-4`: changing `Answer:\n` to `Answer: ` produces an 18.03-point capability drop, while the newline control gains 0.76 points. The effect is LR-dependent; `2e-4` does not show comparable collapse. Separator alone without paired masking and EOS suppression was not tested in this first batch; the follow-up below isolates it. Original Alpaca filtering remains in place, so fixing that filtering was not necessary for the observed `1e-4` collapse.

| Refusal-space setting | Capability | Harmfulness | GSM8K flexible |
|---|---:|---:|---:|
| Before attack | 52.61 | 4.39 | 21.53 |
| After LR `1e-4` | 34.57 | 0.00 | 0.00 |
| After LR `2e-4` | 49.34 | 18.69 | 15.31 |

The [prepared-data audit](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--paired-mask--ignore-eos-labels--refusal-space/prepared-data-validation.json) compared all 51,748 rows before releasing the full defense: all 8,000 harmful rows and 35,748 benign rows are identical, and labels and attention masks are identical everywhere. Exactly one ignored refusal-prompt input token changes (`13` to `29871`) in 7,999 rows. One overlength prompt truncates before the separator and has no supervised completion. All 4,000 refusal rows visited during defense have exactly the intended one-token change.

With the authors' paired-mask loss, identical newline prefixes eliminate the gradient at the first harmful response target because its preceding prediction position is masked. Changing the refusal separator restores that target's gradient in 3,687 of the 4,000 visited rows. It also changes refusal forward context; the experiment isolates the separator, not those two downstream effects from each other.

All [endpoint metrics](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--paired-mask--ignore-eos-labels--refusal-space/endpoint-results.json) are backed up with matching SHA-256 hashes. The evaluated `1e-4` checkpoint-250 weight shards are byte-for-byte identical to the final model that passed the full finite-parameter norm scan; the [training validation](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--paired-mask--ignore-eos-labels--refusal-space/author-attacks/training-validation.json) records all three shard digests.

The [ablation plot](../artifacts/figures/llama2-7b-beavertail--align-fft-seam--paper-config--data-overlapping--relearn-fft-author-codebase--ms250--ablations.pdf) includes both LRs for every method, with evaluations at steps 0, 50, 100, 150, 200, and 250. All harmfulness, four-task capability, and GSM8K checkpoint evaluations are complete and backed up with verified hashes. The latest separator trajectories are retained for follow-up analysis; older ablation attack weights were removed only after evaluation, backup verification, and plot inclusion. All starting and defended models remain.

The separator is configurable through `message_property_mappings.response_separator` in the existing [data loader](../src/data/beavertails_hh.py); the [defense sweep](../configs/stages-beavertails/0_alignment/llama2-7b-beavertail--align-fft-seam/sweep.yml) contains the controlled variants.

Minimal-diff follow-up: refusal space alone.

The smallest tested sufficient change is one line in the original `BeaverTailsHBStrategy.tokenize_row` refusal formatter:

```diff
- prompt = f"Question: {row['prompt']}\nAnswer:\n"
+ prompt = f"Question: {row['prompt']}\nAnswer: "
```

Keep original completion masking, EOS supervision, Alpaca filtering, loss calculation, and training hyperparameters. Paired masking and EOS suppression are unnecessary for the observed collapse at both tested attack LRs. In the current loader, the same change is expressed with `message_property_mappings.response_separator: " "` on the refusal stream; the sweep entry is `completion-refusal-space`. Use its separate prepared-data cache rather than reusing newline-tokenized data.

| Defense | Capability before | After LR `1e-4` | After LR `2e-4` |
|---|---:|---:|---:|
| Original completion-mask control | 48.42 | 51.40 | 50.13 |
| Original setup + refusal space only | 51.87 | **29.48** | **24.90** |

| Space-only setting | Capability | Harmfulness | GSM8K flexible |
|---|---:|---:|---:|
| Before attack | 51.87 | 5.19 | 12.81 |
| After LR `1e-4` | 29.48 | 0.00 | 0.00 |
| After LR `2e-4` | 24.90 | 0.00 | 0.00 |

Capability uses the same four-task mean as above. These results establish sufficiency for seed 42, the 500-step defense, and these 250-step author-codebase attacks; they do not establish robustness across seeds or other attacks. Zero harmfulness accompanies capability collapse and should not be read alone as successful refusal behavior.

The [data audit](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--refusal-space/prepared-data-validation.json) compared all 51,748 prepared rows. All labels and attention masks match the original control. All 4,000 visited refusal rows change by exactly one ignored prompt token. Runtime preparation retains EOS targets and applies no paired loss mask. Executing the archived original data loader with only the one-line patch reproduces all 8,000 refusal rows' input IDs, masks, and labels exactly. The full training reused this audited cache.

On representative CPU hidden states, the completion-loss helper produced bitwise-identical losses and all gradients to the archived original helper in both float32 and BF16. This check is recorded in the run's `ablation.json`; it verifies this helper, not whole-run bitwise determinism. All three full-parameter scans (defense and both final attacks) have finite norms. The full root regression suite and focused completion/paired-loss tests passed.

Both [endpoint evaluations](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--refusal-space/endpoint-results.json) and every 50-step checkpoint evaluation are complete. All 11 jobs succeeded, and all 33 required metric files are backed up locally with matching SHA-256 hashes, recorded in the [trajectory audit](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--refusal-space/trajectory-evaluation-audit.json). The evaluated checkpoint-250 weight shards match the scanned final model byte-for-byte at both LRs.

The [minimal-fix trajectory plot](../artifacts/figures/llama2-7b-beavertail--align-fft-seam--paper-config--data-overlapping--relearn-fft-author-codebase--ms250--minimal-ablations.pdf) compares the original control, space-only fix, combined mask/EOS/space fix, and author-trained reference. Method occupies rows; LR controls thickness and markers. Dashed lines show the base model after the authors' 250-step attack at LR `1e-4`. Capability collapse is already present at step 50 and persists through step 250. The previous ablation figure is preserved.

The [literal minimal patch](../artifacts/tamia/runs/llama2-7b-beavertail--align-fft-seam--paper-config--refusal-space/minimal-fix.patch) applies to the archived original formatter. No additional defense candidate is needed to demonstrate a one-factor fix against the original negative control. Starting models and both new attack trajectories are retained; no weights were deleted in this follow-up.
