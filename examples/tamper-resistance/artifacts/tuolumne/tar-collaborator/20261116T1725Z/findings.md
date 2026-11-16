# Collaborator checkpoint: original attacks and landscapes

The supplied checkpoint produces a much more pronounced random-plane fortress
than our locally reproduced K16 defender under matched plotting settings.
It also retains more resistance to the tested LR `1e-6` attacker. That is a
checkpoint contrast, not an isolated test of hardware or one defense setting:
the [published configuration](https://huggingface.co/mremila/z7b-wmdp-bio-5pp-unlearn-fft-me-tar-k16/blob/aa62e643db200fec5997d0b7f66c668bf0ecdac8/training_config.yml)
differs in defense and inner learning rates, retention strength, sampling,
and batching.

The Hub revision is `aa62e643db200fec5997d0b7f66c668bf0ecdac8`.
Its locally measured initial scores are WMDP **41.94%** and MMLU **53.74%**,
close to the upload's **42.28% / 53.61%**. This is a different published result
from the earlier selected-config reference of 31.34% / 53.45%.

## Capability and forget-text loss

These are two independent linear attack schedules, both starting fresh from
each defender. Their decay differs after the first update; the 128-step run is
not a prefix of the 1,000-step run. Lower WMDP means less measured biology
capability. MMLU measures retained general capability with biology tasks excluded.

| Defender | Attack schedule | WMDP before / final (%) | MMLU before / final (%) | Fixed-probe CE before / final (nats) |
| --- | ---: | ---: | ---: | ---: |
| Local K16 reproduction | 128 | 33.06 / 51.61 | 54.02 / 56.93 | 11.238 / 4.304 |
| Supplied collaborator model | 128 | 41.94 / 42.86 | 53.74 / 53.39 | 15.516 / 10.246 |
| Local K16 reproduction | 1,000 | 33.06 / 53.57 | 54.02 / 58.06 | 11.238 / 1.949 |
| Supplied collaborator model | 1,000 | 41.94 / 45.97 | 53.74 / 54.00 | 15.516 / 4.021 |

On the 1,000-step schedule, the collaborator model's highest measured WMDP is
**46.08%**, at updates 256, 384, and 768; the local control peaks at **53.69%**
at update 192. These maximize over saved checkpoints of this one learning
rate, not over a stronger attacker family. The collaborator model starts with
less unlearning, but ends with lower WMDP under this attack. Both defenders
meet the original five-percentage-point MMLU budget before attack.

Descent directions still exist. The first saved update lowers the collaborator
model's fixed-probe CE by **0.812 nats**, versus **0.088** for the local control.
The corresponding raw update norms are 0.10877 and 0.09767, so equal optimizer
settings do not mean equal parameter displacements. The new first-update
plane reproduces its separately measured endpoint CE on both the fixed probe
and the actual first batch.

The collaborator model's fixed and 32-document attack-held-out CE eventually fall to
4.021 and 4.064. These 32 documents are absent from the recorded attacker batches;
their defense/pretraining exposure is not established. This is substantial language-loss reduction, while WMDP only
rises by about four points. The presence of CE descent does not establish
complete benchmark recovery. Conversely, a taller random-plane fortress alone
does not establish robustness to other attacks.

The supplied model's 128-step endpoint MMLU is 53.39%, slightly below the
absolute five-point-drop threshold of 53.59%. Its 1,000-step endpoint is 54.00%.
The budget above describes the initial defenses; attack outcomes retain their
measured utility scores without imposing a new checkpoint selection rule.

## Views to compare

- [Original static recipe: large radii, shared scales between models](static/comparison-large-radii.png)
  and [small radii](static/comparison-small-radii.png). The static center CE is
  15.521 versus 11.238; each radius uses the same vertical/color range for both models.
- [1,000-step broad random views](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-random.png),
  [shared-reference local views](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-reference.png),
  and [center-relative local views](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-delta.png).
- [1,000-step actual and projected trajectories](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-trajectories.png).
- [First-update plane](cases/collaborator-lr1e-6-ms1000/first-update-landscape.png)
  and [the first eight update segments](cases/collaborator-lr1e-6-ms1000/early-update-profiles.png).
- [Full figure index, animations, and measurement files](figure-index.md)
  and [the comparison report](checkpoint-report.md).

The new 1,000-step fitted plane explains 99.12% of summed squared raw displacement
at 13 sampled saved checkpoints, but captures only 12.16% at the first update. Its fitted path is
a projection, not the optimizer's visited path. Use actual checkpoint losses,
the first-update view, and the recorded projection errors for early behavior.
Only the interpolation endpoints are saved optimizer states; the visible
stair steps in interpolated BF16 evaluations are not evidence of discontinuous
optimizer jumps.

The old K16 reference plane captures only 5.17% of the supplied model's
128-step endpoint displacement. Moving reference panels translate those fixed
directions through each real checkpoint; they do not contain the full attack
path. Even the 128-step attack's fitted plane, with 99.12% sampled displacement
energy, gives a first-step projected CE 1.20 nats below the actual saved state.
This is why the direct checkpoint and first-update diagnostics accompany the
projected paths.

## What was held fixed

This uses the original named attack configs and observation pipeline: full
fine tuning with AdamW, LR `1e-6`, no warmup, zero weight decay, clipping at 1,
batch size 1, sequence length 512, seed 1, data seed 42, BF16, and eager attention.
All 24,432 prepared rows have identical token/label hashes and source positions
to the earlier control. Every actual optimizer batch matches its schedule's
control in order. The supplied weight bytes match the pinned Hub LFS hash;
parameter layout, vocabulary, and probe tokenization checks pass.

Dynamic comparisons reuse the same ten fixed documents, the compute study's
32 held-out documents, and the exact physical random and baseline attack-reference
bases. The earlier 128-step control used a different held-out cohort, so its
held-out curve remains separate. Static figures retain the original BF16 random
directions and default attention recipe; dynamic views use shared FP32
orthonormal directions and BF16 eager evaluations. Comparisons are matched
within each recipe, rather than treating small cross-recipe CE differences as
a centering error.

The collaborator supplies one final checkpoint, reported as 125 defense updates.
No intermediate defense history was reconstructed. These results cover one
uploaded checkpoint, one attacker seed, and LR `1e-6`; the larger original
attack learning rates remain untested on this checkpoint. No causal claim is
made about which defense hyperparameter produced the difference.

[Verification](verification.json), [numerical profile implementation audit](profile-implementation-validation.json),
and the frozen source profiles record the checks. The profile entry point and
every locally called helper/class have identical Python syntax trees to the
earlier implementation. The full module files differ; each cache retains
separate source, protocol, and input checks. The reference-selection adapter
was extended to validate and render the cached measurements for this comparison.
