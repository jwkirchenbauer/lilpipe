# RIO: simulation depth and attacker matching

This follows the [hyperparameter study](TAR_HPARAM_STUDY.md),
[cleanroom study](TAR_CLEANROOM_STUDY.md), and
[inner-versus-actual attack analysis](TAR_INNER_BASIS_STUDY.md).
The prior work was committed and pushed through `d0681786` before this phase.

Start every new defense from our original pinned Zephyr checkpoint. Use RIO:
retain coefficient 4, outer learning rate 1e-5, 125 outer updates, and the
existing data, precision, outer schedule, seeds, and maximum-entropy objective.
RIO produces raised shoulders in the measured random slices, but its existing
defender misses the original MMLU utility target. Assess utility for every new
defender; raised shoulders alone do not establish a closed loss barrier.

## 1. Inner simulation depth

Compare K = 4, 16, 64, 256 at the same 125 outer updates. Reuse the completed
RIO K16 control and its standard attack. New simulations keep batch size one,
constant inner LR 1e-5, the original AdamW defaults, and fresh optimizer state
at each outer call. Preserve the original outer batch and its 16 simulation
examples: K4 uses the first four; K64 and K256 cycle those same 16 examples.
This changes simulation depth and repeated exposure, without changing the
outer forget/retain stream. It does not simulate 256 new attack documents.

The respective inner-update budgets are 500, 2,000, 8,000, and 32,000. Record
supervised tokens and measured allocation time as well as updates. This is a
fixed-outer-update comparison; it does not hold total defense compute fixed.
The inner and actual attacker learning rates differ, so K is not an exact
prediction of the number of actual attack updates resisted.

## 2. Simulation data and optimizer fidelity

At K16, cross two simulation data policies with two optimizer policies:

| Simulation data | Legacy RIO optimizer | Actual-prefix optimizer settings |
|---|---|---|
| Fresh original per-outer simulation examples | Existing RIO control | New optimizer-only contrast |
| Exact first 16 standard-attack minibatches, repeated at every outer call | New data-only contrast | New joint contrast |

Also reverse the same fixed 16 minibatches under both optimizer policies.
These are order controls with identical examples and token budgets. They do
not introduce a moving window through the 1000-step attack. The original
outer forget and retain examples, loss, and 125-update schedule stay fixed.
The original simulation supervises padded EOS positions; the actual attack's
label masking differs. Replaying exact minibatches matches the full tokens,
masks, and labels as well as the selected examples. Therefore the fresh-versus-
prefix contrast includes input preparation and repeated use of a small pool;
it does not isolate source-document identity alone.

Legacy AdamW uses constant LR 1e-5, weight decay 0.01, and no inner clipping.
The matched setting uses the actual attack prefix's recorded LR values from
its LR1e-6, 1000-update linear schedule, weight decay zero, clipping at one,
and explicit optimizer settings. All simulations start with fresh moments.
Matching a fresh K-step attack does not join the 125 simulations into a single
continuous attack. Exact batch hashes and optimizer settings are necessary
but insufficient for numerical trajectory parity: verify the real 7B update
path before describing any condition as numerically matched.

## Measurements

Run the same independently initialized LR1e-6, 1000-update standard attack
against every defender, preserving its exact cached data and order. Save dense
early checkpoints and the established later milestones. Measure WMDP recovery
throughout, including the highest measured recovery and the first measured
crossings of declared thresholds; report initial scores and MMLU alongside
them. Distinguish resistance within K from escape after K, and CE descent from
benchmark recovery. A fixed-prefix improvement that disappears after step 16
is limited resistance to that trained subsequence.

Use the existing fixed-ten and attack-held-out-32 probes, literal shared random
and reference directions, common physical radii and CE scales, each attack's
fitted plane, first-update slices, actual-minibatch losses, and projection
residual checks. Preserve defense history and attack dynamics. Report the
actual checkpoint losses separately from projected reconstructions. Compare
absolute and center-relative views so a changing central loss cannot masquerade
as a changing shoulder. The initial study uses the existing paired seed;
replication is needed before claiming a systematic effect beyond these runs.
