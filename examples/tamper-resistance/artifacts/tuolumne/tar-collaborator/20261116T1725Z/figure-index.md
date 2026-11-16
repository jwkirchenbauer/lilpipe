# Figure index

Every defense panel for the collaborator is the supplied final checkpoint
(reported 125 defense updates). No collaborator defense progression is shown.
Held-out probes exclude recorded attacker batches; defense/pretraining exposure
is not established.

[Findings](findings.md) · [Paired comparison report](checkpoint-report.md)
· [All benchmark scores](evaluation-scores.csv)

## Before attack

- [Large-radius comparison](static/comparison-large-radii.png) and [PDF](static/comparison-large-radii.pdf).
- [Small-radius comparison](static/comparison-small-radii.png) and [PDF](static/comparison-small-radii.pdf).
- [Original static plot recipe, with per-panel scales](static/landscape-3d.png).

## 128-step linear schedule

Paired panels use common physical axes, camera, and CE/color limits.
The reference directions come from the original local K16 attack.

- [Broad shared-random planes](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-random.png) · [center-relative](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-delta.png).
- [Local shared-reference planes](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-reference.png) · [center-relative](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-delta.png).
- [Actual loss, capability, and projection quality](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-trajectories.png).
- [Fixed random, shared-reference, and condition-fitted planes](cases/collaborator-lr1e-6-ms128/surfaces-and-checkpoints.png) · [contours](cases/collaborator-lr1e-6-ms128/attack-paths.png).
- [Moving broad random slices](cases/collaborator-lr1e-6-ms128/random-landscape-evolution.png) · [animation](cases/collaborator-lr1e-6-ms128/random-landscape-evolution.gif).
- [Moving local slices](cases/collaborator-lr1e-6-ms128/attack-landscape-evolution.png) · [animation](cases/collaborator-lr1e-6-ms128/local-landscape-evolution.gif).
- [Actual loss and capability](cases/collaborator-lr1e-6-ms128/loss-and-capability.png).
- [First-update plane](cases/collaborator-lr1e-6-ms128/first-update-landscape.png) · [first eight update segments](cases/collaborator-lr1e-6-ms128/early-update-profiles.png).
- [Observed minibatch dynamics](cases/collaborator-lr1e-6-ms128/minibatch-dynamics.png) · [fixed batch slices](cases/collaborator-lr1e-6-ms128/minibatch-landscapes.png) · [moving batch slices](cases/collaborator-lr1e-6-ms128/local-minibatch-landscapes.png).
- [Projection checks](cases/collaborator-lr1e-6-ms128/projection-fidelity.png) · [projection data](cases/collaborator-lr1e-6-ms128/projection-quality.csv) · [actual checkpoint losses](cases/collaborator-lr1e-6-ms128/surfaces/actual-losses.csv).
- [Settings](cases/collaborator-lr1e-6-ms128/study.json) · [native source and data checks](input-validation.json).

## 1,000-step linear schedule

Paired panels use common physical axes, camera, and CE/color limits.
The reference directions come from the original local K16 attack.

- [Broad shared-random planes](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-random.png) · [center-relative](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-delta.png).
- [Local shared-reference planes](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-reference.png) · [center-relative](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-delta.png).
- [Actual loss, capability, and projection quality](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-trajectories.png).
- [Fixed random, shared-reference, and condition-fitted planes](cases/collaborator-lr1e-6-ms1000/surfaces-and-checkpoints.png) · [contours](cases/collaborator-lr1e-6-ms1000/attack-paths.png).
- [Moving broad random slices](cases/collaborator-lr1e-6-ms1000/random-landscape-evolution.png) · [animation](cases/collaborator-lr1e-6-ms1000/random-landscape-evolution.gif).
- [Moving local slices](cases/collaborator-lr1e-6-ms1000/attack-landscape-evolution.png) · [animation](cases/collaborator-lr1e-6-ms1000/local-landscape-evolution.gif).
- [Actual loss and capability](cases/collaborator-lr1e-6-ms1000/loss-and-capability.png).
- [First-update plane](cases/collaborator-lr1e-6-ms1000/first-update-landscape.png) · [first eight update segments](cases/collaborator-lr1e-6-ms1000/early-update-profiles.png).
- [Observed minibatch dynamics](cases/collaborator-lr1e-6-ms1000/minibatch-dynamics.png) · [fixed batch slices](cases/collaborator-lr1e-6-ms1000/minibatch-landscapes.png) · [moving batch slices](cases/collaborator-lr1e-6-ms1000/local-minibatch-landscapes.png).
- [Projection checks](cases/collaborator-lr1e-6-ms1000/projection-fidelity.png) · [projection data](cases/collaborator-lr1e-6-ms1000/projection-quality.csv) · [actual checkpoint losses](cases/collaborator-lr1e-6-ms1000/surfaces/actual-losses.csv).
- [Settings](cases/collaborator-lr1e-6-ms1000/study.json) · [native source and data checks](input-validation.json).

The post-hoc fitted plane differs between attacks. Its projections can have
material loss error, especially early in training. Actual checkpoint traces
and first-update slices provide the corresponding direct measurements.


