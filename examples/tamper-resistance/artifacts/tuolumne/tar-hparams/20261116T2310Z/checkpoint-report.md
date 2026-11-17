# Controlled TAR hyperparameter comparison

Coverage: **complete**. Only exact measured endpoints are shown; pending values are not final measurements.

Controlled hyperparameter contrasts in the original BF16 pathway with the local metadata sampler: retain coefficient 2.5 or 4, inner LR 1e-6 or 1e-5, and outer LR 5e-6 or 1e-5. Explicit case labels and matrix.json identify each condition. The existing B control is reused. All conditions retain K16, 125 defense updates, and matched LR1e-6 attacks over 1000 updates. Joint changes can interact; this is one defense/attacker seed. The uploaded checkpoint's legacy OPTML data pipeline is not a factorial condition and is not reproduced by changing these three values alone.

Attack learning-rate schedules are matched by their declared 128 or 1000 update horizon. These horizons have different learning-rate decay and remain separate comparisons. Shared reference and random directions retain the original baseline's exact physical axes: local half-width 0.1571333988 and broad random half-width 125 global L2. The figures show translated slices through the same fixed forget CE objective. Geometry uses raw saved coordinates; CE uses each saved analysis's evaluation casting convention.

[Summary CSV](checkpoint-comparison-summary.csv) · [Validation](checkpoint-comparison-validation.json)

| Checkpoint / attack schedule | Fixed CE initial / final | WMDP initial / final | MMLU initial / final | Raw endpoint L2 | Fitted energy fraction | Minimum reference / fitted capture fraction | Maximum reference / fitted CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| reproduced-1000 (1000) | 11.238 / 1.949 | 33.06 / 53.57 | 54.02 / 58.06 | 0.939 | 0.958 | 0.366 / 0.127 | 4.074 / 0.612 |
| hparam-r-lr1e-6 (1000) | 10.957 / 1.950 | 33.64 / 54.61 | 54.91 / 57.84 | 0.985 | 0.959 | 0.040 / 0.111 | 8.170 / 0.795 |
| hparam-i-lr1e-6 (1000) | 6.933 / 2.607 | 51.84 / 50.12 | 54.62 / 54.94 | 1.155 | 0.977 | 0.000 / 0.143 | 4.299 / 0.084 |
| hparam-o-lr1e-6 (1000) | 11.257 / 2.005 | 27.07 / 51.27 | 48.88 / 55.02 | 1.247 | 0.958 | 0.012 / 0.089 | 8.814 / 0.702 |
| hparam-rio-lr1e-6 (1000) | 14.408 / 5.965 | 36.29 / 44.59 | 52.62 / 53.96 | 1.996 | 0.982 | 0.000 / 0.048 | 8.443 / 1.958 |

## Matched 1000-step schedule: reproduced-1000--hparam-r-lr1e-6

Common local frames: [0, 1, 8, 32, 128, 512, 1000]; common broad frames: [0, 128, 1000]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-r-lr1e-6-trajectories.pdf)

## Matched 1000-step schedule: reproduced-1000--hparam-i-lr1e-6

Common local frames: [0, 1, 8, 32, 128, 512, 1000]; common broad frames: [0, 128, 1000]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-i-lr1e-6-trajectories.pdf)

## Matched 1000-step schedule: reproduced-1000--hparam-o-lr1e-6

Common local frames: [0, 1, 8, 32, 128, 512, 1000]; common broad frames: [0, 128, 1000]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-o-lr1e-6-trajectories.pdf)

## Matched 1000-step schedule: reproduced-1000--hparam-rio-lr1e-6

Common local frames: [0, 1, 8, 32, 128, 512, 1000]; common broad frames: [0, 128, 1000]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--hparam-rio-lr1e-6-trajectories.pdf)

## Scope and limits

The initial attack state is each defense endpoint. Each new defender has nine saved checkpoints across its 125 defense updates. The analysis uses this observed defense history and the subsequent independent attack. The original B control reuses its completed saved trajectory. The ten fixed probe documents, corpus revision, physical basis, and parameter layout are checked. Attack-held-out means excluded from recorded attacker batches, with no claim about defense or pretraining exposure. Those 32-document CE curves are overlaid only when their content hashes agree; the earlier 128-step set can differ. Fitted energy sums squared raw displacements over selected checkpoints, not centered PCA or an equal-time average. It does not guarantee high capture at every state or accurate projected CE. N/A capture means a recorded rank-zero trajectory. Only the declared LR 1e-6 attacks are compared; this is not a maximum over attacker learning rates or optimizer seeds.

Starting utility must be assessed alongside WMDP. The original base MMLU reference is 58.587299%; a five percentage point utility-drop budget requires at least 53.587299%. The CSV records initial utility drops without forcing a defender selection. One defender checkpoint and one attack optimizer seed do not establish variation across seeds.

- reproduced-1000: [early-update-profiles](../../tar-compute/20261110T1815Z/cases/baseline-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- reproduced-1000: [first-update-landscape](../../tar-compute/20261110T1815Z/cases/baseline-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-r-lr1e-6: [early-update-profiles](cases/hparam-r-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-r-lr1e-6: [first-update-landscape](cases/hparam-r-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-i-lr1e-6: [early-update-profiles](cases/hparam-i-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-i-lr1e-6: [first-update-landscape](cases/hparam-i-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-o-lr1e-6: [early-update-profiles](cases/hparam-o-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-o-lr1e-6: [first-update-landscape](cases/hparam-o-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-rio-lr1e-6: [early-update-profiles](cases/hparam-rio-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- hparam-rio-lr1e-6: [first-update-landscape](cases/hparam-rio-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
