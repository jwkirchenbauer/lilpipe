# Supplied and reproduced defense checkpoint comparison

Coverage: **complete**. Only exact measured endpoints are shown; pending values are not final measurements.

This is a multi-factor defense checkpoint contrast. The supplied collaborator HF checkpoint mremila/z7b-wmdp-bio-5pp-unlearn-fft-me-tar-k16, revision aa62e643db200fec5997d0b7f66c668bf0ecdac8, used outer defense LR 1e-5, inner LR 1e-5, and retain coefficient 4, compared with reproduced settings 5e-6, 1e-6, and 2.5. Sampling/batching also differs (OPTML authors_documents). Differences cannot be attributed to a single hyperparameter. This compares the supplied checkpoint against the original reproduced K16 checkpoints.

Attack learning-rate schedules are matched by their declared 128 or 1000 update horizon. These horizons have different learning-rate decay and remain separate comparisons. Shared reference and random directions retain the original baseline's exact physical axes: local half-width 0.1571333988 and broad random half-width 125 global L2. The figures show translated slices through the same fixed forget CE objective. Geometry uses raw saved coordinates; CE uses each saved analysis's evaluation casting convention.

[Summary CSV](checkpoint-comparison-summary.csv) · [Validation](checkpoint-comparison-validation.json)

| Checkpoint / attack schedule | Fixed CE initial / final | WMDP initial / final | MMLU initial / final | Raw endpoint L2 | Fitted energy fraction | Minimum reference / fitted capture fraction | Maximum reference / fitted CE gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| reproduced-1000 (1000) | 11.238 / 1.949 | 33.06 / 53.57 | 54.02 / 58.06 | 0.939 | 0.958 | 0.366 / 0.127 | 4.074 / 0.612 |
| reproduced-128 (128) | 11.238 / 4.304 | 33.06 / 51.61 | 54.02 / 56.93 | 0.629 | 0.982 | 0.366 / 0.366 | 0.676 / 0.676 |
| collaborator-lr1e-6-ms128 (128) | 15.516 / 10.246 | 41.94 / 42.86 | 53.74 / 53.39 | 0.548 | 0.991 | 0.000 / 0.800 | 4.993 / 1.202 |
| collaborator-lr1e-6-ms1000 (1000) | 15.516 / 4.021 | 41.94 / 45.97 | 53.74 / 54.00 | 2.809 | 0.991 | 0.000 / 0.122 | 9.911 / 1.558 |

## Matched 128-step schedule: reproduced-128--collaborator-lr1e-6-ms128

Common local frames: [0, 1, 8, 128]; common broad frames: [0, 128]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-128--collaborator-lr1e-6-ms128-trajectories.pdf)

## Matched 1000-step schedule: reproduced-1000--collaborator-lr1e-6-ms1000

Common local frames: [0, 1, 8, 32, 128, 512, 1000]; common broad frames: [0, 128, 1000]. All panels in each figure share physical bounds, CE/color limits, and camera orientation.

- local-reference: [PNG](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-reference.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-reference.pdf)
- local-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-local-delta.pdf)
- broad-random: [PNG](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-random.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-random.pdf)
- broad-delta: [PNG](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-delta.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-broad-delta.pdf)
- Actual/projection/capability trajectories: [PNG](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-trajectories.png) · [PDF](checkpoint-comparisons/final/reproduced-1000--collaborator-lr1e-6-ms1000-trajectories.pdf)

## Scope and limits

The initial attack state is each defense endpoint. The collaborator provides a supplied final checkpoint (reported 125 defense updates), with no observed training history. No intermediate defense history was reconstructed. The ten fixed probe documents, corpus revision, physical basis, and parameter layout are checked. Attack-held-out means excluded from recorded attacker batches, with no claim about defense or pretraining exposure. Those 32-document CE curves are overlaid only when their content hashes agree; the earlier 128-step set can differ. Fitted energy sums squared raw displacements over selected checkpoints, not centered PCA or an equal-time average. It does not guarantee high capture at every state or accurate projected CE. N/A capture means a recorded rank-zero trajectory. Only the declared LR 1e-6 attacks are compared; this is not a maximum over attacker learning rates or optimizer seeds.

Starting utility must be assessed alongside WMDP. The original base MMLU reference is 58.587299%; a five percentage point utility-drop budget requires at least 53.587299%. The CSV records initial utility drops without forcing a defender selection. One defender checkpoint and one attack optimizer seed do not establish variation across seeds.

- reproduced-1000: [early-update-profiles](../../tar-compute/20261110T1815Z/cases/baseline-lr1e-6/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- reproduced-1000: [first-update-landscape](../../tar-compute/20261110T1815Z/cases/baseline-lr1e-6/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- reproduced-128 fixed-probe provenance: The legacy fixed probe is the first ten qualifying documents at the pinned original corpus revision; verified against its source and document hashes.
- collaborator-lr1e-6-ms128: [early-update-profiles](cases/collaborator-lr1e-6-ms128/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- collaborator-lr1e-6-ms128: [first-update-landscape](cases/collaborator-lr1e-6-ms128/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
- collaborator-lr1e-6-ms1000: [early-update-profiles](cases/collaborator-lr1e-6-ms1000/early-update-profiles.png) (existing profile adapter output; first-update q0 differs between conditions)
- collaborator-lr1e-6-ms1000: [first-update-landscape](cases/collaborator-lr1e-6-ms1000/first-update-landscape.png) (existing profile adapter output; first-update q0 differs between conditions)
