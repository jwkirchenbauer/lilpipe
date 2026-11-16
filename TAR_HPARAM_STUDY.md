# Controlled TAR landscape contrasts

Follow the [checkpoint comparison](TAR_COLLABORATOR_CHECKPOINT.md) and [attack/defense basis analysis](TAR_BASIS_STRUCTURE.md), retaining the [original reproduction](TAR_REPRODUCTION_PLAN.md) data pipeline. Change the three published hyperparameter differences individually and jointly:

| Condition | Retain coefficient | Simulated attack LR | Outer defense LR |
|---|---:|---:|---:|
| B: completed local control | 2.5 | 1e-6 | 5e-6 |
| R: retain only | 4 | 1e-6 | 5e-6 |
| I: inner LR only | 2.5 | 1e-5 | 5e-6 |
| O: outer LR only | 2.5 | 1e-6 | 1e-5 |
| RIO: all three | 4 | 1e-5 | 1e-5 |

Run four fresh K16 defenses for 125 updates, then the standard LR1e-6, 1000-update attacks. Preserve precision, optimizers, schedules, seeds, cached datasets, and actual batch order. Save defense and attack histories. Require actual input tensors and their order to match the completed control before accepting a run.

Compare end-defense shape and attack dynamics using the existing common random/reference planes, fitted attack and first-update views, fixed ten and attack-held-out 32 probes, projection checks, and WMDP/MMLU milestones. Also repeat the original static radius series at each new defense endpoint to test the tower-versus-plateau distinction directly. Keep its original BF16 random-direction convention distinct from the newer FP32 orthonormal dynamics. Use shared coordinates and CE/color limits within each convention. Interpret changes in fortress shape together with immediate descent and eventual attack recovery.

These single-change contrasts estimate effects at B, and the joint condition tests their combined effect. They do not identify all interactions. One seed is a screening study. RIO keeps our sampler, so it does not reproduce the collaborator's legacy data pipeline merely by matching these values.

Benchmark coverage includes each defended endpoint as attack step zero, WMDP through the saved attack milestones, and MMLU at attack steps 0, 128, 512, and 1000. Intermediate defense checkpoints receive CE and landscape analysis; they do not repeat the full benchmarks.

Exact configurations and execution evidence: [study directory](examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/).
