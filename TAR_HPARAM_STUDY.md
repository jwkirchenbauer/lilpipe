# Controlled TAR landscape contrasts

Completed measurements: [results](examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/report.md) and [figure index](examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/figure-index.md). This follows the [original reproduction](TAR_REPRODUCTION_PLAN.md), [collaborator checkpoint comparison](TAR_COLLABORATOR_CHECKPOINT.md), and prerequisite [attack/defense basis analysis](TAR_BASIS_STRUCTURE.md).

The interaction between the simulated attack LR and outer defense LR produces the tower-like raised shoulders. Raising retain alone preserves the original plateau; at the higher learning rates, retain raises the center and reduces its depression. These effects are conditional on the selected levels and this training seed.

| Condition | Retain coefficient | Simulated attack LR | Outer defense LR | Measured scope |
|---|---:|---:|---:|---|
| B | 2.5 | 1e-6 | 5e-6 | Reused control, standard attack |
| R | 4 | 1e-6 | 5e-6 | Defense, standard attack |
| I | 2.5 | 1e-5 | 5e-6 | Defense, standard attack |
| RI | 4 | 1e-5 | 5e-6 | Defense endpoint |
| O | 2.5 | 1e-6 | 1e-5 | Defense, standard attack |
| RO | 4 | 1e-6 | 1e-5 | Defense endpoint |
| IO | 2.5 | 1e-5 | 1e-5 | Defense endpoint |
| RIO | 4 | 1e-5 | 1e-5 | Defense, standard attack |

Each defense uses 125 updates and 16 simulated attack steps per update. Each standard attack uses LR1e-6 for 1000 updates. Precision, optimizers, schedules, seeds, cached datasets, and actual batch contents/order match the completed control. The new attacks each have 31 verified task/checkpoint measurements, including WMDP throughout and MMLU at 0/128/512/1000. Intermediate defense checkpoints supply CE and landscape history.

The [eight-condition endpoint findings](examples/tamper-resistance/artifacts/tuolumne/tar-hparams-factorial/20261117T0130Z/static/full-factorial/findings.md) compare all radii with shared scales in the original BF16 convention. At radius 125, IO has six immediate neighbors above center and two slightly below; RIO has all eight higher. These finite samples establish neither a local minimum nor a closed barrier. RI/RO/IO have no attack or benchmark measurements.

The joint RIO recipe still admits immediate descent: fixed-ten CE falls by 0.5901 nats on the first update. After 1000 updates its WMDP accuracy is 44.6%, versus B's 53.6%, while MMLU is 54.0%, versus 58.1%. Its initial utility also misses the original five-percentage-point limit. The lower benchmark recovery therefore accompanies a utility difference; it is not a shape-only comparison or a maximum over attacker recipes.

The native views preserve common random and original 128-step attack-reference directions, and fit each condition's own 1000-step attack plane. First-update, actual-minibatch, attack-held-out, and projection-fidelity views distinguish measured optimizer motion from its two-dimensional projections. Keep this convention separate from the original static BF16 radius series. Matching the three published hyperparameters also leaves the collaborator's legacy data pipeline outside this controlled study.

Exact configs, sources, completed measurements, and execution evidence are in the [main study](examples/tamper-resistance/artifacts/tuolumne/tar-hparams/20261116T2310Z/) and [three-pair endpoint extension](examples/tamper-resistance/artifacts/tuolumne/tar-hparams-factorial/20261117T0130Z/).
