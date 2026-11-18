# Simulated inner attack versus the recorded actual attack

Completed an observation-only replay of local B from our original Zephyr
checkpoint: K16, 125 outer updates, retain coefficient 2.5, inner LR 1e-6 and
outer LR 5e-6. All nine saved model states, 125 batches and 2,000 finite inner
losses match the original. The archived standard 1000-update attack was reused.

Recorded inner calls 1/32/64/96/125 at updates 1/2/4/8/12/16. Per-call bases now
come from these actual simulated trajectories, replacing the outer-path proxy.
Exact aggregate inner products support rank-two and full sampled-span comparisons
with matching 16-step and longer 1000-step actual trajectories.

The simulator is partially aligned with the actual attack. At the last call,
the six-direction span captures 12.83% of the first recorded displacement,
55.34% at attack step 16 and 16.44% at step 1000 in BF16-rounded coordinates.
This is far more overlap than the outer-path proxy showed. Different examples,
optimizer settings and starting states remain bundled, so these observations
do not establish why the defense fails.

[Report and numeric evidence](examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/report.md)
and [figures with full paths](examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/figure-index.md).

This follows the [original reproduction](TAR_REPRODUCTION_PLAN.md),
[outer-path basis study](TAR_BASIS_STRUCTURE.md), and
[hyperparameter study](TAR_HPARAM_STUDY.md).
