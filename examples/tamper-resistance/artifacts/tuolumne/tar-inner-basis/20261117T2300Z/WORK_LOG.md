# Inner trajectory observation

This study reruns local B from the pinned original Zephyr checkpoint. The training
configuration changes only the observing trainer class, output directory and run
label. K16, 125 outer updates, source data/cache/order, seed42, retain2.5, inner
LR1e-6 and outer LR5e-6 remain unchanged. No additional attacker is trained.

Five actual pre-outer inner calls are observed: 1, 32, 64, 96, 125. Six inner
states per call (1,2,4,8,12,16) remain transient in RAM. The recorder publishes
only exact raw/BF16 Gram and cross-Gram matrices, module blocks, batch identities,
and anchor-gap norms against all26 archived B attack checkpoints. Per-call fits
are compared with a matching six-point attack prefix and the prior13-point
1000-step attack fit. Projection is measured at all26 archived states. Cross-call
inner angles and a pooled span are not measured by these per-call aggregates.

The optional completed-inner-step hook and recorder passed four focused tests,
including stochastic-model update/outer-gradient/RNG equivalence and direct
Gram arithmetic. Four geometry tests passed, including dense-SVD comparisons,
unequal-rank normalization, zero-motion handling and resumed reports. The frozen
geometry copy was also tested directly. Black and Bash checks passed. Test logs
are under logs; the initial pytest command lacked pytest in the train prefix,
so the successful recorder test uses the activated train interpreter with the
existing system Python3.12 pytest appended only for that test process. No
environment was installed or modified.

Source/configuration snapshots and their small-file hashes are frozen in
source-manifest.sha256. The launcher preview was inspected before submitting
f3kiCDSHPFio (100 minutes, one exclusive node, pbatch/guard). The guard bank had
LevelFS0.305 versus verival0.294. Device diagnostics and the existing ROCm BLAS
settings remain enabled. The first post-submission checks show SCHED; no model
work has occurred. The scheduler initially reserved a start around 02:45 UTC
on November18. status/all.json and replay-validation.json are authoritative;
scheduler COMPLETED alone is insufficient because the launch wrapper echoes
after worker exit.

Before drawing conclusions, require original batch equality, exact saved outer
parameter equality at all nine checkpoints, and complete per-call/reference
identity checks. The actual attack starts at outer state125, while inner call t
starts at state t-1. Report this difference and the measured anchor-gap norm;
direction mismatch is not by itself a causal explanation of defense failure.
