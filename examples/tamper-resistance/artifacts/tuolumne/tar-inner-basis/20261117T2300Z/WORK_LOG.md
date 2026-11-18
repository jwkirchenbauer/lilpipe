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

The 100-minute queue reservation remained several hours away. After confirming
that job was still SCHED, it was canceled without allocation. The unchanged
frozen training worker was resubmitted as training-only in a 60-minute pdebug
window (f3kiHbvUTK5q); launcher-short-window-v2.json records the inspected preview
and exact argv. Its initial reservation was 23:28 UTC, rather than 02:45 UTC.
Validation/reporting will use a separate short allocation, gated by the actual
train status. The defense update horizon and all scientific settings remain
unchanged. Optimizer/RNG checkpoints at 32/64/96/125 permit investigation and
continuation if needed; no duplicate training worker is active or queued.

The v2 worker started at 23:26 UTC and failed its topology preflight before
model loading: ROCR_VISIBLE_DEVICES exposed one GPU while the worker expected
four. The launcher then echoed completion, so its scheduler COMPLETED state
does not indicate scientific success. The v3 launcher sets gpus_per_node=4,
which aligns ROCR/HIP/CUDA visibility with the four-device model map. The shared
launcher intentionally retains its exclusive-node, one-outer-task request and
does not support a gpus_per_task override; this matches the existing manual
device-placement pathway. The previous plan had incorrectly declared only one
visible GPU while the worker expected four.
The original source, hyperparameters, dataset, and seeds remain unchanged. No
training output or inner observation exists from v2.

The corrected v3 request passed topology checks and started at 00:17 UTC on
November18. Its first16 inner updates were captured transiently, but CPU FP64
Gram reduction advanced only32/291 parameters in roughly10 minutes. Read-only
process diagnostics showed CPU activity and about68GB read; no outer checkpoint
or completed outer observation had been published. This reduction would not fit
the allocation. The run was stopped before the first outer update; its incomplete
marker and logs are retained as history (f3kiN4NibZAw, inactive/canceled).

The new frozen reduction-gpu-v2 profile changes aggregate measurement only:
FP64 products run on cuda:0 with1,048,576-element blocks. It preserves the
original source snapshot and exact B config. The optional CPU/default path and
GPU selection/chunk settings passed four focused CPU tests, including unchanged
inner parameters, outer gradients, inputs and RNG. A compute-node preflight
compares CPU/GPU aggregate arithmetic on fixed toy tensors and checks active
device/RNG preservation before loading the real model. The profile writes fresh
inner-gpu-v2 aggregates; none are mixed with the unfinished CPU attempt. Its
versioned replay validator and analysis spec point only to that new directory.

GPU reduction request f3kiuKH4AK75 was submitted after inspecting the v4 dryrun
and confirming the old CPU worker was inactive. Its initial reservation was
01:11 UTC. The profile manifest includes all new numerical/preflight/validation
code; the original manifest remains intact. The updated maintained tests also
verify explicit CPU selection, chunk-size metadata and rejection of unsupported
reduction devices. No new attacker training has been added.
