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

The GPU worker started earlier, at00:39 UTC. Its toy CPU/GPU comparison passed
with zero error and preserved RNG/device state. Shared-storage scans still took
about46 minutes for the first observation. That completed aggregate is retained
under inner-gpu-v2; outer checkpoint1 and its first batch/loss match B. The job
was canceled at01:31 UTC, and its incomplete defense output was archived under
history/remote-reference-v4/models/defense before restarting.

Execution profile reference-local-v3 changes reference transport only: exact
temporary copies on node SSD/RAM, source/destination SHA verification, bounded
capacity, and private ownership-checked cleanup. The recorder remains byte
identical to the GPU profile. Seven synthetic staging tests passed. The new
worker interleaves host allocations across NUMA nodes and retains the same
explicit four-GPU model placement. All scientific settings remain B. The
inspected v5 launcher was submitted as f3kjLy4He4Qw after confirming the prior
worker inactive. Full replay and transfer validation remain required.

The v5 request passed its GPU arithmetic preflight, then failed before staging
or training because the node Bash lacks EPOCHSECONDS. Profile reference-local-v4
uses the portable date command for the private cache suffix. The failed profile
is retained unchanged; the new inspected v6 launch makes no scientific change.

The v6 request f3kjNJR1JYnB started immediately at01:38 UTC on tuolumne1016.
Exact staging completed for405,224,603,064 bytes across27 weight files (about
180GB SSD and225GB tmpfs). Training began at01:49 UTC. Model diagnostics confirm
one process and the intended four-device layer map. The first observed inner
call is in progress. Filesystem capacity does not establish unified-memory
headroom; read-only node diagnostics showed108GiB available during capture.

The maintained analyzer now also partitions globally fitted basis-axis norms
across module families. This differs from fitting separate axes within each
family. Five synthetic geometry tests passed, including dense-coordinate SVD
agreement and empty spans; the frozen postprocess-v4 copy was tested directly.
Black checks passed. Code is committed as8699a3e0. Postprocessing additionally
checks all2,000 inner losses for exact equality, summarizes shifted padding
targets/document overlap from existing records, and compares the completed first
aggregate with the remote-reference trial. No new model training is introduced.

Despite local references, CPU block construction remained the bottleneck
(144/291 parameters after roughly14 minutes of reduction). The v6 trainer was
interrupted before its first outer update. Axolotl caught SIGINT and returned
success after saving an initial model, so its wrapper status incorrectly said
complete; this is not a completed defense. Its output is archived under
history/local-reference-v6/models/defense and the transfer provenance confirms
explicit cache cleanup. The new worker checks final global_step125 before
declaring training complete. Original replay-validation requirements are intact.

Profile reference-local-v5 opts into GPU block construction as well as reduction.
It transfers native slices, preserves FP32 subtraction and direct-endpoint BF16
rounding, and holds about528MiB of live chunk payload. The CPU/default path stays
unchanged. Five focused tests passed, including mixed endpoint dtypes and BF16
rounding boundaries. GPU gap-dot reductions may differ in low bits; preflight
now compares those too. The frozen profile and launcher preview were inspected,
and100-minute pbatch request f3kjfWPuvCko started at02:16 UTC on tuolumne1222.
No prior training worker remains active. GPU parity, speed and full replay
agreement are still to be measured on this profile.

The GPU-native scan remained slow. Synthetic benchmark f3kjqsZiK7nP isolated
resident GPU construction at4.3ms/chunk versus231ms for CPU construction;
GEMM took under1ms and explicit split-K did not help. File-backed benchmark
f3kjuKNyNuZ9 then measured fresh-map direct builds around95–157ms versus4ms
for resident arrays, with exact Gram/gap parity after materialization. This
identifies mapped-file transfers as a concrete issue; synthetic timings alone
do not establish total real-model speed or exclude memory pressure.

An attempt to extend v7 to3h was rejected: running duration updates require
instance-owner privileges. SIGINT did not terminate the nested worker cleanly;
f3kjfWPuvCko was then canceled and confirmed inactive before any outer update.
Its cache provenance still says complete rather than cleaned; physical cleanup
on tuolumne1222 is unconfirmed. Its model output and partial marker are retained
as interrupted history, not scientific results.

Reference-local-v6 uses owned, native CPU parameter copies before GPU transfer,
one parameter at a time. Stream synchronization preserves their lifetime. The
largest clone payload is14,155,776,000 bytes across27 references. Six focused
tests passed, including exact mixed-dtype aggregate equality and archive hash
preservation. This change is committed as7dcc31f8. The inspected240-minute
pbatch replay f3kjxbbFnWk7 started at02:53 UTC on tuolumne1825. All B scientific
settings and the five observation calls are unchanged. The long time limit is
headroom; no other training worker is active. Full replay validation and final
inner-versus-actual geometry remain outstanding.

At03:25 UTC, the v8 replay had reached outer step16. Its first full inner
aggregate completed at03:17 UTC. Canonical JSON comparison of every whole-model
and family matrix/norm field (excluding anchor-gap scalars) was exact against
the earlier CPU-build/GPU-reduce aggregate from f3kiuKH4AK75. Allocated final
postprocessing still checks scalar-gap tolerance and saved defense equality.
This confirms the storage change preserves the first actual7B measurement; it
does not replace full replay validation or establish later-call parity.

Read-only protocol audit found no execution-blocking issue in postprocess-v6.
All five selected old-B inner batches and actual steps1–16 have full512-token
attention masks: each16-row set has8,176 active shifted targets and zero active
padding targets. Thus the known padding-policy difference is not realized in
these sampled comparisons. Token-plus-mask signatures overlap zero times with
the actual prefix; this establishes different tokenized rows, not original
document identities. No archived actual step-zero exists after model loading,
so saved-anchor equality cannot establish native in-memory initialization
equality. Report raw and BF16-rounded endpoint geometry separately.

At04:16 UTC a tool-server restart interrupted the assistant, while the cluster
job continued normally. Recordings1/32/64 were complete and96 was in progress;
the latest completed outer update was95. No training was relaunched.

Before final completion, an operator error in the frozen worker's completion
check was found: it looks for trainer_state.json in the model root, whereas
the original Axolotl output only has it in checkpoint-125. The running worker
is unchanged and its raw status will be retained. The unexecuted postprocess-v6
gate now checks the actual terminal checkpoint and still requires exact saved
values at all nine checkpoints, all2,000 inner losses, input-order equality,
the actual saved attack anchor and cleaned transfer provenance. It writes a
separate status/verified-defense.json after those checks.

Completed: f3kjxbbFnWk7 exited at04:48 UTC after125 outer updates and all five
inner observations. The frozen root-state path check then failed as anticipated;
the current1825 cache was explicitly cleaned. Postprocessing f3kksk1z9NGP was
submitted via the launcher, then its pending queue was changed from pdebug to
pbatch using the supported update API. It ran at04:51–04:59 UTC. All125 batches,
all nine saved model values/dtypes and the2,000 ordered finite inner-loss pairs
match B exactly. The final model also equals the archived attack's saved source.
Whole/family first-call matrices match the earlier implementation exactly;
anchor-gap summation differs by at most2.05e-14, within the declared FP64 check.
status/verified-defense.json and status/postprocess.json record verified completion.

Independent interpretation review confirmed partial, horizon-dependent simulator
alignment rather than the outer-path proxy's near orthogonality. The report
separates first recorded displacement from an unarchived post-load step zero,
and separates geometric capture from causal resistance. All four figures were
visually inspected. A capture-plot legend/axis-label collision was corrected by
f3kkzzKnKyeX using cached numeric tables; all CSV hashes remained unchanged and
render-v2-provenance.json records the new presentation. The corrected figure was
inspected again. Report.md and figure-index.md contain the completed results.
