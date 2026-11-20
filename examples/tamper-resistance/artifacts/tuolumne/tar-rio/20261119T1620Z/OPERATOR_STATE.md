# Current execution state

Current migration and monitoring state is now maintained at
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/OPERATOR_STATE.md`.
The user authorized moving training to Matrix and accepts expected backend
floating-point variation. The record below describes the earlier Tuolumne
queue state; consult the Matrix ledger and cancellations.json for replacements.

## Resumed on tuolumne1004

The user resumed the session on 2026-11-20. The read-only 900-second monitor
was running in session 36640 from 18:19 through 19:34 UTC, then exited.
It was restarted detached with nohup/setsid on `tuolumne1004` at 20:37 UTC,
PID 818473 (parent 1, separate session 818473). All 26 production allocations
remained pending then (15 SCHED, 11 DEPEND).
No production jobs have been moved or canceled. The Matrix side assessment
is under `source/matrix-assessment`; it uses separate short `pbatch/verival`
probes and leaves the production registry unchanged. Matrix job 515248
started immediately but failed before running the probe because the launcher's
default `cuda/12.6.0` module is unavailable. Job 515334 then passed the CUDA/BF16
probe using `--cuda_module=`. Four-process probes 515362 and 515386 exposed
device-mask issues. Job 515394 passed on all four H100s after its wrapper
verified ownership of all four step GPUs and explicitly exposed that set.
All five Matrix probes are inactive and remain outside the production registry.
The user explicitly authorized the dedicated CUDA environment and checkpoint
comparison with 'Go ahead' on 2026-11-20. The reviewed installer completed
successfully: `/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1`,
Torch 2.12.1+cu126, Python 3.12.16. The nine-point RIO comparison job 515852
completed and passed its unchanged 0.01-nat tolerance (maximum error 0.0052534,
peak GPU reservation 70.02 GiB). Results are under `matrix-assessment/checkpoint-probe`.
A full 868-example WMDP benchmark comparison completed as Matrix job 515883,
using the standard four-process Accelerate invocation and frozen evaluation
helpers. It uses separate `matrix-assessment/benchmark-probe` outputs.
The old benchmark result has empty task hashes, so per-sample hash comparison
is unavailable. All nine frozen task-source files and full sample counts
match. Matrix scored 36.8664% versus archived Tuolumne 36.2903%, a net five
more correct answers (+0.5760 percentage points). Reused benchmark controls
must be rerun on CUDA before comparing small effects across the migrated
matrix; do not silently combine backend scores. Both allocations started
immediately and are inactive/COMPLETED with scientific result state complete.
Neither Matrix job is in the
26-phase production registry. See the root `TAR_MATRIX_ASSESSMENT.md`.
The optional CPU-center landscape code has eleven passing CPU tests; all
current Tuolumne source snapshots remain unchanged. The nine-point completed-RIO
checkpoint comparison passed. Its fixed tolerance is
0.01 CE nats; do not relax it automatically or infer whole-study portability.
Matrix runtime preparation, optional CPU-center support, its tests, and the
assessment document were committed and pushed as `45ef3d0c`. The user's local
AGENTS.md edit permitting launch_matrix.py is preserved and unstaged.
The completed checkpoint/benchmark validation, installation provenance,
numerical comparisons, and updated assessment were committed and pushed as
`bea41361`. Source snapshots and small results are curated; raw weights,
basis arrays, cache files, core dumps, and raw launcher logs remain outside Git.
The checkpoint probe dry run in `launches/matrix-assessment/launch-k5zk5s7z`
was reviewed: 45 minutes, one H100, 28 CPU cores, pbaccount verival, fresh
CUDA prefix. It also requires `SBATCH_MEM_PER_NODE=256G` at submission, because
the memory request is supplied through Slurm's environment. Authorization,
installation, submission, and validation are now complete; do not resubmit
the same probe against its completed output directory.

## Login-node migration history

The user explicitly paused the Codex session on 2026-11-20 to move login nodes.
The old login node is `tuolumne2151`. Its read-only production monitor and
standalone polling sleep have been stopped. Session 85225 is no longer a live
monitor; do not assume it continues on the new node. **Batch requests were left
in place.** At 18:06 UTC all 26 production allocations were still pending
(15 SCHED, 11 DEPEND); both auxiliary requests were pending too. No new
measurements or errors appeared overnight, and no pdebug jobs were active.

On explicit resume, read the repository AGENTS.md and this whole file, then
read ~/.bashrc and /p/vast1/kirchenb/llnl-tools/README.md before Python/env work.
Recheck live Flux states and scientific status files before releasing any hold.
The authoritative production registry is `batch-operation-jobs.json`; the
auxiliary jobs are `f3m5LNyUCuVq` (new-anchor parity) and `f3m6GBJfJki3`
(precision audit). Keep both outside the 26-phase production registry.

The read-only monitor can then be restarted from the example directory with:

```bash
bash -ic 'set -e
mm_activate /usr/workspace/wsb/kirchenb/tuo-torch-rocm72-demo
python -u artifacts/tuolumne/tar-rio/20261119T1620Z/source/ops-batch/poll.py --watch >> artifacts/tuolumne/tar-rio/20261119T1620Z/logs/batch-poll.log 2>&1'
```

It polls every 900 seconds and never submits or retries jobs. Check the two
auxiliary jobs separately. The K4 surface/profile gates below still apply.
The final batch report is prepared but unsubmitted. Commits through `831abe9a`
are pushed; newer run artifacts, precision-audit code, and operational notes
remain on shared VAST storage for final curation. Preserve all local artifacts.
The untracked example test `tests/test_tar_segment_ops.py` and the 16 KiB
`tuolumne1009-python-3777779.core` also remain; do not accidentally add raw
weights, optimizer states, tokenized corpora, basis arrays, or core dumps to Git.

## Execution details to resume

**User instruction: no job chaining or split production work in `pdebug`.**
This is recorded in the root AGENTS.md and `status/no-pdebug-continuations.json`.
All debug host watchers are stopped. All five K256 debug-chain requests were
canceled with zero runtime, as were ten queued debug measurements. Do not
restart any `ops-debug` watch or `ops-segments` execute command.

Current production requests are in `batch-operation-jobs.json`; the manager is
`source/ops-batch/submit.py`. Its 26-job plan uses `pbatch` only: one continuous
K256 defense (currently 240 minutes), its original 1000-update attack, and the eight cases'
evaluations, landscapes, and early-update profiles. Normal batch dependencies
are allowed. Source snapshots and numerical analysis settings are unchanged.
All 26 production requests are submitted; see `logs/batch-production-submit.log`
and `logs/batch-k4-submit.log`. The read-only monitor is
`source/ops-batch/poll.py --watch`, writing `logs/batch-poll.log`.
Its old session 85225 was stopped for the login-node migration above;
the current detached watcher is PID 818473 on tuolumne1004.
It polls metadata every 900 seconds and never submits or retries anything.

Both independent K4 debug allocations ended. Evaluation `f3m4LzCv3XxK` reached
the 60-minute allocation limit, preserving verified partial task results.
Analysis `f3m4LzTgxNqd` completed geometry but its surface worker crashed with
exit 139 at attack checkpoint zero, after the defense actual losses succeeded.
**The launcher masks child failures: Flux reported that analysis allocation
COMPLETED even though status/analysis.json says failed. Check scientific status
files and validations, never allocation success alone.** A premature commentary
claim that landscapes were complete was corrected explicitly.

K4 is now continued in batch: evaluation `f3m4u9CgNceT`, analysis
`f3m4u9bi9TZ9`, and profiles `f3m4uEWvPHxj`. Geometry and all valid caches are
reused, with no numerical source/settings changes. Seven other landscape jobs
and all eight profile jobs are held at urgency zero until the K4 surface retry
is validated. Their IDs and prior urgencies are in `status/landscape-holds.json`.
Only K4 analysis is enabled for this controlled retry. After K4's scientific
analysis status and validation pass, release its profile job and the seven
other landscape analyses by restoring their recorded prior urgencies. Keep
the seven other profile jobs held until K4 profiles are complete and validated:
the study's `profile_reference_case` is K4, so these profiles need that shared
reference. Do not infer any of these completions from Flux COMPLETED. Training
and all benchmark jobs remain enabled.

Read-only diagnosis found an exact-code precedent in the collaborator run:
its surface stage also exited 139 during a periodic asynchronous stack dump.
That run subsequently completed with a diagnostics-only trace wrapper. The
same wrapper is now used via `source/analysis-runtime`; the numerical module
remains byte-identical at SHA256 `e8ab333a...f1c953054`. Fatal-fault and manual
SIGUSR1 reporting remain enabled. `analysis-runtime-amendment.json` preserves
the exact operational change, old/new wrapper hashes, and confirmation that
all 16 landscape/profile allocations were unstarted at amendment time. The
cause remains unestablished. The K4 retry now uses this operational wrapper.

A read-only packing audit left the pending plan unchanged. Evaluation already
uses four Accelerate processes on four GPUs per case; bundled evaluation cases
would run serially. Landscape/profile workers expose one selected GPU per case.
Their existing group interface caps concurrent cases at two, with no verified
successful group allocation found in this study or the original reproduction.
Do not retrofit four-case bundles or bypass the K4 surface/profile-reference
gates. The analysis-runtime group recursion retains the diagnostics wrapper.

Model work and plotting always go through launch_tuo.py; the production manager
is metadata-only and requires explicit interactive Bash activation.

All new defenses and standard attacks except K256 have completed. K4, K64,
and five K16 fidelity variants are measured against archived RIO K16. Preserve
the legacy AMP behavior in the K sweep. Matched-update fidelity variants use
the frozen `matched-cache-v2` source. The real 7B parity pilot v2 was bitwise
exact at states 1/2/4/8/16 and had zero error in all 16 losses.

An independent second-anchor parity verification is now queued at the new
forward-matched defender: `pilots/rio-prefix-new-anchor/job.json`, job
`f3m5LNyUCuVq`, one 45-minute `pbatch/guard` allocation with no dependencies.
It uses the same frozen matched-cache-v2 code and 1000-update schedule, stops
after 16 updates, and compares against the already recorded new matched
condition's attack states. This is outside the 26-job production registry;
monitor its scientific status separately and require it complete before the
final readout. Do not assume exact parity before its result exists.

A read-only loader audit found that exact replay begins after checkpoint
reload: the saved defender has 67 FP32 tensors, while the Axolotl loader path
loads BF16 before restoring norm/embedding/head modules to FP32. Value loss at
this boundary is supported by source, but has not been measured. The outer
straight-through expression `anchor + (attacked - anchor).detach()` also was
not covered by the replay pilot. `rio_replay_protocol` is preparing one small
saved-tensor precision audit under `source/precision-audit`.
It will compare both native checkpoints to their saved loaded parity anchors,
and quantify the straight-through arithmetic at those parity anchors. This
does not reconstruct native training-time inner endpoints. Keep these scopes
explicit; do not claim perfect training-time fidelity from replay parity.
After review and three passing focused tests, this audit was submitted as
`f3m6GBJfJki3`: one 45-minute `pbatch/guard` allocation depending on the
new-anchor parity job. It is outside the 26-job production registry. It uses
the exact root model files used by both pilot configurations, not separate
checkpoint saves. Require its scientific JSON state complete and its CSV
before final reporting; the prepared final worker now checks and summarizes
these results. A synthetic metadata smoke test checks summary arithmetic.

Enabled pending production requests now use the permitted `guard` fallback
within `pbatch`; held requests remain unchanged. The fair-share check and
`allocation-bank-amendment.json` record this. Pending single-allocation limits
were sized from measured runtime/cache coverage: K256 defense 240 minutes,
K4 surface retry 180, K4 remaining evaluation 120. All update budgets, inputs,
workers, dependencies, and numerical settings are unchanged. See
`allocation-duration-amendment.json`; there is no debug continuation.

The auxiliary K4 restart experiment also reproduced all model tensors, losses,
batches, rates, and counters exactly; optimizer/scheduler/RNG files were
byte-identical. Its original checker compared allocator telemetry, causing a
false failure. `pilots/resume-proof/comparison.v1.json` preserves that result;
`validate_numeric_history.py` documents the strict allocator-only amendment,
and `comparison.json` plus `proof-status.json` record the numerical pass.
This proof does not authorize debug chaining.

Source commits through `831abe9a` are pushed; tracked source changes are clean.
The maintained final RIO report
contains required fixes for missing attack K metadata and fidelity replay
audit wiring. Do **not** run the old frozen final report module unchanged:
The corrected report-only profile is now frozen under `source/report-v4`,
retaining the original numeric tar_dynamics.py byte-for-byte. Its
`run_report.sh` and `launch-argv.json` are prepared, with a reviewed 120-minute
`pbatch` dry-run, but no final report allocation is submitted. The worker
requires every declared scientific phase to be complete, collects final
benchmark CSVs, rerenders cached native figures/profiles, then runs the fixed
strict RIO report. Keep any eventual final-report job record separate from
the 26-job production registry used by the read-only monitor.
The final report must wait for complete measurements, utility evaluation, and
validation; avoid numeric tables of pending results. Raw checkpoints, optimizer
states, basis arrays, and tokenized training batches stay out of Git. Curate
source/config provenance, validated numerical results, reports, and figures.

Shell cwd is always the tamper-resistance example. Before Python/environment
work, read ~/.bashrc and /p/vast1/kirchenb/llnl-tools/README.md and activate an
absolute prefix through `bash -ic 'set -e; mm_activate ...; ...'`. Bootstrap is
`/usr/workspace/wsb/kirchenb/tuo-torch-rocm72-demo`; train and evaluation prefixes
are `lilpipe-tar-train-rocm72-v4` and `lilpipe-tar-eval-rocm72-v2` under that same
workspace. The bootstrap lacks PyYAML; use the train environment for code that
needs it. Do not install or mutate environments. Poll slowly with `sleep 900`.
