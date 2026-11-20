# Matrix transition state

## Latest update: 2026-11-20 23:10 UTC

Both full production defenses are advancing on Matrix: K256516131 and full
CUDA K4 control516185. Keep their validated four-GPU placement. The optional
two-GPU pilot is prepared under source/transition/two-gpu and
pilots/cuda-training-two-gpu but DEFERRED, NOT RUN, and not a required study
phase. It was prepared while four-GPU requests queued; the queue then cleared.

Native K4/control evaluations and all six other completed main cohorts have
finished. Fresh base, one-GPU calibration, CUDA parity516180, and CUDA precision
audit516186 are complete. CUDA parity is exact for all16 losses and five checked
states. Both original auxiliary Flux jobs also completed. All26 original
production requests are superseded; no original production work remains queued.

K4 control attempt516132 failed before model loading because the generated case
lacked output directories. Slurm canceled dependents516137/516162/516163/516172
and disallowed requeue. Their rows remain superseded in matrix-jobs.json and the
failed log is preserved. Root created all case directories and the attack's
models/defense symlink. Submission now rejects missing directories (unit test).
Current K4 control chain: defense516185 -> attack516187 -> eval516191 and
analysis516192 -> profiles516197. Analysis also waits on516068; profiles also
wait on516150. No training took place in the failed allocation.

Root verified K4 training/evaluation/geometry complete and fixed CE endpoints
through attack1000. K4 profiles516150 can run concurrently with the remaining
independent grids. Slurm rejected adding a new dependency on the already
completed evaluation516069, so its dependency was removed after verifying the
inputs. This change is recorded in the ledger. Other profiles retain
the full reference-profile gate on516150 and their own analysis gate. All other
analysis requests still wait on full K4 analysis516068.

source/report-matrix is prepared, reviewed, and tested. It revalidates fresh
CUDA base and K16 scores, preserves exact CUDA replay gates, labels ROCm archive
comparisons as diagnostics, and compares the paired K4 scores. It rerenders
cached plots directly without resetting completed phase statuses. The helper
now supports `report final` (120min pbatch); no final report job submitted yet.
Its scientific completion marker is report-matrix-references.json. Final
report must wait for every active native required phase; ignore superseded
failed attempts. source/misc and all currently executing snapshots stay frozen.

Current monitor: PID1301232 on tuolumne1004, literal sleep900 between polls.
Last23:00 snapshot had13 native completions,3 running,23 pending, no errors;
both Flux auxiliaries inactive/complete. Old error text before22:14 is historical.
Commits d96191b1 and b1b74c03 were pushed. Curate fresh small source/config/proof
files and results; preserve user's unstaged root AGENTS.md and unrelated files.
Never add models, optimizers, raw corpora, .npy bases, caches, or cores to Git.

## Latest update: 2026-11-20 22:22 UTC

The notes below this section include earlier preparation states. The current
job ledger is authoritative. CUDA training pilot516076 completed successfully:
initial weights/batches/rates exact, update cosine0.9931, relative update
error0.1174, max loss error0.04494, all256 K256 inner steps. K256 production
defense516131 (180min) and attack516136 (90min, afterok516131) are submitted;
full CUDA K4 control defense516132 and attack516137 are also submitted. These
four requests were pending at22:16. The old K256 training requests and their
three dependent measurement requests are superseded in cancellations.json.

One-GPU full WMDP calibration516135 completed in105.9sec with exact agreement
on all six subtask scores and868 examples against the four-GPU CUDA reference.
Remaining seven main evaluations use the versioned source/eval-onegpu worker;
K4/control evaluations516069/516070 retain their original four-process layout.
K64 one-GPU production checkpoint0 has validated both benchmark groups. Old
remaining evaluation requests have been canceled in favor of native jobs.
Base Zephyr WMDP+MMLU reference516152 uses a fresh standalone output under
control-study/base-reference, for consistent CUDA utility/recovery thresholds.

Native K4 analysis516068 is processing surfaces. K4 profiles516150 wait on its
successful completion. The other main/control/backend-control analyses wait
on516068 (and their attack producer if needed). Their profiles must wait on
their own analysis AND516150, preserving both full validation gates. These
are ordinary pbatch dependencies; no pdebug jobs or production splitting.

The mixed monitor was restarted at22:14, PID1301232, after recording expected
Flux dependent cancellations. It runs literal sleep900 between polls. Old
error text in logs/poll.log before22:14 reflects those superseded requests.

Maintained native runtime changes and42 unit tests were committed b1b74c03;
training lock d96191b1. Neither had been pushed at this update. User's root
AGENTS.md edit remains unstaged (including its existing trailing whitespace).

matrix_transition_audit is preparing the fresh CUDA parity/precision clones.
ROOT REVIEW caught wrong legacy trainer resolution in its first parity wrapper:
the corrected wrapper must use a fresh source/parity-matrix with matched-cache-v2
src, NOT current source/misc + legacy source/src. Do not submit until confirmed.
Same-backend simulated/genuine equality and all16 loss matches stay strict;
AMD archive comparisons are diagnostics. Precision rows must identify each
anchor backend accurately (old archived_rio_v2 is ROCm, new anchor is CUDA).
matrix_train_runtime is preparing source/report-matrix, using fresh base/K16
scores to replace historical report constants; no report job submitted yet.

Remaining: finish profile submissions and supersede old analyses/profiles,
validate/submit auxiliary clones, validate/submit final report after all gates,
measure full K4 backend drift, curate small provenance/results and push. All
current execution snapshots must remain immutable. No raw arrays/models/data
in Git.

This is the active transition root. The original training/provenance root is
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z`.

The user explicitly authorizes moving real work, including training, to Matrix
after validation and accepts expected floating-point differences. No pdebug
chaining or split production allocations. Use launch_matrix.py, pbatch/verival,
dry-run/review/submit. Every Python/env command requires interactive Bash,
set -e, and mm_activate with an absolute prefix. Read root AGENTS.md, ~/.bashrc,
and /p/vast1/kirchenb/llnl-tools/README.md first. No ad hoc pip/env mutation.

Installed CUDA prefixes under `/usr/workspace/wsb/kirchenb`:
`lilpipe-tar-eval-cu126-v1` and `lilpipe-tar-train-cu126-v1`, Torch2.12.1+cu126.
Train lock was reviewed/committed as d96191b1 before installation. All 222
common non-Torch train package versions match the old ROCm train lock. The
correct reviewed installer is ORIGINAL/source/matrix-training-runtime;
the agent's initial run-local installer is unused and lacks activation fixes.

The original completed defenses/attacks are reused as read-only models via
symlinks. Fresh CUDA metrics live here. Cases' settings only add
cpu_landscape_anchor=true and redirect capability score output. K4 geometry
is reused literally from saved .npy directions; the other cases can fit new
geometry from their saved states. K16 control has its own fresh measurement
study at control-study/cases/hparam-rio-lr1e-6. Its old AMP/native training
states remain the control; all inference scores are recomputed on CUDA.

Source/misc is frozen for currently running jobs: DO NOT modify it. It combines
the original legacy training observer/TAR source with the current native
CUDA wrappers and memory-bounded analysis module. The optional profile mode
also keeps anchors, deltas, and random vectors on CPU; all endpoint validations
remain. Tests passed: 24 native-evaluation tests before snapshot; 4 profile
arithmetic tests; 11 dynamics tests. Later maintained evaluator adds one-GPU
support (27 tests passed), snapshotted separately under source/eval-onegpu.
Do not copy it over the currently running four-GPU source/misc.

Actual production jobs are in matrix-jobs.json:

- 516068: full K4 analysis, one H100, 240 min, 256G host memory; RUNNING.
  All actual checkpoint losses through attack1000 have completed, now grids.
- 516069: full K4 evaluation, four H100s,150 min,256G; RUNNING. Full WMDP+MMLU
  checkpoint0 and WMDP checkpoint1 verified complete with no rejected results.
- 516070: full reused K16 control evaluation, four H100s,150 min,256G; RUNNING
  after about eight minutes in queue.
- 516076: training pilot, four H100s,45 min,256G; RUNNING. K4 four-update
  prefix with full125 schedule/12warmup, followed by K256 one-update prefix
  with full125 schedule. Existing frozen segmented-v3 observer supplies the
  stop callback. The new subclass adds only on_train_end GPU memory recording.
  The pilot never changes training max_steps or production configurations.

Pilot validation is pilots/cuda-training/status/validation.json. Required:
initial K4 states bitwise equal, outer data/rates identical, loss delta<=0.05,
relative update error<=0.25, update cosine>=0.95 (predeclared before pilot),
and256 logged inner updates in the K256 prefix. Source compare code and pilot
configs/allowed diffs are under source/transition and pilots/cuda-training.
Use measured K256 outer-step time to size one uninterrupted production run.

K256 production will write the original canonical case outputs with original
YAML/max125/K256 settings. Cancel the queued original Flux defense/attack
before submitting native replacements, after the pilot passes, to prevent
duplicate writers. All other completed training is preserved. A separate
full CUDA K4 backend control is being prepared under backend-control by
matrix_train_runtime, no jobs submitted; it estimates accumulated backend
drift without overwriting original K4.

Only original K4 Flux evaluation f3m4u9CgNceT has been canceled so far, replaced
by516069 after its checkpoint0/1 validations. Original registry is unchanged;
cancellations.json records supersession. Other original requests remain
pending/held. Keep K4 full-analysis/profile-reference validation gates before
bulk analysis/profile migration. Do not release old held Flux requests.

The old monitor818473 was stopped. The active mixed read-only 900-second
watcher is PID1227514 on tuolumne1004, detached with nohup/setsid. It tracks
Matrix jobs plus remaining original Flux requests and both auxiliary parity
and precision jobs, excluding explicit cancellations. Log logs/poll.log;
snapshot status/poll.json. It reports scientific phase failure as well as
scheduler failure, and does not submit/retry.

One-GPU benchmark calibration is prepared but NOT submitted yet:
source/transition/run_onegpu_calibration.{py,sh}, source/eval-onegpu/misc.
Root fixed a reference schema bug: read the actual four-GPU result file from
ORIGINAL/matrix-assessment/benchmark-probe/result.json matrix_result_files,
then verified_task(), rather than nonexistent reference field. It validates
full868 examples/subtask counts, uses same BF16 batch32, and records one-/four-
process score/runtime differences. Request20min,1GPU,128G via launcher, review
dryrun first. If useful, bulk evaluation can use the versioned one-GPU worker
on partially occupied nodes; record process counts and source amendment.

Earlier calibration: nine RIO CE points matched ROCm within0.0052534 nats
(fixed0.01 limit),70.02GiB GPU reserved. Four-GPU WMDP36.8664% versus archived
ROCm36.2903%,868samples, no per-sample hashes in reference. Keep fresh CUDA
control scores for comparisons. These results were pushed in bea41361.

Remaining work: monitor pilots/jobs, transition remaining queued training and
evals/analyses/profiles after gates, port or replace pending auxiliary pilots
without weakening same-backend simulated/genuine parity, adapt final report
runtime/source paths, collect all results/figures, then curate/commit/push.
Do not add raw models, optimizers, basis arrays, corpora, cores, or caches.
Preserve user's unstaged root AGENTS.md and pre-existing untracked files.
