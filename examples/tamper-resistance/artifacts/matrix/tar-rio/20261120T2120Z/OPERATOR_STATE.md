# Matrix transition state

## Final update: 2026-11-21 06:04 UTC

All 44 tracked phases are complete, including strict report516680 and the
render-only title cleanup516682. The Matrix queue is empty. The 900-second
watcher has stopped because there is no remaining work to monitor.

Start at report.md; rio-report.md has complete declared measurement coverage.
All nine profile-overlap input pairs retained their saved CSV/NPZ hashes.
The geometry-reuse adapter reconciles only two operational settings for the
archived K4/K16 geometry; original geometry metadata remains byte-identical.
The render cleanup shortened subplot titles without changing numerical inputs;
plot-rendering-validation.json records both input and final figure hashes.
The final K landscape and trajectory figures have been visually inspected.

Scientific limits are explicit: all four K conditions fail the initial utility
gate. K256 trained on CUDA, while the main K4/K64 and fidelity conditions trained
on ROCm. The paired K4 control differs by up to 5.76 WMDP percentage points and
2.98 MMLU percentage points, so this is a material comparison limitation.
The legacy K sweep also retains the original AMP-cache behavior; exact parity
was demonstrated only for the matched implementation at the tested CUDA anchor.

Final small-artifact curation and commit include completed figures, summaries,
validation proofs, and execution sources. Raw models, optimizer states, basis
arrays, datasets and caches stay outside Git. Preserve the user's AGENTS.md
change and unrelated untracked files.

## Latest update: 2026-11-21 05:07 UTC

All main conditions and the fresh K16 control have finished analyses/profiles.
Only repaired CUDA K4 backend analysis531/profile532 and strict report534 remain.
Last04:54 monitor:38 native completed,2 running,1 pending, plus2 completed Flux
auxiliaries =40/43 complete, no errors. Backend531 reused geometry and now has
complete local frames, loading its attack basis before broad frames. Its first
27GiB basis direction took275sec (slow VAST read, still progressing). Backend
profile532 is near the end of its first-update grid and continues its eight
saved-update segments. Both retain their original requested limits.

All nine profile overlap releases are complete; backend532 used retry-v3 /
peer-v3 and dynamically resolved final534. After everything finishes, recheck
all status/profile-overlap-readiness.json loss hashes and save final proof.
Source/link fix is in backend-control/source-link-amendment.json; no numerical
source or science settings changed. Watcher2067621 remains active, sleep900.

Agent prepared source/transition/final-curation-candidates.json: review-only
inventory with current/future paths, small scalar NPZ exception, no directory
symlink traversal or large tensors/data. Rescan/hash AFTER strict report passes,
do not assume its earlier sizes are final. Final outputs include root report.md,
rio-report.md, backend-control-drift.csv and report-matrix-references.json.
Inspect final key charts and backend drift, curate/commit/push; preserve user's
AGENTS.md and unrelated files. Latest pushed commit da95eecd; subsequent ops,
overlap proofs and source-link fix are still uncommitted.

## Latest update: 2026-11-21 03:56 UTC

The failed backend branch is repaired and resubmitted:
analysis531 (peer-v3) -> profile532 (retry-v3), with final report534 (retry-v3).
All three replaced340/349/351 rows explicitly name their replacements. The
case-local source links now resolve to the unchanged frozen helper; geometry
is reused. The case's analysis.json can still show the old failed state while
531 is pending; the new worker resets it when it actually starts.

Read-only watcher is RESTARTED, PID2067621 on tuolumne1004, parent1, sleep900.
Old1301232 exited on the caught failure. First new03:54 poll:30 native complete,
8 running,3 pending, plus2 completed Flux auxiliaries; no current errors.
Final report534 retains all unfinished full analyses and profiles. Completed
producer markers were verified and omitted from new dependencies (Slurm purge).

Overlap helper now accepts --profile-attempt/--analysis-attempt and resolves
the unique active report from the ledger instead of hardcoding351. For the new
backend profile use retry-v3/peer-v3. Release532 only after its 37-row/7x128
readiness gate passes, or let its own analysis dependency finish normally.
All other eight profiles have already been released or completed. Saved loss
hashes have been rechecked unchanged for K64,K256,forward-legacy after both
analysis/profile completed; status/profile-overlap-verification.json records it.
Continue checking the rest at completion. Full numerical source stays frozen.

Remaining main analyses337,338 and K16 control339 were running. Profiles344--348
were running. Everything trained and all benchmark evaluations are complete.
Continue slow polling, inspect final figures and paired backend drift, complete
strict report marker report-matrix-references.json, curate and commit/push.
Latest pushed da95eecd; later ops metadata/fix/overlap proofs await curation.

## Latest update: 2026-11-21 03:52 UTC

Monitor caught backend-control analysis340 failing in surfaces: the generated
attack case lacked source/misc/reference_get_ll.py because its case-local
source symlink was absent. Geometry had finished and is preserved. Profile349
and final351 were canceled by Slurm's dependency gate. Watcher1301232 exited
on this error and MUST be restarted after replacement jobs are registered.

Root restored both backend case/source links to the same frozen TARGET/source;
source-link-amendment.json records this metadata-only repair. No numerical
source/settings/checkpoints changed. submit_peer.py now checks this helper
path before allocation; four regression tests pass. Old340/349/351 rows are
superseded. Replacement analysis531 is submitted with attempt peer-v3 and
reuses completed geometry. Profile retry-v3 dryrun is being reviewed next,
then final report retry-v3 will be submitted against the current live jobs.
Do not reuse the canceled351 dependency graph or report restoration helper's
hardcoded retry-v2 attempt for real submission.

Other work was progressing normally: last03:44 poll had28 native completions,
10 running plus the above failed/canceled branch. K256 profile342 completed.
All training/benchmarks remain complete. Finish this branch repair, restart
900-second watcher, continue all remaining phases, verify overlap input hashes,
inspect final figures/backend drift, curate and commit/push. Latest pushed
commit remains da95eecd. Preserve user's AGENTS.md and unrelated files.

## Latest update: 2026-11-21 03:34 UTC

Completed full analyses: K4/K64/K256 and both legacy prefix conditions
(329,332,333,334,335). Analyses336--340 are running, including fresh K16 and
paired CUDA K4 controls. K64 profile341 has completed; its saved immutable
CSV/NPZ hashes were rechecked unchanged after both phases, recorded in
status/profile-overlap-verification.json. Profile342 was at update7 around
03:31. Profiles343--345 are running. Readiness checks also passed for forward/
reverse matched and K16 control; root released profiles346,347,348 with their
exact applied input proofs. Only CUDA K4 profile349 retains its own-analysis
dependency until its actual-loss pair becomes complete (or analysis finishes).

Final report351 still requires ALL full analyses and profiles. Last03:29
monitor:26 native completed,10 running,5 pending, plus2 completed auxiliaries;
no errors. All training/benchmarks are complete. Continue slow polling with
sleep900, finish remaining profile releases only after input checks, verify
all overlap loss-file hashes, inspect final figures and backend drift, then
curate/commit/push. Source runtime stays frozen. Latest pushed da95eecd; later
operational files/overlap helper and proof files still await curation.

## Latest update: 2026-11-21 03:17 UTC

K4, K64 and K256 full analyses329/332/333 have passed. Root inspected the
K4 broad/fitted figures and K64/K256 broad evolution figures; rendered labels,
shared within-case scales and checkpoint panels are intact. Do not infer the
whole K comparison before fresh K16 and paired CUDA K4 controls finish.

Five profile jobs341--345 have now been released after the exact immutable
input gate (K64,K256,forward/reverse legacy,fresh matched). Their applied proofs
are under source/transition/profile-overlap-*-applied.json and each case's
status/profile-overlap-readiness.json. No source/settings changed. The other
profiles346--349 remain dependent on their own analyses; check readiness at
later15min polls to overlap them if useful. Final351 retains all analysis and
profile completion dependencies. Verify loss-file hashes after overlaps finish.

Last02:59 monitor:23 native completed,9 running,9 pending, plus2 completed Flux
auxiliaries; no errors. Original training and all benchmarks are complete.
First two released profiles341/342 were running; K64 profile was at update3
around03:11. The remaining analyses are progressing through geometry/grids.
Keep frozen source/peer-landscape and source/misc unchanged. Watcher1301232
still uses sleep900. Latest pushed commit da95eecd; operational metadata and
overlap helper/proofs still need final curation/commit along with results.

## Latest update: 2026-11-21 02:20 UTC

K4 full analysis516329 PASSED. Root visually inspected its
random-landscape-evolution.png and surfaces-and-checkpoints.png; axes/markers
and broad versus fitted views render correctly. All training and evaluations
remain complete. Five sibling analyses were running at the02:14 monitor poll;
no errors. All observed peer assignments expose two H100s with P2P true.

After a read-only code/write-lock audit, root released K64 profile516341 and
K256 profile516342 to overlap their unfinished landscape grids. Both fresh
readiness checks passed: geometry complete, every basis header/stat valid,
37 finite/unique expected actual checkpoint rows through1000, complete finite
7x128 minibatch matrix matching settings. Exact CSV/NPZ hashes and geometry
identities are saved in each case's status/profile-overlap-readiness.json and
source/transition/profile-overlap-*-applied.json. New helper overlap_profiles.py
first dry-runs the checks, then --apply removes only the profile dependency.
Analysis and profile outputs/locks are disjoint. Final report516351 STILL waits
for every full analysis and every profile. No numerical settings/source changed.

At subsequent15min polls, the same per-case gate may release profiles343--349
once their actual outputs are complete; do not release merely because geometry
exists. If their own analysis finishes first, no change is needed. After each
overlapped profile/analysis finishes, verify the saved immutable CSV/NPZ hashes
are unchanged and record that proof before final curation. Pending profiles
remain original one-GPU workers. Watch1301232 is alive with literal sleep900.

Continue to completion, inspect the final comparison figures/backend drift,
curate small result files (no tensors/corpora), update curation hashes, and
commit/push. Latest pushed commit is da95eecd; later operational metadata is
not committed yet. Preserve user AGENTS.md and unrelated files.

## Latest update: 2026-11-21 01:19 UTC

K4 peer516329 started00:56:34UTC, earlier than the forecast. Production device
checks passed on matrix22: global Slurm IDs1,3, inherited local CUDA mask0,1,
two distinct H100 UUIDs, peer access true in both directions. It reused
completed geometry/actual losses/grid rows and is now in the last broad frame
(attack1000). Keep the full K4 completion gate for siblings516332--516340.

All training AND all benchmark evaluations are now complete, including K256
evaluation516146. Existing final-report dependency on146 was established while
it was live and is valid; do not edit merely because its record later purges.
Monitor1301232 remains healthy, polls every900sec. Last01:14 snapshot:21 native
completed,1 running,19 pending, plus2 completed Flux auxiliaries; no errors.

Commit da95eecd is PUSHED: frozen peer runtime, exact6.1784x pilot proof,
binding/localization tests, reviewed restored DAG and backend reference fix.
Only user's AGENTS.md change and unrelated untracked files remained after it.
Next: verify K4 completion and inspect its figures, monitor sibling analyses,
profiles341--349 and final report351, resolve failures, assess paired backend
drift, curate final small results/figures, and commit/push. Do not mark the
study finished before report-matrix-references.json passes.

## Latest update: 2026-11-21 00:43 UTC

The DAG is fully restored. K4 peer-v2 analysis516329 (2GPUs/56CPU/224G/360min)
is pending for resources. Its last provisional forecast was01:25UTC. Nine
sibling analyses516332--516340 wait afterok516329. Nine profiles516341--516349
wait on their respective analyses; the completed K4 reference profile516150
is verified directly, avoiding Slurm dependencies on purged completed jobs.
Strict final report516351 waits on all19 new analysis/profile jobs plus current
K256 benchmark516146. Its completion marker is report-matrix-references.json.
All requests are pbatch/verival. Exact reviewed command plans are
source/transition/restored-{analyses,profiles,report}-plan.json.

No new peer source edits: final runtime was validated by516325 (nine loss pairs
exact, tested model bytes exact,6.1784x point throughput). Failed attempts and
canceled descendants remain explicitly superseded in the ledger. The final
report independently checks ten attack cases and does not read old job IDs.

All training is complete and validated. Continue sleep900 monitoring via
watcher1301232; verify the first production peer assignment/P2P capability and
cache reuse when516329 starts. Finish analyses/profiles/report, inspect figures
and paired K4 backend drift, curate final small results, then commit/push.
Preserve user's unstaged AGENTS.md and unrelated untracked files. No raw model,
optimizer, basis-array, data-cache or corpus files should enter Git.

## Latest update: 2026-11-21 00:39 UTC

The final mask-preserving two-GPU pilot516325 PASSED: nine CE pairs exactly
equal, tested perturbed model bytes equal,6.1784x point throughput. Explicit
--gpus_per_task=2 binding probe516323 also passed on both H100s. Final worker
source/peer-landscape is FROZEN; do not edit. Production uses two visible GPUs,
preserves Slurm's local CUDA mask and validates published global IDs without
using them as CUDA indices. No source math/probes/radii/coordinates changed.

New K4 analysis516329 is pending for resources (six-hour pbatch/verival request,
2GPUs/56CPU/224G); last forecast01:25UTC, provisional. It resumes cached rows.
Nine sibling peer-v2 analyses516332--516340 are submitted afterok516329; all
remain blocked until full K4 validation. The nine retry-v2 profile dryruns are
under review next, followed by final report resubmission. restore_dag.py keeps
reviewed command plans separate from real submission. Old canceled attempts
remain superseded; do not requeue or reuse them. Final report source remains
source/report-matrix, its strict completion marker unchanged.

Training is all complete; benchmark516146 is the only unfinished evaluation.
Watch1301232 remains alive with sleep900 and now tracks the restored ledger.
Remaining work: finish submitting profiles/report, monitor/recover, inspect
finished figures and backend drift, curate small source/proof/result files,
commit/push, and finish the strict report. Preserve user's AGENTS.md edit.

## Latest update: 2026-11-21 00:28 UTC

All production training is complete, including K256 attack516136. K256
benchmark516146 is running; every other benchmark is complete. Original K4
profiles516150 passed. Remaining work: landscape grids, other profiles, strict
report, figure inspection, curation and push.

The peer landscape pilot516308 PASSED: identical losses at all nine points,
identical perturbed parameter bytes,6.2818x point speedup. K4 replacement516314
then failed BEFORE Torch: despite a two-GPU request, its step supplied only
SLURM_STEP_GPUS=3 and CUDA_VISIBLE_DEVICES=1. Its dependency gate canceled all
remaining analysis/profile/report allocations. These attempts are preserved
and marked superseded in matrix-jobs.json. New allocations must be submitted;
old IDs516219 and516227 are no longer live. Completed cached rows survive.
The old active K4 analysis516068 and its pending closure516227 were deliberately
canceled only after all consumers were rewired to516314, so no duplicate writer
is active. The interrupted peer-rewire script performed no successful updates.

Explicit per-task binding probe516323 is submitted (2GPUs,56CPU,224G,5min,
pbatch/verival, --gpus_per_task=2). It records inherited visibility, UUIDs and
device cgroup rules; it never widens the mask. Await that proof. The peer agent
is changing matrix_env/runtime to preserve inherited CUDA visibility even when
global Slurm IDs differ, while still requiring exactly two distinct H100s.
No peer numerical source is running during these edits. Source hashes will
change, requiring a NEW nine-point pilot before retrying production. The peer
submission helper is being amended with explicit binding and attempt suffixes
to retain failed ledger entries. Submit only K4 first; retain its full validation
gate before releasing siblings. All profile workers stay on the frozen one-GPU
source. Final report source/report-matrix remains frozen and can be resubmitted.

Backend-control/study.json now explicitly names rio-k4-cuda-lr1e-6 as its profile
reference; otherwise the default referenced an absent baseline case. Only root
metadata changed; per-case settings and numerical source are unchanged. The
amendment is recorded in backend-control/profile-reference-amendment.json.

## Latest update: 2026-11-21 00:13 UTC

K256 defense516131 is complete and its 1000-step attack516136 is running on
Matrix. The paired full CUDA K4 defense, attack, benchmarks and the original
K4 profile516150 have all completed. The read-only sleep900 watcher1301232
is still alive on tuolumne1004. The main outstanding work is K256 attack/eval,
landscape grids, the remaining profiles, and the strict report.

A separate two-GPU LANDSCAPE pilot was submitted as516304. It stopped before
Torch/model loading because SLURM_STEP_GPUS was absent. Some Matrix nodes
provide allocation isolation only through the device cgroup (as in successful
profile516150). The peer runtime is being amended to preserve the inherited
visibility when assignment IDs are absent and require exactly two distinct
H100 devices after initialization. It never guesses physical device IDs or
widens visibility without recorded Slurm IDs. This feasibility pilot is kept
outside the required production ledger, in pilots/tar-rio-peer-landscape-pilot/
submissions.json. Existing one-GPU analyses/closure remain unchanged.

The landscape pilot requires identical perturbed parameter bytes, nine CE
differences <=1e-4 nats, and greater than2x synchronized point throughput to
justify two GPUs. Only adopt after all gates pass. Its source is under
source/peer-landscape. The earlier two-GPU TRAINING pilot remains deferred;
all production training uses the validated four-GPU map.

## Latest update: 2026-11-20 23:40 UTC

The complete CUDA K4 defense and its 1000-update attack have passed training
and input-order validation. Their evaluation516191 is running. K256 defense
516131 is advancing; its original attack/evaluation jobs remain correctly
dependent. K4 profile516150 is running concurrently with the original K4 grids.
All other analysis/profile requests retain their scientific validation gates.

All nine pending analysis limits were increased in place to360min. Slurm
rejected increasing the active K4 allocation516068. A new six-hour pbatch
continuation516227 waits afterany516068 and invokes exactly the same frozen
worker on the same case. It reuses completed stages/grid rows; if the original
finishes fully, the continuation only validates/reuses them. No pdebug use.
The original K4 row is marked superseded for required completion tracking but
its allocation still runs. Every downstream analysis now waits afterok516227
instead of516068. Other already-established attack dependencies are retained;
completed516187 was omitted from newly edited dependencies after verifying
its scientific completion because Slurm had purged its controller record.

Final report516219 is submitted with all remaining required native dependencies,
now including516227. The report's completion marker is
report-matrix-references.json; source/report-matrix must now stay frozen.
The K4 continuation plan is source/transition/k4-closure-plan.json. Current
dependency lists and resource amendments are recorded in matrix-jobs.json;
the original argv fields remain as historical submitted requests.

Commits a41ba1d9 and031dda60 are pushed: execution snapshots, configurations,
complete calibration/replay/precision proofs, and clarified precision measures.
CSV exports were normalized from CRLF to LF with parsed rows checked equal;
raw copies remain outside Git and csv-curation.json records both hashes.
The optional two-GPU pilot remains deferred and unrun. No numerical source or
scientific training settings changed during these scheduling amendments.

Continue slow monitoring (PID1301232, sleep900). Resolve failures, inspect the
finished figures and numerical report, assess K4 backend drift, curate final
small results/plots, and commit/push. No results should be declared final before
the strict report passes. User's root AGENTS.md remains unstaged and unchanged.

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
