# Collaborator checkpoint comparison execution

The supplied Hub model is pinned at commit
`aa62e643db200fec5997d0b7f66c668bf0ecdac8`. Its published defense recipe differs
from our local selected K16 recipe. This run compares the saved models under
the original attacks; it does not retrain either defense.

Two independent attacks use LR 1e-6 and linear schedules of 128 and 1,000
updates. Each is compared with the completed local run at its own schedule.
Prepared attack inputs, data/optimizer seeds, BF16 settings, and observation
hooks are retained. Source snapshots and runtime configs live under source/
and cases/. The initial snapshot was frozen before training. Later report
changes, if needed, use separate immutable profiles.

Checkpoint verification checks the full downloaded weight hash against the
pinned Hub LFS declaration, parameter shapes against the shared bases, and
tokenization against the original Zephyr tokenizer on all fixed and held-out
probes. Training requires this validation to pass.

The dynamic views reuse the original physical random and attack-reference
bases. The fitted plane is specific to each new attack. A single supplied
defense endpoint replaces a defense history; no missing defense states are
interpolated or inferred. The static view repeats the original eight-radius
recipe, separately from the orthonormal dynamic views.

## Measurements and validation

Both attacks and all 54 declared benchmark task groups are measured. The static
eight-radius comparison, early update segments, and first-update planes are
measured as well. [Findings](findings.md), the [figure index](figure-index.md),
and the [paired report](checkpoint-report.md) give the scientific results.
`completion.json` is written only after all remaining dynamic grids, plots,
benchmark groups, endpoint identities, and source checks pass. It is the
authoritative completion record.

Training has no added defense or recent clean-room intervention. Every observed
attack batch matches its original schedule's control. Axolotl generated a
model-specific prepared-cache directory; every token/label row and source
position was checked against the original prepared data. Different cache
fingerprints do not indicate a changed corpus.

## Source profiles and recoveries

- The initial source snapshot was frozen before execution. `static-v1` repeats
  the original static recipe; its directions and attention convention differ
  from the orthonormal eager-attention dynamics recipe.
- `analysis-v1` uses the unchanged numerical analysis. First-update measurements
  completed, but the comparison adapter assumed a case name absent here.
  `analysis-v2` adds an explicit profile-reference selection and validates the
  existing numerical caches before rendering. The profile entry point and its
  complete local call closure match the older implementation's syntax trees;
  the full module files differ. See `profile-validation-amendment.json` and
  `profile-implementation-validation.json`.
- Initial dynamic and long benchmark allocations reached their 55-minute
  limits. Successors reuse validated completed stages, rows, and raw benchmark
  results. Several earlier debug-queue jobs were canceled while pending and
  replaced by batch-queue allocations; they produced no duplicate measurements.
- The long surface sampler's first resume exited with segmentation fault 139
  while emitting a periodic asynchronous Python stack dump. The cause is not
  established. `analysis-v3` leaves the numerical module byte-identical and
  disables only periodic stack dumps in a bootstrap wrapper. Fatal-fault and
  manual signal reporting remain active. See `analysis-retry-amendment.json`.
  Its pending allocation was shortened from 55 to 35 minutes based on cached
  progress; `allocation-duration-amendment.json` records this scheduler change.
  The original batch sampler began during a debug-queue replacement attempt;
  that replacement exited at the exclusive analysis lock before measuring
  anything. `queue-replacement-attempt.json` records the brief extra allocation.
- `report-v1` requires complete training, evaluation, analysis, and early-profile
  states. It refreshes native figures using the final benchmark CSVs, then
  writes strict paired comparisons. Final coverage requires validated grids,
  physical bases, probes, schedules, endpoints, and source metadata.
- `seal-v4` verifies the complete measurements independently, hashes terminal
  weights, and creates the curated artifact manifest. Large model, optimizer,
  and basis files remain on project storage. `seal-v1` through `seal-v3` are unused
  earlier drafts. Review corrected the record type passed to the final paired
  validator before sealing. Intermediate `renderings-v2`/`renderings-v6` folders
  are not curated.
  The active seal allocation's growing stdout log is excluded from its own
  manifest; completed earlier allocation logs remain included.

Fifty focused test methods passed: ten preparation/provenance tests and forty
comparison tests. The evidence and scope are recorded in `verification.json`.
No environments were changed. All model computation, benchmark evaluation,
plot generation, and bulk weight hashing used `launch_tuo.py` and the original
ROCm workaround flags. Allocations used `verival`; the final analysis retry,
report, and seal use the permitted `guard` fallback after the fair-share check
recorded in `allocation-bank-amendment.json`.
