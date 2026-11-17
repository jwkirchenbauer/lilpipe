# Execution notes

The prerequisite attack/defense basis analysis completed before training submission. Its report and sealed evidence are committed in `3ff4a0a4`. Benchmark scope selection is committed in `9054bb07`; it includes only the four attack cases, with each defense endpoint evaluated at attack step zero.

Scientific inputs, canonical configs, and numerical sources are frozen. `preparation-validation.json` confirms byte-identical trainer/loss/sampler source against B, matching prepared-cache inodes, and only the three declared hyperparameter changes. `evaluation-scope-amendment.json` records the analysis-only scope amendment. Operational wrappers independently require passed training and exact batch-order validation.

The first four defense requests were queued for 120 minutes in pbatch. After confirming that all remained queued, they were canceled without running; dependent attacks/evaluations were canceled by their scheduler dependencies. Their original job JSON files remain as execution history. The recorded original defense runtime was 1087 seconds, so the replacements request 55-minute pdebug allocations. The original static radius measurement took 403 seconds; its new allocations request 30 minutes. These changes affect scheduling only.

Current job records:

- `defense-jobs-v2.json`: four active defense requests.
- `attack-jobs-v2.json`: four fresh 1000-step attacks, each dependent on its defense.
- `operational-jobs-v2.json`: four benchmark jobs dependent on attacks, plus the R first-update profiles. Evaluation keeps its 180-minute pbatch allowance: the prior 1000-step checkpoint study needed about 66 minutes across its initial and resumed benchmark jobs.
- `static-jobs.json`: four original-convention static radius measurements dependent on defenses.

At 2026-11-16 23:42 UTC, R was training and saving defense checkpoint 8. The startup diagnostic confirmed one process and four GPUs, with 1,875,968,000 / 1,744,896,000 / 1,744,896,000 / 1,875,972,096 parameters assigned to logical GPUs 0–3. The other defenses were queued. Poll at roughly 15-minute intervals and inspect status files before scheduling follow-on stages.

Remaining submissions are intentionally gated by actual completion, because the launcher accepts only one scheduler dependency URI. Use the reviewed jobs in `operational-dryrun-plan.json` for `landscapes-r-i`, `landscapes-o-rio`, `profiles-i-o`, `profiles-rio`, and `checkpoint-report`, after all listed prerequisites pass. The replacement R profile job is in `operational-jobs-v2.json`; do not use the canceled first-attempt ID. Frozen wrappers enforce the same prerequisites even if a scheduler dependency reports success incorrectly.

Run the original-static comparison after all four static measurements pass, using `source/profiles/static-v1/run.sh report RUN`. Run `source/comparison-v1/run.sh RUN` after native analysis, attack benchmarks, and first-update profiles complete for every condition. The aggregate helper has eight passing fixture tests and a successful real B cache preflight (`logs/control-cache-preflight.log`); it requires no intermediate defense benchmark files. It will create `aggregate-comparison-v1/final` and refuses to overwrite existing output.

Before final publication, inspect all matched-scale figures, write measured findings and a figure index, validate coverage/centers/projection identities, seal curated small artifacts, and commit the completed study. Preserve the old sealed control and collaborator artifacts.
