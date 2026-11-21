# Same-backend CUDA follow-up

## Update: 2026-11-21 06:42 UTC

All eleven phases are submitted, including strict report516719 after benchmarks,
landscapes and profiles. The report worker has been reviewed and explicitly
formatted. Its preflight re-verifies five reused score tables from the raw CUDA
task files, validates eleven frozen input/source hashes, and rejects a changed
training-source hash. Independent strict checkpoint/profile checks passed for
the reused CUDA K4/K256 producers. The report writes only under this new run,
uses fresh CUDA utility references, and records hashes before/after reporting.

First 15-minute poll at06:31: K16 defense completed successfully in11:30;
K64 defense was running. K16 attack has since started. All inspected training
logs confirm one process, four GPUs, torch2.12.1+cu126 and CUDA12.6. No errors.
Watcher2697267 remains active with literal sleep900. Continue until all eleven
phases complete, then inspect final charts, curate final outputs and push them.

The original RIO depth/fidelity study is complete and pushed in commit
`ffab7ad5`. Its final report is
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z/report.md`.

The user then authorized additional correctness/comparability runs until morning.
The full paired K4 backend control showed material score drift, so this follow-up
adds CUDA K16 and K64 defense/attack runs to the already completed CUDA K4 and
K256. All four will share the same numerical source and CUDA training prefix.
The archived K16 also used an older trainer source revision; the new K16 avoids
that source difference within the new K comparison.

Prepared configuration differences are limited to the intended K/inner epochs,
names/output paths and paired defended-model paths. The original fixed probes,
random/reference directions, 125 outer updates and 1,000-update attack schedule
are preserved. Training validation requires the original canonical defense and
attack minibatches. The legacy autocast-cache behavior is preserved and must be
stated in the results; this follow-up changes the backend comparison, not the
defense implementation. `protocol.json` records the intended scope.

Submitted on Matrix using `pbatch/verival`:

| Phase | K16 | K64 |
|---|---:|---:|
| Defense | 516701 | 516702 |
| Attack | 516703 | 516704 |
| Benchmarks | 516707 | 516708 |
| Landscapes | 516709 | 516710 |
| Early-update profiles | 516711 | 516712 |

All jobs passed launcher dry-run inspection. Training uses one process and four
GPUs; benchmarks/profiles use one GPU; the validated landscape worker uses two
explicitly bound GPUs. Scheduler limits alone were shortened after submission
to improve backfill: defense45/75 minutes and attacks60 minutes, with large
margins over measured same-recipe CUDA timings. The amendment and original
120-minute submitted scripts are retained.

The detached read-only watcher is PID2697267 on the current Tuolumne login node.
It runs `ops/watch.py`, then literal `sleep 900`; it stops on an error or after
all eleven expected phases finish. Its initial poll had ten declared jobs,
no errors, and the report not yet submitted. `matrix-jobs.json` is authoritative.
Do not touch unrelated user jobs516696/516699.

The report worker is under `ops/finish_report.py` and is submitted as516719.
Ledger key is `report`, with `status/report.json` as its scientific completion
marker. The report verifies fresh
scores, original strict geometry checks, common physical scales, complete
profiles, unchanged reused inputs, same-backend configuration controls, utility
limits and paired backend/source-revision differences. No writes under completed
producer folders. Then inspect the figures, curate small outputs and commit/push.

Original completed results, frozen sources, models and fitted basis arrays remain
in place. Preserve the user's root AGENTS.md change and unrelated untracked files.
