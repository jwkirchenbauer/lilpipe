# Execution record

All model training, attacks, evaluations, and landscape computations are complete.

The study contains eight 125-call defense continuations, 36 attack trajectories,
162 regular benchmark points, nine default local searches, and one adaptive
fine-grid search. Nine canonical geometries contain five planes each; the
stronger new-data attacker has two additional planes with three probe cohorts.
There are 3,807 grid records. Findings and the figure index explain the scientific
outcomes; complete execution does not imply that every objective, utility gate,
or numerical-fidelity criterion was satisfied.

The final CPU/plot suite passed 71 methods, with two GPU methods skipped.
Both were checked separately on allocated GPUs: geometry cache parity
(job f3jQhgUN1L79) and cross-device scalar gradient reductions
(job f3jRb5csBrCF). Black and diff checks passed. final-tests.log and the copied
GPU logs preserve the verification; completion.json records final coverage.

## Reproduction sources

| Profile | Purpose |
| --- | --- |
| train-v2 | Original six continuations and their attacks |
| train-v3 | Refreshed-retain continuations and their attacks |
| analysis-v9 | Final benchmark collection and evidence report |
| geometry-v2 | Canonical native-FP32 grids |
| geometry-plot-v4 | Canonical plots |
| geometry-variant-v1 | Strong new-data attacker and its actual loss cohort |
| local-fine-v1 | Separate adaptive fine-grid diagnostic |
| final-plots-v1 and final-plots-v2 | Shared comparison ranges and exports |
| seal-v3 | Coverage validation, hashes, and curated file selection |

Every executing profile is preserved. The immutable original study.json,
data.json, source archive, and source-manifest.json remain unchanged.
case-extensions.json, retain-refresh.json, and the dated amendments declare
the followups. The default local-search results remain unchanged; the adaptive
case writes local-search-fine.

All meaningful computation used launch_tuo.py with the verival bank. Model
work used one process and four explicitly assigned GPUs, with module-device
checks and FP32 parameters/forwards. ROCm flags disabled hipBLASLt preference
and COMGR caching. Training prefix:
 /usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4
Evaluation/plotting prefix:
 /usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2
No environments were created or mutated for this study.

## Final analysis details

The useful entropy continuation passes native utility at MMLU 54.1435%.
At attack update four, its matched training-data attack reaches WMDP 26.7281%,
while the same settings on the other eight examples reach 51.9585%.
Those examples were excluded from this continuation; pretraining and original
K16 training use are not ruled out.

Canonical figure limits are CE 0–82. The utility-passing group and the
additional attacker view share CE 0–32. Reference axes remain literally shared;
first-update/fitted axes depend on the recorded attack. Control geometry
snapshots used as reference inputs are archived by SHA under revisions/.
The current study uses new FP32 directions and probes, so absolute appearances
are not directly interchangeable with the earlier BF16 plots.

The initial streaming geometry timed out and is archived. A source-archive
initialization race was fixed before rerunning the affected benchmark.
Superseded queued jobs were canceled before execution and recorded in the job
registries. All declared scientific outputs were subsequently completed.

Large checkpoint weights, optimizer states, and runtime caches remain under
the project artifact directory. The curated Git result contains sources,
declarations, raw score/probe records, figures, and checksums. Verify
artifact-manifest.sha256 from examples/tamper-resistance. The manifest excludes
itself; its Git commit supplies the final content address.

Earlier method commits are 5fff2c55 and 7f337304. The results commit is available
in Git history and the final response. Original reproduction and compute-study
artifacts are unchanged.
