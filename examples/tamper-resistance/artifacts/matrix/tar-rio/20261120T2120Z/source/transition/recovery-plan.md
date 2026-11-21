# RIO analysis/profile/report recovery DAG

Scope: recovery after the two-GPU peer pilot passed and the first production
peer analysis attempt was canceled by incorrect GPU binding. This is a planning
note only; it submits no jobs and does not modify the ledger or workers.

## Required analysis coverage

The report worker requires ten new peer analyses:

- Main study (8): `rio-k4-lr1e-6`, `rio-k64-lr1e-6`,
  `rio-k256-lr1e-6`, `rio-prefix-forward-legacy-lr1e-6`,
  `rio-prefix-reverse-legacy-lr1e-6`, `rio-fresh-matched-lr1e-6`,
  `rio-prefix-forward-matched-lr1e-6`, and
  `rio-prefix-reverse-matched-lr1e-6` under `$RUN/cases/`.
- K16 control (1): `hparam-rio-lr1e-6` under
  `$RUN/control-study/cases/`.
- CUDA K4 backend control (1): `rio-k4-cuda-lr1e-6` under
  `$RUN/backend-control/cases/`.

Submit the main K4 analysis first with `submit_peer.py --attempt peer-v2`.
Queue the other nine analyses with `--attempt peer-v2` and
`--dependency <new-K4-peer-job-id>`. The afterok dependency prevents them from
running until K4 completes its full analysis and validation. Inspect the peer
job's detailed `status/analysis-peer-<jobid>.json` and geometry/surface outputs.
Do not use any canceled peer or closure job ID as a dependency.

## Required profile coverage

K4 main profile output for `rio-k4-lr1e-6` is already complete (job 516150).
After confirming its `status/step-profiles.json` and reference outputs remain
complete, schedule the other nine profile outputs with
`launch.py profiles CASE --attempt retry-v2 --run RUN --dependency <own-new-peer-analysis-id>`:

- Seven remaining main attacks: K64, K256, forward/reverse legacy,
  fresh matched, and forward/reverse matched.
- K16 control: `hparam-rio-lr1e-6` in `control-study`.
- CUDA K4 backend control: `rio-k4-cuda-lr1e-6` in `backend-control`.

Use `--run "$RUN"`, `--run "$RUN/control-study"`, or
`--run "$RUN/backend-control"` as appropriate. Each profile depends on its
own fresh peer analysis. The verified completed K4 profile is an input/reference
gate, not a dependency to resubmit or to attach to every new profile job.

The new attempt suffix creates separate `profiles:CASE:retry-v2` ledger keys,
launch groups, and logs while retaining historical canceled rows. Peer analyses
use `analysis:CASE:peer-v2` keys.

## Final report gate

Queue `launch.py report final --attempt retry-v2` behind all required
new analyses and nine retry profiles. Its worker requires their case status
files to read `complete`. Also confirm every other report preflight input remains valid:
all main training/order validations, full verified CUDA evaluations for the
eight main attacks and K16/backend-control studies, the completed K4 profile,
fresh base and defended references, exact same-backend parity, complete
precision audit, and backend-control training/evaluation.

The candidate report dependency set is the ten peer-v2 jobs plus nine retry-v2
profile jobs. At submission time, include only members still live in Slurm;
verify completed jobs from their status markers and omit them because Slurm may
have purged their controller records. Add benchmark 516146 only if it is still
running. Omit all canceled/superseded IDs, especially 516219 and 516227. The
report worker rechecks scientific states and writes
`$RUN/report-matrix-references.json` only after all preflights and strict
comparators pass.

## Frozen report-source findings

`source/report-matrix/finish_report.py` checks the eight main attack rows plus
the K16 and CUDA K4 auxiliary attacks. For each attack it requires complete
training, evaluation, analysis, and step-profile status; training/order and
full-horizon contracts are independently checked. It also verifies score-table
coverage, fresh CUDA references, exact simulated-versus-genuine parity, and the
precision audit. It makes no scheduler-ID or ledger lookups.

`source/report-matrix/run_report.sh` invokes the pinned report worker directly.
The old `source/report-matrix/submission-plan.json` records canceled job 516219
and the former dependency on 516227, but neither the report shell worker nor
`finish_report.py` reads that file. Keep it as historical evidence; construct
the replacement dependency list from currently valid jobs and case markers.
