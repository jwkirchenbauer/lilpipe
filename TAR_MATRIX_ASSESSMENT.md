# Matrix assessment, 2026-11-20

Matrix is now running production evaluation and landscape work for the
[RIO study](TAR_RIO_K_AND_FIDELITY.md). Following user authorization and a
successful training pilot, the unfinished K256 defense and attack have also
been submitted there. All requests use `pbatch/verival`.
The detached monitor on `tuolumne1004` polls both clusters every 900 seconds.
The active run is
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/matrix/tar-rio/20261120T2120Z`;
its `matrix-jobs.json` and `cancellations.json` record replacements.

Using `ssh matrix` and the documented Matrix launcher, the CUDA/BF16 probe
passed on an H100. Its measured queue wait was one second. The four-Slurm-process
NCCL all-reduce also passed after explicit device localization, with zero
seconds of recorded queue wait. These use the existing starter environment,
not the TAR runtime. Jobs were 515334 and 515394, respectively.

At 18:40 UTC, Slurm's `sbatch --test-only` predicted starts at 19:31 UTC for
each of these independent requests under `verival`:

| Request | Duration | Host memory | Estimated wait |
|---|---|---|---|
| One H100, 28 CPU cores | 3 hours | 256 GiB | 51 minutes |
| One H100, 28 CPU cores | 3 hours | All node memory | 51 minutes |
| Four H100s, 112 CPU cores, exclusive node | 4 hours | All node memory | 51 minutes |

These are independent forecasts. A subsequent real 45-minute, one-H100 request
started in the same second it was submitted and finished in 4 minutes 41 seconds
(job 515852, 20:44:25–20:49:06 UTC). The queue evidence favors Matrix at these
observation times; whole-study throughput remains unmeasured.

The necessary adaptations are concrete:

- Use `--cuda_module=`: the launcher's default `cuda/12.6.0` module is absent.
- The site supplies a one-device mask even for an unbound four-GPU step.
  The successful probe verified that the step owned all four GPUs, exposed
  that allocated set, and selected devices by local rank. The two earlier
  multi-process probes failed before this correction; preserve their logs.
- Use the dedicated CUDA evaluation prefix. The resolved 123-package lock
  preserves all 102 common non-Torch package versions; Torch retains public
  version 2.12.1 with a CUDA 12.6 build. The starter lacks TAR dependencies.
  Following explicit user authorization, the maintainer's fresh-prefix
  Micromamba+uv workflow installed it successfully on 2026-11-20.
- The unchanged landscape worker exceeds H100 memory: its model, saved center,
  and two FP32 directions alone require 80.93 GiB versus 79.22 GiB available.
  The optional `cpu_landscape_anchor` setting keeps saved centers on the CPU
  and copies one tensor at a time during perturbation. Eleven CPU tests pass,
  including checks of the original perturbation arithmetic and rounding.
  The real H100 checkpoint comparison below now verifies this storage change
  and measures the memory requirement.
- Replace the frozen workers' Flux/environment preflight with explicit
  Slurm/CUDA checks. Preserve model, data, saved directions, physical radii,
  and output validation; keep separate caches during the comparison.

The checkpoint probe passed at all nine end-of-defense random-plane points
from the completed RIO K16 control. It used the same saved directions, radius
125, checkpoint, and fixed documents. Its predeclared tolerance stayed at
0.01 CE nats per point. The largest CUDA-versus-ROCm difference was **0.0052534
nats**; the center difference was 0.0003815 nats. CPU-center versus GPU-center
perturbations were bitwise equal in the CUDA arithmetic check.

Peak PyTorch allocation was **69.43 GiB**, reserved GPU memory **70.02 GiB**,
and process host memory **69.96 GiB**. The measurement took 270.8 seconds.
This validates nine coarse random-plane points; the full moving/fitted-plane
panels and early-update profiles still need a production validation.

A separate full 868-example WMDP evaluation completed as job 515883, using
the original four-process Accelerate command, batch size 32, zero-shot settings,
and the same frozen task definitions. It also started immediately and finished
in 2 minutes 19 seconds (20:53:54–20:56:13 UTC).

| Same defended checkpoint | WMDP accuracy |
|---|---:|
| Archived Tuolumne evaluation | 36.2903% |
| Matrix evaluation | 36.8664% |
| Difference | +0.5760 percentage points |

That is a net difference of five correct answers, not a count of changed
predictions. Full sample counts matched, with no rejected Matrix results.
All nine frozen task-source files matched. The archived reference lacks
per-sample input hashes, so that identity check is unavailable. These checks
do not isolate the backend as the sole cause of the score difference.

The dedicated CUDA training lock preserves the 222 common non-Torch package
versions from the ROCm training environment. A short training pilot (516076)
completed both a four-update K4 prefix and one full K256 outer update, keeping
the production 125-update schedule, warmup, data, and optimizer settings.
Initial weights were bitwise equal and outer batches/rates matched exactly.
The K4 update cosine was **0.9931**, relative update error **0.1174**, and maximum
loss difference **0.04494 nats**, within the predeclared migration limits.
This establishes approximate agreement over a short prefix. A full K4 CUDA
defense and standard attack are submitted as a separate backend control to
measure accumulated drift.

The K256 update took 31.1 seconds, suggesting roughly 65 minutes for 125 updates
before checkpoint I/O. Its production defense requests three hours. The old
queued K256 defense and attack were canceled before submitting replacements
516131 and 516136, preventing duplicate writers. The full CUDA K4 control jobs
are 516132 and 516137. Original completed ROCm training remains an input to the
study, so K256-versus-other-K comparisons must report the training backend and
consider the paired K4 backend control.

Production wrappers now explicitly support Slurm/CUDA. K4 full landscape and
profile validation remains the gate for broad analysis migration. Saved centers
and early-profile vectors can reside on the CPU while retaining the original
arithmetic and endpoint checks. Fresh CUDA benchmark results, including the
reused K16 control, share one measurement backend; archived ROCm scores are
kept as provenance. A one-GPU evaluation pilot tests whether partially occupied
nodes can reduce queue time and GPU usage.

Runtime manifests:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/runtime/matrix`.
Probe code, accounting, failed and successful logs, and prepared installer:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-assessment`.
Launcher logs are in the same run's `launches/matrix-assessment/logs` directory.

Checkpoint result and pointwise comparison:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z/matrix-assessment/checkpoint-probe/result.json`
and `comparison.csv` in that directory.
Benchmark comparison:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z/matrix-assessment/benchmark-probe/result.json`.
