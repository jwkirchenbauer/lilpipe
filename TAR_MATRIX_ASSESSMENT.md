# Matrix assessment, 2026-11-20

Matrix looks promising for finishing the evaluations and landscape measurements
in the [RIO study](TAR_RIO_K_AND_FIDELITY.md). It is not a drop-in replacement.
The Tuolumne monitor has resumed on `tuolumne1004`, polling every 900 seconds.
At 18:49 UTC, all 26 production allocations remained pending. No production
requests have been canceled or moved; all work remains in `pbatch`.

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

These are forecasts, not submitted jobs or throughput measurements. They are
encouraging beside the roughly day-long Tuolumne wait, but do not establish
when the entire study would finish on Matrix.

The necessary adaptations are concrete:

- Use `--cuda_module=`: the launcher's default `cuda/12.6.0` module is absent.
- The site supplies a one-device mask even for an unbound four-GPU step.
  The successful probe verified that the step owned all four GPUs, exposed
  that allocated set, and selected devices by local rank. The two earlier
  multi-process probes failed before this correction; preserve their logs.
- Prepare a dedicated CUDA evaluation prefix. The resolved 123-package lock
  preserves all 102 common non-Torch package versions; Torch retains public
  version 2.12.1 with a CUDA 12.6 build. The starter lacks TAR dependencies.
  Environment creation is awaiting the explicit authorization required by
  the root AGENTS.md.
- The unchanged landscape worker exceeds H100 memory: its model, saved center,
  and two FP32 directions alone require 80.93 GiB versus 79.22 GiB available.
  The optional `cpu_landscape_anchor` setting keeps saved centers on the CPU
  and copies one tensor at a time during perturbation. Eleven CPU tests pass,
  including checks of the original perturbation arithmetic and rounding.
  Actual H100 model memory and numerical agreement remain unmeasured.
- Replace the frozen workers' Flux/environment preflight with explicit
  Slurm/CUDA checks. Preserve model, data, saved directions, physical radii,
  and output validation; keep separate caches during the comparison.

The prepared checkpoint probe will compare nine end-of-defense random-plane
points with the completed RIO K16 control, using identical directions and data.
Its predeclared tolerance is 0.01 CE nats at every point. It also checks
CPU-center versus GPU-center perturbations bitwise on CUDA and measures peak
GPU and host memory. It has not been submitted.

If that check passes, migrate evaluation and landscape work first, without
duplicate writers or bypassing the K4 validation gates. Keep training and exact
replay audits on Tuolumne: moving only some training conditions to CUDA would
add a backend difference to the scientific comparisons.

Runtime manifests:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/runtime/matrix`.
Probe code, accounting, failed and successful logs, and prepared installer:
`/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z/source/matrix-assessment`.
Launcher logs are in the same run's `launches/matrix-assessment/logs` directory.
