# TAR compute extension

Follow the [initial reproduction plan](TAR_REPRODUCTION_PLAN.md).

Compare **500 defense updates × 16 simulated attack updates** against
**125 × 64**, using **125 × 16** as the completed baseline. Each extension
uses 8,000 simulated updates. For 64 updates, first use four attack epochs
over the same 16 documents; record data reuse and actual GPU time.

Start fresh from the same base model, preserve the data and other settings,
declare how the learning-rate schedule scales, and save optimizer state.
Compare resistance at matched MMLU capability, targeting at most a five-point
drop. Evaluate the collaborator's 1,000-step relearning configs at `1e-5`,
`1e-6`, and `5e-6`; report the highest recovered WMDP score up to each budget.

Also visualize our completed defense checkpoint with the repository's
[loss-grid computation](examples/tamper-resistance/scripts/analysis/loss-landscapes/get_ll.py)
and [3D plotter](examples/tamper-resistance/scripts/analysis/loss-landscapes/plot_ll_3d.py).

Started 2026-11-10. Landscape shape and attack dynamics are the primary outcomes;
WMDP and MMLU measure recovered capability and retained utility alongside them.
Artifacts: `examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/`.
Live [measurements and figures](examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/report.md) show coverage explicitly.

The defense LR remains `5e-6`. Use linear decay over each defense horizon and
warmup `48/500 = 12/125`; the two 8,000-inner-update conditions then align LR
progress at common inner-update boundaries. The K64 condition reuses its 16
attack examples four times per defense update; the K16 condition receives more
outer batches. Record these exposure differences and actual node/GPU time.

Independent attacks from the shared baseline defense also test:

- The three named attack LRs, holding the 1,000-step schedule fixed.
- The four named Gaussian-noise configs, per-scalar sigma `1e-4`, `1e-3`, `1e-2`,
  and `1e-1`, seed 42, attack LR `1e-5`. Measure the actual noise after BF16 casting
  and evaluate utility before attack.
- Three full-size, with-replacement bootstraps of the common prepared attack
  dataset, seeds 101–103, attack LR `1e-5`. Keep the defender fixed. Preserve
  prepared-row identities, multiplicities, and actual batch hashes; identical
  token sequences can give multiple candidate source rows.

All new attacks use the named 1,000-step linear schedule, whose first 128 steps
differ from the prior 128-step schedule. Use the new clean attacks as controls.
Save dense early model snapshots and optimizer/scheduler/RNG state at selected
milestones. Compare common probes and fixed reference directions across conditions,
plus each condition's fitted attack plane. Keep actual loss and projection-error
checks, shared-scale and center-relative views, and actual minibatch measurements.
Label capability matching only when measured snapshots support it; also report
the fixed-budget utility/resistance tradeoff.

Use first-update-aligned slices and interpolation between the first eight saved
updates to check whether fitting the whole trajectory hides early escape. The
optimizer visits the recorded endpoints; interpolated segments are diagnostic.
