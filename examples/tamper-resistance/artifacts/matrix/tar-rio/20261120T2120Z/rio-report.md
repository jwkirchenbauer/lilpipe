# RIO simulation depth and attacker fidelity

Coverage: **complete**. Every declared measurement is present.

The K comparison holds RIO retain coefficient at 4, inner and outer learning rates at 1e-5, and outer defense updates at 125. K16 reuses our archived RIO defense and attack, including its defense history. All actual attacks use the same 1,000-update schedule. The parity pilot tests the first 16 updates at that archived defended checkpoint. Simulations restart optimizer state at each outer call; they do not form one continuous attack.

[Summary CSV](rio-summary.csv) · [Measured trajectories CSV](rio-trajectories.csv) · [Validation](rio-comparison-validation.json) · [Numerical parity pilot](pilots/rio-prefix-new-anchor/models/attack/parity/validation.json) (**exact**)

## Cases

| Case | Final probe CE | Peak measured WMDP | First ≥45% | First ≥50% | First ≥90% of fresh CUDA base (55.07%) | Initial MMLU | Final MMLU | Initial utility gate (≥53.5798%) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rio-k16 | 5.962 | 44.93 @ 256 | not reached | not reached | not reached | 52.47 | 54.02 | fail |
| rio-k4-lr1e-6 | 3.758 | 48.16 @ 24 | 0 | not reached | not reached | 52.06 | 53.85 | fail |
| rio-k64-lr1e-6 | 3.435 | 47.58 @ 512 | 192 | not reached | not reached | 52.73 | 53.51 | fail |
| rio-k256-lr1e-6 | 3.139 | 46.89 @ 1000 | 512 | not reached | not reached | 46.86 | 53.42 | fail |
| rio-prefix-forward-legacy-lr1e-6 | 4.463 | 38.02 @ 16 | not reached | not reached | not reached | 50.48 | 25.86 | fail |
| rio-prefix-reverse-legacy-lr1e-6 | 2.617 | 46.66 @ 1000 | 2 | not reached | not reached | 53.63 | 54.96 | pass |
| rio-fresh-matched-lr1e-6 | 1.971 | 53.23 @ 1000 | 40 | 48 | 48 | 51.99 | 56.60 | fail |
| rio-prefix-forward-matched-lr1e-6 | 2.022 | 53.80 @ 1000 | 56 | 80 | 80 | 50.33 | 57.15 | fail |
| rio-prefix-reverse-matched-lr1e-6 | 1.994 | 53.92 @ 768 | 24 | 40 | 40 | 53.89 | 56.90 | pass |

## Figures

The four-way K comparison and data/optimizer factorial overlay use common measured frames and shared physical/color bounds. Other cases have paired K16 plots. Trajectories use the union of actual-loss and evaluation steps; blank cells mean that metric was not measured at that step. No interpolation is implied.

- rio-k16--rio-k4-lr1e-6 local-reference: [rio-k16--rio-k4-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-k4-lr1e-6-local-reference.png)
- rio-k16--rio-k4-lr1e-6 local-delta: [rio-k16--rio-k4-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-k4-lr1e-6-local-delta.png)
- rio-k16--rio-k4-lr1e-6 broad-random: [rio-k16--rio-k4-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-k4-lr1e-6-broad-random.png)
- rio-k16--rio-k4-lr1e-6 broad-delta: [rio-k16--rio-k4-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-k4-lr1e-6-broad-delta.png)
- rio-k16--rio-k4-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-k4-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-k4-lr1e-6-trajectories.png)
- rio-k16--rio-k64-lr1e-6 local-reference: [rio-k16--rio-k64-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-k64-lr1e-6-local-reference.png)
- rio-k16--rio-k64-lr1e-6 local-delta: [rio-k16--rio-k64-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-k64-lr1e-6-local-delta.png)
- rio-k16--rio-k64-lr1e-6 broad-random: [rio-k16--rio-k64-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-k64-lr1e-6-broad-random.png)
- rio-k16--rio-k64-lr1e-6 broad-delta: [rio-k16--rio-k64-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-k64-lr1e-6-broad-delta.png)
- rio-k16--rio-k64-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-k64-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-k64-lr1e-6-trajectories.png)
- rio-k16--rio-k256-lr1e-6 local-reference: [rio-k16--rio-k256-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-k256-lr1e-6-local-reference.png)
- rio-k16--rio-k256-lr1e-6 local-delta: [rio-k16--rio-k256-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-k256-lr1e-6-local-delta.png)
- rio-k16--rio-k256-lr1e-6 broad-random: [rio-k16--rio-k256-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-k256-lr1e-6-broad-random.png)
- rio-k16--rio-k256-lr1e-6 broad-delta: [rio-k16--rio-k256-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-k256-lr1e-6-broad-delta.png)
- rio-k16--rio-k256-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-k256-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-k256-lr1e-6-trajectories.png)
- rio-k16--rio-prefix-forward-legacy-lr1e-6 local-reference: [rio-k16--rio-prefix-forward-legacy-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-prefix-forward-legacy-lr1e-6-local-reference.png)
- rio-k16--rio-prefix-forward-legacy-lr1e-6 local-delta: [rio-k16--rio-prefix-forward-legacy-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-prefix-forward-legacy-lr1e-6-local-delta.png)
- rio-k16--rio-prefix-forward-legacy-lr1e-6 broad-random: [rio-k16--rio-prefix-forward-legacy-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-prefix-forward-legacy-lr1e-6-broad-random.png)
- rio-k16--rio-prefix-forward-legacy-lr1e-6 broad-delta: [rio-k16--rio-prefix-forward-legacy-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-prefix-forward-legacy-lr1e-6-broad-delta.png)
- rio-k16--rio-prefix-forward-legacy-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-prefix-forward-legacy-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-prefix-forward-legacy-lr1e-6-trajectories.png)
- rio-k16--rio-prefix-reverse-legacy-lr1e-6 local-reference: [rio-k16--rio-prefix-reverse-legacy-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-prefix-reverse-legacy-lr1e-6-local-reference.png)
- rio-k16--rio-prefix-reverse-legacy-lr1e-6 local-delta: [rio-k16--rio-prefix-reverse-legacy-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-prefix-reverse-legacy-lr1e-6-local-delta.png)
- rio-k16--rio-prefix-reverse-legacy-lr1e-6 broad-random: [rio-k16--rio-prefix-reverse-legacy-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-prefix-reverse-legacy-lr1e-6-broad-random.png)
- rio-k16--rio-prefix-reverse-legacy-lr1e-6 broad-delta: [rio-k16--rio-prefix-reverse-legacy-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-prefix-reverse-legacy-lr1e-6-broad-delta.png)
- rio-k16--rio-prefix-reverse-legacy-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-prefix-reverse-legacy-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-prefix-reverse-legacy-lr1e-6-trajectories.png)
- rio-k16--rio-fresh-matched-lr1e-6 local-reference: [rio-k16--rio-fresh-matched-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-fresh-matched-lr1e-6-local-reference.png)
- rio-k16--rio-fresh-matched-lr1e-6 local-delta: [rio-k16--rio-fresh-matched-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-fresh-matched-lr1e-6-local-delta.png)
- rio-k16--rio-fresh-matched-lr1e-6 broad-random: [rio-k16--rio-fresh-matched-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-fresh-matched-lr1e-6-broad-random.png)
- rio-k16--rio-fresh-matched-lr1e-6 broad-delta: [rio-k16--rio-fresh-matched-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-fresh-matched-lr1e-6-broad-delta.png)
- rio-k16--rio-fresh-matched-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-fresh-matched-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-fresh-matched-lr1e-6-trajectories.png)
- rio-k16--rio-prefix-forward-matched-lr1e-6 local-reference: [rio-k16--rio-prefix-forward-matched-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-prefix-forward-matched-lr1e-6-local-reference.png)
- rio-k16--rio-prefix-forward-matched-lr1e-6 local-delta: [rio-k16--rio-prefix-forward-matched-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-prefix-forward-matched-lr1e-6-local-delta.png)
- rio-k16--rio-prefix-forward-matched-lr1e-6 broad-random: [rio-k16--rio-prefix-forward-matched-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-prefix-forward-matched-lr1e-6-broad-random.png)
- rio-k16--rio-prefix-forward-matched-lr1e-6 broad-delta: [rio-k16--rio-prefix-forward-matched-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-prefix-forward-matched-lr1e-6-broad-delta.png)
- rio-k16--rio-prefix-forward-matched-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-prefix-forward-matched-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-prefix-forward-matched-lr1e-6-trajectories.png)
- rio-k16--rio-prefix-reverse-matched-lr1e-6 local-reference: [rio-k16--rio-prefix-reverse-matched-lr1e-6-local-reference.png](rio-figures/rio-k16--rio-prefix-reverse-matched-lr1e-6-local-reference.png)
- rio-k16--rio-prefix-reverse-matched-lr1e-6 local-delta: [rio-k16--rio-prefix-reverse-matched-lr1e-6-local-delta.png](rio-figures/rio-k16--rio-prefix-reverse-matched-lr1e-6-local-delta.png)
- rio-k16--rio-prefix-reverse-matched-lr1e-6 broad-random: [rio-k16--rio-prefix-reverse-matched-lr1e-6-broad-random.png](rio-figures/rio-k16--rio-prefix-reverse-matched-lr1e-6-broad-random.png)
- rio-k16--rio-prefix-reverse-matched-lr1e-6 broad-delta: [rio-k16--rio-prefix-reverse-matched-lr1e-6-broad-delta.png](rio-figures/rio-k16--rio-prefix-reverse-matched-lr1e-6-broad-delta.png)
- rio-k16--rio-prefix-reverse-matched-lr1e-6 actual/projection/capability trajectories: [rio-k16--rio-prefix-reverse-matched-lr1e-6-trajectories.png](rio-figures/rio-k16--rio-prefix-reverse-matched-lr1e-6-trajectories.png)
- rio-k-four-way local-reference: [rio-k-four-way-local-reference.png](rio-figures/rio-k-four-way-local-reference.png)
- rio-k-four-way local-delta: [rio-k-four-way-local-delta.png](rio-figures/rio-k-four-way-local-delta.png)
- rio-k-four-way broad-random: [rio-k-four-way-broad-random.png](rio-figures/rio-k-four-way-broad-random.png)
- rio-k-four-way broad-delta: [rio-k-four-way-broad-delta.png](rio-figures/rio-k-four-way-broad-delta.png)
- rio-k-four-way actual/projection/capability trajectories: [rio-k-four-way-trajectories.png](rio-figures/rio-k-four-way-trajectories.png)
- rio-data-optimizer-2x2 local-reference: [rio-data-optimizer-2x2-local-reference.png](rio-figures/rio-data-optimizer-2x2-local-reference.png)
- rio-data-optimizer-2x2 local-delta: [rio-data-optimizer-2x2-local-delta.png](rio-figures/rio-data-optimizer-2x2-local-delta.png)
- rio-data-optimizer-2x2 broad-random: [rio-data-optimizer-2x2-broad-random.png](rio-figures/rio-data-optimizer-2x2-broad-random.png)
- rio-data-optimizer-2x2 broad-delta: [rio-data-optimizer-2x2-broad-delta.png](rio-figures/rio-data-optimizer-2x2-broad-delta.png)
- rio-data-optimizer-2x2 actual/projection/capability trajectories: [rio-data-optimizer-2x2-trajectories.png](rio-figures/rio-data-optimizer-2x2-trajectories.png)

## Data and compute provenance

| Case | K | Simulation data | Simulation update protocol | Defense inner updates | Defense supervised tokens | Defense training seconds | Wall / allocated GPU hours |
|---|---:|---|---|---:|---:|---:|---:|
| rio-k4-lr1e-6 | 4 | fresh | legacy | 500 | 255500 | 821.6 | 0.297 / 1.189 |
| rio-k64-lr1e-6 | 64 | fresh | legacy | 8000 | 4088000 | 2745.0 | 0.819 / 3.278 |
| rio-k256-lr1e-6 | 256 | fresh | legacy | 32000 | 16352000 | 3681.1 | 1.073 / 4.290 |
| rio-prefix-forward-legacy-lr1e-6 | 16 | forward | legacy | 2000 | 1022000 | 1168.1 | 0.388 / 1.553 |
| rio-prefix-reverse-legacy-lr1e-6 | 16 | reverse | legacy | 2000 | 1022000 | 1287.8 | 0.424 / 1.694 |
| rio-fresh-matched-lr1e-6 | 16 | fresh | matched | 2000 | 1022000 | 1292.4 | 0.417 / 1.667 |
| rio-prefix-forward-matched-lr1e-6 | 16 | forward | matched | 2000 | 1022000 | 1302.7 | 0.430 / 1.719 |
| rio-prefix-reverse-matched-lr1e-6 | 16 | reverse | matched | 2000 | 1022000 | 1308.9 | 0.421 / 1.686 |

- rio-k16: [early-update-profiles](control-study/cases/hparam-rio-lr1e-6/early-update-profiles.png)
- rio-k16: [first-update-landscape](control-study/cases/hparam-rio-lr1e-6/first-update-landscape.png)
- rio-k4-lr1e-6: [early-update-profiles](cases/rio-k4-lr1e-6/early-update-profiles.png)
- rio-k4-lr1e-6: [first-update-landscape](cases/rio-k4-lr1e-6/first-update-landscape.png)
- rio-k64-lr1e-6: [early-update-profiles](cases/rio-k64-lr1e-6/early-update-profiles.png)
- rio-k64-lr1e-6: [first-update-landscape](cases/rio-k64-lr1e-6/first-update-landscape.png)
- rio-k256-lr1e-6: [early-update-profiles](cases/rio-k256-lr1e-6/early-update-profiles.png)
- rio-k256-lr1e-6: [first-update-landscape](cases/rio-k256-lr1e-6/first-update-landscape.png)
- rio-prefix-forward-legacy-lr1e-6: [early-update-profiles](cases/rio-prefix-forward-legacy-lr1e-6/early-update-profiles.png)
- rio-prefix-forward-legacy-lr1e-6: [first-update-landscape](cases/rio-prefix-forward-legacy-lr1e-6/first-update-landscape.png)
- rio-prefix-reverse-legacy-lr1e-6: [early-update-profiles](cases/rio-prefix-reverse-legacy-lr1e-6/early-update-profiles.png)
- rio-prefix-reverse-legacy-lr1e-6: [first-update-landscape](cases/rio-prefix-reverse-legacy-lr1e-6/first-update-landscape.png)
- rio-fresh-matched-lr1e-6: [early-update-profiles](cases/rio-fresh-matched-lr1e-6/early-update-profiles.png)
- rio-fresh-matched-lr1e-6: [first-update-landscape](cases/rio-fresh-matched-lr1e-6/first-update-landscape.png)
- rio-prefix-forward-matched-lr1e-6: [early-update-profiles](cases/rio-prefix-forward-matched-lr1e-6/early-update-profiles.png)
- rio-prefix-forward-matched-lr1e-6: [first-update-landscape](cases/rio-prefix-forward-matched-lr1e-6/first-update-landscape.png)
- rio-prefix-reverse-matched-lr1e-6: [early-update-profiles](cases/rio-prefix-reverse-matched-lr1e-6/early-update-profiles.png)
- rio-prefix-reverse-matched-lr1e-6: [first-update-landscape](cases/rio-prefix-reverse-matched-lr1e-6/first-update-landscape.png)

## Scope

The control and each case are checked for declared checkpoints, shared fixed probes, matching corpus revision, saved shared directions, complete evaluation frames, paired defense status, and the replay-inner audit. Missing measurements stop this final report. Fresh-original versus exact-prefix simulation changes padding and target-label construction as well as example source, so it cannot be attributed only to source identity. Forward versus reverse replays use the same fixed minibatch pool and token budget, isolating order. Per-case generated figures and the source study record are the provenance for projected CE and landscapes. No full parity or continuous-state claim follows from these finite observations.
The K sweep and reused legacy K16 condition retain the original enclosing AMP autocast behavior; that source does not refresh cached parameter casts after each inner optimizer update. The v1 real-model parity pilot with that cache behavior matched update 1 and diverged at later checked states. The v2 per-update cache refresh matched the five checked parameter states and all 16 losses at its tested anchor. Thus K results describe the frozen legacy simulator conditions and are not numerical parity claims about actual attacker trajectories.

## Backend and numerical scope

All score tables in this report are freshly re-collected from verified Matrix CUDA 12.6 evaluations. The fresh base Zephyr WMDP and MMLU references and the defended K16 checkpoint-zero MMLU reference are also CUDA results; archived ROCm scores are not substituted for CUDA scores.

The base Zephyr reference contains 868 verified WMDP examples across 6 subtasks and 13,322 verified MMLU examples across 53 subtasks. Base scores are WMDP 55.069124% and MMLU 58.579793%. The fresh CUDA K16 defended checkpoint-zero MMLU reference is 52.469599%.

Within the main eight-condition study, K256 defense training used CUDA. K4, K64, and the fidelity-condition training checkpoints were reused from ROCm; their evaluations were rerun on CUDA. Floating-point variation between training backends is expected. A separately trained full CUDA K4 defense and 1,000-step attack are isolated under `backend-control`; attack step 0 evaluates the corresponding defended endpoint. The score comparison estimates backend drift under the same K4 recipe, fixed seeds, sampled-batch protocol, probes, and schedule.

The same-backend CUDA simulated-versus-genuine parity result is exact by the predeclared step and loss checks. Archived AMD/ROCm comparisons remain diagnostics and do not gate this report.

[Fresh base validation](control-study/base-reference/validation.json) · [Same-backend parity](pilots/rio-prefix-new-anchor/models/attack/parity/validation.json) · [Precision audit](source/precision-audit/results/precision-audit.json) · [K4 backend score deltas](backend-control-drift.csv)

### K4 backend-control score differences

These are percentage-point differences between the ROCm-trained main K4 attack and the CUDA-trained K4 attack, both evaluated with verified CUDA task files. Step 0 is the defended endpoint included in each attack evaluation; later rows are attack checkpoints.

| Task | Common checkpoints | Largest absolute difference | Difference at attack step 1000 |
|---|---:|---:|---:|
| wmdp_bio_robust | 27 | 5.760369 pp | +4.493088 pp |
| mmlu_no_bio | 4 | 2.980033 pp | +2.326978 pp |

### Same-backend CUDA parity at the new matched anchor

The matched-cache-v2 simulated and genuine CUDA runs had exactly equal losses for all 16 prefix updates and exactly equal parameter states at updates 1, 2, 4, 8, and 16 (maximum absolute loss error 0). This is a strict same-backend result at the new forward-matched anchor; it does not establish parity for every training-time simulation or change the legacy K-sweep AMP-cache behavior described above.
Archived ROCm comparisons are diagnostic only. Their maximum reported relative update error across the five states is 0.12733079; cross-backend bitwise equality is not required.

### Checkpoint reload and parameter precision

The audit's `native_to_reload` measures loaded anchor minus the exact native file supplied to the loader. Its `native_bf16_roundtrip` measures loaded anchor minus native values cast through BF16 and back. A zero roundtrip residual proves that the reload is explained by this cast; it does not mean the native parameters were unchanged.

| Anchor | Reload L2 | Changed coordinates | Loaded minus BF16-roundtrip L2 | Reload / first optimizer update L2 | Step-16 straight-through relative error |
|---|---:|---:|---:|---:|---:|
| archived_rio_v2 (archived-rocm) | 0.1006617 | 188,195,817 | 0 | 2.2897 | 0.0002634 |
| forward_matched_new_anchor (fresh-cuda) | 0.09494724 | 188,178,244 | 0 | 2.6544 | 0.0002979 |

A raw displacement from a native defense file to a saved attack checkpoint includes this reload offset. The optimizer update is measured from the loaded anchor. This distinction matters for interpreting early trajectory directions and coordinate concentration. The straight-through audit checks saved-dtype `anchor + (endpoint - anchor)` arithmetic, not the defender's complete training-time simulation. No forward pass is performed here, so these norms alone do not establish a CE change or explain robustness.

The reused K4 and K16 geometry retains its original producer provenance. Only the capability-score output path and CPU anchor storage option differ from that saved geometry protocol; neither enters geometry computation. All original protocol hashes, input identities, progress accumulators and other settings remain checked. See [geometry reuse validation](geometry-reuse-validation.json).
