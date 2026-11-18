# TAR inner simulations versus the actual attack

The simulated attacker has substantial but incomplete overlap with the actual attack. The earlier comparison with the **outer defense trajectory** understated that overlap. At the last observed inner call, the full six-direction sampled span captures **12.83% of the first recorded attack displacement**, **55.34% of the displacement at attack step 16**, and **16.44% at step 1000**, using BF16-rounded endpoints. This supports incomplete early and late coverage; it does not establish that coverage mismatch causes the defense failure.

Start with the [capture-through-attack figure](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/capture-by-step.png), then the [plane-overlap figure](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/subspace-overlap.png). The [figure index](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/figure-index.md) gives full paths and explains all four new panels.

We reran local **B** from the original pinned Zephyr checkpoint: K16, 125 outer updates, retain coefficient 2.5, forget coefficient 1, inner LR 1e-6, and outer LR 5e-6. We reused the archived standard LR 1e-6, 1000-update attack. No new attacker was trained. Inner calls 1, 32, 64, 96 and 125 were observed, with states at inner updates 1, 2, 4, 8, 12 and 16. The [submitted config](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/cases/baseline-inner-recording/config.yml) and [analysis specification](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/source/reference-local-v6/analysis-spec.json) preserve the settings and exact snapshot selections.

**Replay verification passed.** All 125 mixed batches match the original order and content. All nine saved defense checkpoints have exactly equal parameter values and dtypes; all 2,000 ordered, finite inner-loss values match. The final defender also matches the saved checkpoint used to initialize the archived attack. See [replay validation](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/replay-validation.json), [verified completion](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/status/verified-defense.json), and [transport/anchor checks](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/protocol-summary.json). A final wrapper check used the wrong trainer-state path; its raw failure status is retained, while completion was verified from checkpoint-125 and these checks.

**How much of the actual motion was simulated?**

These are percentages of squared parameter-displacement norm projected into the last inner call's sampled span. They are not percentages of CE decrease or distance traveled.

| Actual attack step | Raw, all six directions | BF16-rounded, all six | BF16-rounded, two dominant axes |
|---|---:|---:|---:|
| 1 | 3.80% | 12.83% | 12.60% |
| 16 | 67.39% | 55.34% | 54.54% |
| 24 | 71.68% | 58.90% | 57.20% |
| 128 | 36.08% | 28.54% | 27.31% |
| 1000 | 20.88% | 16.44% | 15.70% |

Most of the first recorded displacement is outside this sampled inner span: 87.17% of its BF16-rounded squared norm. Alignment then rises, peaks at the saved step 24, and falls as the attack extends beyond the simulated 16-update horizon. The extra four inner directions add little coverage. This is not just a consequence of drawing a two-dimensional picture.

Across the six matching saves through step 16, the full inner span captures 49.73% of norm-weighted BF16 target energy; across the 13 saves through step 1000, it captures 27.57%. These aggregates weight larger displacements more heavily. They should not replace the step-by-step curve.

For comparison, the earlier outer-defense span captured only 0.033% at the first update and 0.888% at step 1000 in the BF16 view. That was a different object, not TAR's simulated attacker. [Earlier report](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-structure/20261116T2200Z/report.md).

**The planes align partially.**

“Matching horizon” means the same six saved step numbers through 16, not matched minibatches or identical optimizer settings. Both plane fits use two uncentered, norm-weighted principal axes.

| Endpoint representation | Angles, matching horizon | Plane overlap | Angles, 1000-update fit | Plane overlap |
|---|---|---:|---|---:|
| Raw | 29.81°, 78.60° | 39.59% | 42.25°, 88.78° | 27.42% |
| BF16-rounded | 41.88°, 76.99° | 30.25% | 47.70°, 89.36° | 22.65% |

Plane overlap is the mean squared cosine of the two principal angles. In the long comparison, one direction overlaps materially and the second is almost perpendicular. Two axes explain 95.08% of sampled inner displacement energy in the BF16 view, so the inner plane represents its dominant motion.

At calls 64, 96 and 125, matching-horizon BF16 plane overlap is approximately 29–30%. Earlier calls overlap less, but their starting parameters are farther from the final defender and their examples differ. This is not an isolated effect of defense training time.

The main conclusion survives alternative fits. At call 125, matching-horizon rank-two overlap ranges from 27.3–30.7% in BF16 and 36.7–39.6% in raw coordinates when switching between cumulative displacements, saved-interval updates, and equal-unit versus norm-weighted fits. However, BF16 energy capture for the interval-update fit is 27.52%, versus 49.17% for cumulative displacements. Sharing a cumulative drift does not imply matching each update.

**Module structure is shared, but its energy allocation is precision-sensitive.**

The table partitions the globally fitted matching-horizon axes by module, giving each orthonormal axis equal weight.

| Module | Raw inner | Raw actual | BF16 inner | BF16 actual |
|---|---:|---:|---:|---:|
| MLP | 65.60% | 35.93% | 39.54% | 51.82% |
| Output head | 18.75% | 47.40% | 50.74% | 35.55% |

The relative MLP/head ordering reverses between representations. A claim that one side consistently favors the output head is therefore unsupported. Independently refitting within each module gives stable matching-horizon plane overlap of 43.75% in attention and 42.74% in MLP, in both representations. The BF16 head overlap is 21.81%, versus 46.84% raw. These local refits and whole-model energy shares answer different questions. Neither measures a module's causal contribution to CE recovery. [Module figure](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/family-basis-energy.png).

**What this experiment does and does not isolate.**

Each inner call starts before its outer update; the actual attack starts from the final saved defender. At call 125, the raw anchor gap is 0.000145 in L2 norm, about 0.15% of the first actual displacement norm. Endpoint rounding raises the BF16 gap to 0.00443, about 12.36% of its first displacement norm. The actual trainer's post-loading step-zero parameters were not archived, so saved-anchor equality does not prove equality after dtype handling.

Both views subtract endpoints in FP32 and accumulate inner products in FP64. The BF16 view rounds each endpoint before subtraction; it is a parameter-representation sensitivity check, not a new CE evaluation.

The simulation and actual attack use different tokenized examples. None of the selected 16-row inner batches overlaps the actual first 16 rows by complete token-plus-mask signature. All these rows are full length, so the baseline's different padding-label policies are not active here. Both take minibatches of one at nominal LR 1e-6, but the inner loop uses constant LR, default AdamW weight decay 0.01 and no clipping; the actual attack uses linear decay, zero weight decay and clipping at 1. Those differences remain bundled.

“Full inner span” means the six sampled displacements from one call, not all 16 iterates, all 125 simulations, or every direction protected by the defense. No pooled cross-call span is fitted. Exact aggregate statistics were retained; full inner checkpoint files, coordinate/token sparsity statistics and new CE landscape grids were not retained or computed in this run. Existing CE views of this same archived attack are linked in the figure index.

The simulation captures a substantial component of the actual path, alongside a large component of the first recorded motion outside its sampled span. Whether that component matters for CE descent, or whether TAR fails to protect directions it does simulate, needs a same-state, same-data optimizer comparison and loss measurements along those directions. The present geometry alone does not choose between those explanations.

All eight numeric tables, normalization variants and input/source identities are under [geometry](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/). The completed recording job was f3kjxbbFnWk7; validation and figure generation ran as f3kksk1z9NGP.
