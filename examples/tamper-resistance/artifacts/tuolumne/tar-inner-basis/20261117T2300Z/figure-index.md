# Inner simulation and actual attack: figure paths

1. **Capture through the actual attack.** Read this first. The late inner bases capture little of the first recorded displacement, more around steps 16–24, and less again later. Left panels use two inner axes; right panels use all six sampled directions. Colors identify separate inner calls, not successive points on one inner trajectory.

   [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/capture-by-step.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/capture-by-step.png) · [PDF](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/capture-by-step.pdf)

2. **Subspace overlap.** Rows are inner calls; columns compare the six matching saves through 16 with the 13-save fit through 1000. Rank-two overlap shows partial alignment. Full-rank panels divide overlap by the actual or inner rank separately, so a rank-6/rank-13 comparison has different denominators.

   [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/subspace-overlap.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/subspace-overlap.png) · [PDF](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/subspace-overlap.pdf)

3. **Module energy of the fitted axes.** Compare inner and actual columns within each representation. Actual rows repeat intentionally: the same actual plane is used for every inner call. Raw versus BF16 changes the MLP/head ordering. These are equal-axis geometric shares, not loss contributions.

   [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/family-basis-energy.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/family-basis-energy.png) · [PDF](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/family-basis-energy.pdf)

4. **Turning within each inner simulation.** Each heatmap compares net updates over saved intervals. Raw late-call updates are more correlated than their BF16-rounded differences. Intervals 2–4 and 4–8 contain multiple steps; they are not individual minibatch gradients.

   [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/within-call-update-similarity.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/within-call-update-similarity.png) · [PDF](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/geometry/within-call-update-similarity.pdf)

The [report](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/report.md) defines the measurements and anchor differences. These four figures show parameter-trajectory geometry.

For existing CE landscape context from the same archived standard attack, use:

- [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/first-update-landscape.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/first-update-landscape.png)
- [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/surfaces-and-checkpoints.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/surfaces-and-checkpoints.png)
- [/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/random-landscape-evolution.png](/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/random-landscape-evolution.png)

Those existing CE surfaces use their original random and actual-attack frames. They were not recomputed in the new simulated-inner basis.
