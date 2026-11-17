# Simulated inner attack versus the recorded actual attack

Rerun the completed local B defense with unchanged model, K16, data/order, seeds,
losses and optimizer settings. Observe its simulated attack at outer calls
1, 32, 64, 96 and 125. Within each call, use displacements at inner updates
1, 2, 4, 8, 12 and 16 from that call's exact starting parameters.

Compare these fitted bases with the existing B attack, first at the same six
updates, then separately against its recorded 1000-update path. Report ranks,
principal angles, directional projection capture and module contributions.
Use raw and BF16-evaluated endpoint differences separately. Inner iterates stay
transient; retain aggregate Gram and cross-Gram statistics sufficient for these
comparisons. No additional attack training is planned.

Require exact agreement with the original B batches and all saved outer model
states before interpreting the comparison. Inner call t starts before outer
update t, while the actual attack starts after outer update 125. Quantify that
anchor difference and distinguish observed mismatch from a causal explanation
of resistance. An inner-path basis is also not every direction protected by
the defense objective.

This replaces the outer-path proxy used in the earlier
[basis study](TAR_BASIS_STRUCTURE.md), following the
[hyperparameter study](TAR_HPARAM_STUDY.md).
Execution and results: [study directory](examples/tamper-resistance/artifacts/tuolumne/tar-inner-basis/20261117T2300Z/).
