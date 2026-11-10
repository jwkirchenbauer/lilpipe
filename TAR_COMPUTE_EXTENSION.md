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
