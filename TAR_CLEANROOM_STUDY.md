# Matched attacks and defense objectives

Extension of the [reproduction plan](TAR_REPRODUCTION_PLAN.md) and
[compute study](TAR_COMPUTE_EXTENSION.md).

The question is whether accurate attack simulation changes resistance, and
whether directly protecting attacker cross-entropy (CE) removes descent paths
that terminal maximum entropy leaves available.

Start each defense continuation from the same saved K16 defender. Use FP32
weights and loss evaluation, explicit AdamW settings, and a fixed corpus of
eight 256-token forget sequences. Here **full batch means this entire declared
small corpus**, not the complete source dataset. Reserve separate documents
for new-data attacks, held-out forget measurement, and retained-data measurement.

First verify independent simulator/actual-attacker update implementations on
the real 7B model. Match parameters, optimizer state, data, token reductions,
clipping, and precision. Independently validate short exact and first-order
outer gradients against their respective mathematical definitions.

Then run terminal maximum entropy, terminal CE-floor, and worst-trajectory
CE-floor continuations with identical first-order gradients and retention
regularization. Compare objectives within full-batch and fixed-minibatch regimes.
All use 125 outer updates and four inner updates at LR 1e-5. Full batch uses
eight documents per inner update; minibatches use two, so their inner token
budgets differ fourfold. Report both tokens and updates. This controlled regime
uses explicit zero weight decay and clipping; it is not the original inner
AdamW default. Keep the original reproduction results unchanged.

The CE floor is log(vocabulary size). The worst-trajectory objective penalizes
the lowest corpus CE among the initial state and all four attack states.
The terminal objective only penalizes the final attack state. Retained CE is
a training regularizer, not a utility guarantee. Assess utility with the original
benchmark harness and declared five-percentage-point MMLU limit.

Evaluate the unchanged starting checkpoint and every continuation under matched
attacks, opposite batching, reversed minibatch order, and attacks on unused
forget documents. Extend attacks to 64 updates and preserve checkpoints.
Keep training-corpus and held-out outcomes separate. Plot native-precision
attack trajectories and random versus attack-aligned slices. A high training
CE floor without held-out resistance is a limited fit to the training attack.

Treat these as mechanism experiments from one starting checkpoint and initial
corpus seed. Replicate promising contrasts before claiming improvement. Exact
simulation, successful optimization of its objective, and resistance to other
attacks are three distinct outcomes.

Before the main continuations, the engineering pilot passed actual-model
update parity and a two-update defense test. Long training runs use two
allocations (updates 1–64 and 65–125) with saved outer Adam moments and random
states; a focused test verifies exact agreement with uninterrupted training.
Native FP32 base and starting-checkpoint benchmarks calibrate the utility
comparison. Report both the original utility gate and FP32 base MMLU minus
five percentage points. Recheck update parity at every evaluated defender.
Benchmark the saved defense checkpoints at 32 and 125 updates on both original
task groups. At attack endpoints, benchmark the full WMDP group; retained CE
tracks attack-time utility, and post-attack MMLU is outside this initial plan.
After observing a large early recovery peak in the control, also benchmark
saved attack steps 1, 4, and 16 for all four variants and all nine defenders
(the starting checkpoint plus eight continuations). Report every measured
point and the strongest measured recovery; the attack family remains finite.

Two diagnostic extensions address failures observed during these runs. Keep
the six original continuations unchanged. Repeat full-batch terminal entropy
and worst-path CE with a shared, refreshed retain-data stream: 311 distinct
eligible excerpts from the same source split, eight per outer call, with the
original held-out retain examples excluded. This changes the retain-data
policy while preserving the forget data, update counts, and retain-token budget.
The first retain batch is identical to the original condition.

Separately search for low forget CE inside a fixed parameter ball of radius
0.75 around each final defender and the starting checkpoint. Use eight full-batch
normalized-gradient steps with a declared line-search grid and projection onto
the original ball. Select candidates using training forget CE only; measure
held-out loss and WMDP separately. A feasible low-CE point disproves the stated
mean-CE floor at that radius. An unsuccessful search provides no certificate.
Its line-search compute differs from the ordinary AdamW attack.

For geometry, compare the shared broad random plane, the same random directions
at the actual first-update radius, a literal fixed control-trajectory plane,
the first-update plane, and a plane fitted separately to each attack. Preserve
physical axis scales and use common CE scales when comparing defenders. Projected
saved states are reconstructions in a slice, not measurements of the actual
off-plane trajectory.

Completed 2026-11-11: eight continuations, 36 attack trajectories, 162 regular
benchmark points, nine bounded searches plus one fine-grid followup, and native
landscape grids for all nine defended starting points. The useful entropy
continuation also has first-update and fitted views of its stronger new-data
attacker, with that attacker's own loss measured alongside the defender probes.

[Findings and figures](examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/findings.md)
and the [complete evidence tables](examples/tamper-resistance/artifacts/tuolumne/tar-cleanroom/20261111T0500Z/report/report.md)
separate numerical fidelity, proxy-loss satisfaction, utility, and recovered
benchmark capability. With the same four attack updates and optimizer, WMDP
is 26.73% on matched data versus 51.96% after changing the eight examples;
the defender passes the measured utility gate at MMLU 54.14%.
