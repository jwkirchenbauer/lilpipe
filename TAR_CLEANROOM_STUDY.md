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
