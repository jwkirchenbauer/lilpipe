# TAR clean-room findings

The clearest result is selective resistance. A continuation that preserves the
measured utility gate suppresses its simulated four-step attack, while another
eight forget examples recover much of the benchmark capability with the same
attack settings and compute. Random-plane flatness does not reveal this gap.

This is a mechanism study from one saved K16 defender, one corpus seed, and eight
256-token forget examples. All model weights and forwards use FP32. Each
continuation has four simulated attack updates and 125 outer optimizer calls
(124 with nonzero learning rate). The external attacks use constant AdamW LR
1e-5, clipping at 1, and explicit zero weight decay. These controlled settings
are recorded in [study.json](study.json) and the [execution sources](source/).

## Utility and strongest measured recovery

Native base scores are WMDP **54.9539%** and MMLU **58.6023%**. The native
five-percentage-point utility gate is **53.6023%**; the original gate is
53.5873%. MMLU below is measured before attack. Post-attack MMLU was outside the
declared evaluation plan.

Full means all eight training examples per inner update; mini means groups of
two. Terminal entropy maximizes vocabulary entropy after update four. Terminal
CE protects a mean cross-entropy floor at update four; worst-path CE protects
the minimum mean CE over states zero through four. The floor is log(32,000),
or 10.37349 nats. Fixed retention repeats eight examples; refreshed retention
uses a shared scheduled pool of 311 examples from the same source split.

| Continuation | Retain data | MMLU (%) | WMDP before (%) | Largest measured WMDP, including no attack (%) | Attacker / update |
| --- | --- | ---: | ---: | ---: | --- |
| Starting K16 defender | Original | **54.40** | 34.91 | 54.84 | Reversed mini / 1 |
| Full terminal entropy | Fixed 8 | 22.92 | 26.73 | 42.86 | New data / 16 |
| Full terminal CE | Fixed 8 | 52.96 | 32.83 | 44.59 | Forward mini / 4 |
| Full worst-path CE | Fixed 8 | 53.19 | 38.36 | 38.71 | Reversed mini / 1 |
| Mini terminal entropy | Fixed 8 | 22.92 | 26.73 | 38.94 | New data / 16 |
| Mini terminal CE | Fixed 8 | 48.30 | 26.96 | 41.59 | New data / 4 |
| Mini worst-path CE | Fixed 8 | 47.12 | 27.30 | 27.30 | No attack / 0 |
| Full terminal entropy | Refreshed 311 | **54.14** | 36.52 | 51.96 | New data / 4 |
| Full worst-path CE | Refreshed 311 | **55.56** | 48.27 | 50.00 | New data / 4 |

Every defender has all four declared attack variants evaluated at updates
1, 4, 16, and 64. The table takes a maximum over those measurements and the
unchanged defender, not over every possible attack. [Peak scores](report/peak-wmdp.csv)
and the [complete report](report/report.md) preserve all points and statuses.

The low recovery scores of the six fixed-retain continuations cannot establish
utility-preserving robustness: all miss the utility gate. Full worst-path CE
is close, missing the native gate by 0.4121 points; the entropy cases lose much
more capability. Refreshing retention fixes this major confound for two cases.
Full and mini training still differ fourfold in inner token exposure, so
objective comparisons are controlled within each regime.

## Same attack budget, different forget examples

For the utility-passing entropy continuation, switching only the eight attack
examples produces the decisive transfer gap:

| Defender | Same training eight, attack 4 | Other eight, attack 4 |
| --- | ---: | ---: |
| Starting K16 defender | 49.77% | 46.31% |
| Entropy continuation with refreshed retention | 26.73% | 51.96% |

Both columns use full batches, four updates, the same optimizer and LR, and
8,160 supervised tokens. Both replay checks at the continued defender pass the
declared numerical tolerances. Thus this comparison isolates the attack data
within the tested implementation; it is not explained by a longer attack or
a different batch size.

After the new-data attack, CE on the defender's training eight falls to 4.810,
CE on the attacker's eight to 0.825, and held-out CE to 2.694. Under the matched
attack, training CE remains 10.236 and held-out CE 10.370. The defender learned
a barrier for its declared small training attack, with limited transfer to the
other examples. Those examples are excluded from this continuation; previous
use during pretraining or the original K16 defense is not ruled out.

Endpoint-only evaluation would hide important failures. For the starting
defender, full-batch WMDP rises to 49.77% at update four and falls to 32.49% at
64. Reversing the first minibatch reaches **54.84% after one update and 510
supervised tokens**, within one question of the native base score on the 868
benchmark questions. The forward first minibatch reaches only 31.68%.

## What the landscape views establish

At the starting defender, a random displacement of L2 0.7494 changes training
CE from 11.173992 to 11.173911. The actual first full-batch update has the same
norm and lowers CE to 9.821282. The random plane captures only
7.34e-11 of that update's squared displacement. Typical isotropic directions
therefore give a very different view from an optimizer-selected direction.

The planes are centered on the saved defender. Their zero-displacement and
native parameter-construction checks are recorded with each grid. First-update
planes reproduce the corresponding first saved update to numerical precision.
Fitted planes remain projections: for the control's first update, the fitted
plane captures 24.9% of displacement energy and its reconstructed training-forget
CE differs by 2.277 nats. At update 64 the capture is 99.93%; training-forget
CE error is about 1.2e-5 nats, while heldout-subset CE error is 0.0445 nats.
Use the actual traces for trajectory claims, rather than treating a projected
curve as the optimizer's path.

The broad random plane shares physical unit directions across cases. The
matched-radius random plane uses those same orientations at each case's first
update norm. The control-reference axes and their scale are fixed across all
nine cases; first-update and fitted axes are specific to each recorded attack.
Plot radii are per-axis half-widths: a square's corners lie farther away than
its per-axis radius. The bounded local searches instead constrain total L2
distance to 0.75.

The all-case figures share CE limits 0–82. The utility-passing comparison uses
shared limits 0–32, including the additional attacker view, recorded in
[rendering.json](geometry/rendering.json).
Current grids use new directions and probe cohorts with FP32 evaluation.
Their absolute appearance is not directly interchangeable with the earlier
BF16 plots.

## The recovering attack's own loss surface

The [new-data 3D views](geometry/full-maxent-terminal-fresh-retain/attack-variants/full-new/utility-comparison/slices-3d.png)
and [heatmaps](geometry/full-maxent-terminal-fresh-retain/attack-variants/full-new/utility-comparison/slices.png)
use one model, one center, and identical parameter directions across their three
rows: continuation training eight, held-out eight, and actual attacker eight.
This compares empirical losses on different examples without changing the
parameter frame between rows.

The new-data first update has measured L2 norm 0.8126; the matched-data first
update has norm 0.8047. The transfer experiment fixes optimizer settings and
token compute, rather than total parameter displacement. At that new-data
first checkpoint, training-forget CE falls from 12.0813 to 11.3144, while
attacker CE falls from 11.7556 to 10.0342. The first-update slice reproduces
the latter checkpoint loss within 4.8e-7 nats.

The fitted view is informative but needs its residual checks. Its projection
of the first update captures 47.2% of displacement energy and gives attacker
CE 3.7244, versus the actual checkpoint's 10.0342. That is a real counterfactual
model in the slice, not the optimizer's visited model. Interpreting it as the
actual first step would overstate progress by 6.31 nats. The first-update view
and native checkpoint traces prevent that mistake.

## A search can miss a narrow descent region

The refreshed CE defender initially defeats the declared coarse monotone
search: its smallest normalized-gradient step, 1e-3, raises CE from 12.2915 to
12.3830. A separately recorded adaptive grid finds that 1e-4 lowers CE to
11.9517. Subsequent steps reach CE **7.6117 within actual L2 0.74999849**.

[Measured small-step candidates](report/local-search-fine-first-iteration.png)
show the missed interval. The original coarse result is preserved. The fine
run uses the same eight full-corpus gradient iterations, comprising 32 chunk
backward calls, but more candidate forward evaluations:
301,920 versus 204,000 supervised forward tokens, including probes. This is a
counterexample to the mean-CE floor, not a robustness certificate or a fair
fixed-compute comparison of search algorithms.

Its final WMDP is only 26.73%, versus 48.27% before search. Conversely, the
control's reversed minibatch recovers near-base WMDP while forget CE is still
9.756 and entropy 10.041, close to the uniform reference 10.373. CE descent,
high entropy, and benchmark recovery must therefore be measured separately.

## Objective satisfaction and simulation fidelity are separate

Four of eight matched regimes pass the numerical replay criteria; four do not.
Some CE defenders amplify initially small arithmetic differences over four
updates. Whole-batch versus accumulated gradients and different norm reductions
are both present, so these data do not uniquely attribute the amplification.

There is also a clean matched counterexample to extrapolating the trained
horizon: mini worst-path CE meets its final four-step floor at 10.5816 with
exactly equal paired updates, but the matched 64-step attack reaches CE 5.8816.
That model fails utility. The refreshed worst-path CE defender preserves utility
but misses its own final simulated floor by 1.3123 nats. An unsatisfied training
criterion and an insufficient satisfied criterion are different findings.

Short exact-gradient diagnostics do not settle whether the first-order
approximation causes the main behavior. Exact and first-order AdamW1 gradients
differ, but both tested negative directions improve that diagnostic's objective;
the finite differences are scale-sensitive. See the [diagnostic records](diagnostics/)
and [original-six audit](scientific-audit-final-original-six.md) for their scope.

The operational target should refer to recovered capability under a specified
attack budget, together with utility at the defended model:
U(theta) >= u_min and max over permitted attacks A of C(A(theta)) <= c_max.
The attack family must declare its data, optimization and query/token budgets,
and any parameter constraint. These experiments measure a finite family.
This defines an evaluation target; it does not supply a guaranteed training
algorithm for the worst-case recovery problem.

A CE floor constrains how far loss may fall in a region, rather than requiring
a zero gradient. Maximizing entropy can make its own gradient small without
making the attacker's CE gradient small: at uniform logits, the CE logit
gradient is p minus the one-hot target. Neither surrogate directly enforces the
capability condition above. For example, four single-token
answer probabilities can all lie just below 1/V while retaining a knowledgeable
model's relative ordering. Redistributing the remaining mass over other tokens
makes vocabulary entropy arbitrarily close to log(V), while every answer CE
exceeds log(V) and four-choice predictions stay unchanged. Exact uniformity
ties the choices; near-uniformity need not erase their ordering.

## Figures and reproducibility

The useful comparisons use a shared scale within the utility-passing group:

- [Starting defender: 3D planes](geometry/control/utility-comparison/slices-3d.png)
  and [equal-radius sections](geometry/control/utility-comparison/matched-radius-sections.png).
- [Entropy continuation: 3D planes](geometry/full-maxent-terminal-fresh-retain/utility-comparison/slices-3d.png)
  and [equal-radius sections](geometry/full-maxent-terminal-fresh-retain/utility-comparison/matched-radius-sections.png).
- [CE continuation: 3D planes](geometry/full-ce-worst-fresh-retain/utility-comparison/slices-3d.png).
- [All capability trajectories](report/wmdp-attack-trajectories-full.png),
  [local-search results](report/local-search-ce.png), and
  [all scores and evidence tables](report/report.md).
- [Figure index for every case and the additional attacker](geometry/index.md).

Each case also has full-range heatmaps, 3D PNG/PDF exports, measured grids, and
projection-error records under geometry/. Source profiles, data declarations,
amendments, and checkpoint byte hashes preserve the exact execution. Scope is
one checkpoint, one seed, and a deliberately small forget corpus; this study
does not establish a general tamper-resistance guarantee or a full-corpus
comparison of defense methods.

Verification covers 73 focused test methods: 71 CPU/plot methods passed, and
the two multi-GPU methods passed separately on allocated nodes. Cached and
streamed native parameter constructions and their probe CEs agree exactly at
the validation coordinates. The final coverage and curated-file checksums are
recorded in completion.json and artifact-manifest.sha256.
