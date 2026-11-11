# Independent scientific audit — interim

This audit reviews the completed native FP32 base/control benchmarks, control attacks, actual-model parity test, SGD1 diagnostic, and the six defenders' first 64 outer steps. Some resumed traces had progressed further when read. It does not rank the final defenders: their step-125 capability and utility evaluations are not yet available.

## State labels and budgets

- Defense row `i` simulates an attack from **pre-update θ_(i−1)**. Its `defender_probes` and `defense-i` checkpoint measure **post-update θ_i**. Do not join these as measurements at the same weights. The last scheduled learning rate is zero, so θ_125 equals θ_124; this special case does not change the general labeling rule.
- There are 125 outer optimizer calls and simulations, but 124 calls change parameters. All 125 simulations still consume compute.
- Every sequence has 255 supervised next tokens. Full8/K4 consumes 8,160 inner tokens per outer call, versus 2,040 for mini2/K4: totals of 1,020,000 versus 255,000 over 125 calls. A 64-update external attack consumes 130,560 versus 32,640 tokens. Full16 versus mini64 is an available equal-token comparison.
- Objective contrasts within each regime are controlled. Between-regime contrasts combine batch size and token exposure. Fixed minibatch order is deterministic; order effects should not be described as stochastic minibatch noise.

The data manifest contains 48 distinct forget text hashes and 48 distinct token sequences across train8/new-attack8/heldout32, plus 24 distinct retain texts across train8/heldout16. These are short source-row excerpts, not evidence of independence across all related source documents. Heldout32 is excluded from **main defense updates**. Its first eight documents are gradient targets in the separate SGD diagnostic, so it is not globally untouched throughout every diagnostic.

## Retention overfitting is already visible

These are **post-update step-64** CE measurements in nats:

| Defender | Retain train8 | Retain heldout16 |
| --- | ---: | ---: |
| Full terminal entropy | 0.029 | 10.636 |
| Full terminal CE floor | 0.025 | 4.035 |
| Full worst-path CE floor | 0.058 | 4.701 |
| Mini terminal entropy | 0.025 | 11.383 |
| Mini terminal CE floor | 0.021 | 3.092 |
| Mini worst-path CE floor | 0.022 | 6.359 |

The unchanged control's heldout retain CE is 2.596. The fixed retain8 regularizer has fitted its examples while generalization worsened. These measurements do not establish MMLU failure, which requires the benchmark; they rule out treating low retain training CE as evidence of preserved utility.

## Proxy losses and attacker selection

CE ≥ log(32,000) = 10.37349 is a uniform-prediction reference, not an uncertainty or erasure guarantee. At step 64, full worst-path CE has heldout forget mean CE 10.483 but entropy only 1.599; some sequence CE values are as low as 7.225. Mean-floor success can coexist with confident wrong predictions and uneven suppression. Terminal entropy likewise describes the attacked terminal state, not every state or general utility.

The control full-same attack reaches training CE 0.0082 at step 64, while heldout CE is 5.106. Heldout CE was lower, 2.602, at step 8. Reverse minibatch order also differs substantially from the declared forward order before any new defense is trained. Therefore report memorization, transfer, and order dependence separately. Evaluate capability at intermediate attack checkpoints and report the strongest measured recovery; a final-step-only score can miss a stronger earlier attack. CE and entropy are capability proxies; neither replaces WMDP or utility evaluation.

## Fidelity and the exact-gradient diagnostic

The control four-step simulator comparison is strong but not bitwise exact: full-same maximum relative update-L2 error is about 1.01e−4, with terminal CE gap 3.44e−5; mini-same values are about 3.01e−6 and 2.38e−7. Regime matching and measured numerical fidelity are separate properties. Final defenders require their own replay fields before being labeled verified; baseline parity does not establish fidelity everywhere in weight space.

The actual-model SGD1 diagnostic reports exact/first-order cosine 0.99361 and relative L2 difference 0.11532. Both normalized negative-gradient directions improve the tested negative-CE objective at all three radii. Finite differences are not uniformly convergent: exact-gradient relative errors along its own direction are 42.0%, 1.68%, and 3.23% at radii 0.03, 0.01, and 0.003; along the first-order direction they are 15.6%, 1.37%, and 0.093%. Disclose these values rather than claiming blanket numerical validation or a catastrophic first-order failure. This is clipped SGD1 on heldout8 with terminal negative CE; it does not determine the behavior of the main AdamW/K4 hinge or entropy objectives.

## Utility calibration and causal limits

The native benchmarks use identical task hashes, package versions, batch32, zero-shot prompts, and all 59 leaf tasks: 868 WMDP questions and 13,322 MMLU questions. Native base MMLU is 58.60231%, giving a native five-point-drop gate of **53.60231%**. The original gate remains **53.58729%**. Their difference is 0.01502 percentage points, approximately two MMLU answers; report both without rounding away possible boundary differences. Native control MMLU is 54.39874%, 0.79643 points above the native gate; WMDP is 34.90783%. This calibrates the native evaluation protocol, rather than isolating precision as the sole cause of differences from the original distributed BF16 evaluation.

All main claims remain conditional on one checkpoint, one small corpus, one seed, fixed coefficients, and the tested attack family. A shared geometry basis and physical scale permit comparison of translated local sections; they do not establish full-space connectivity or turn projected checkpoints into an actual attack trajectory.

## Smallest retention followup, conditional on MMLU failure

Fresh retain CE is the first change to test. Use the same predeclared stream of eight retain sequences per outer call for full terminal entropy and full worst-path CE floor, keeping all forget data, initialization, dtype, losses, coefficients, schedules, and update budgets fixed. Shuffle a large deduplicated pool without replacement before cycling; record the available pool size and actual reuse, and exclude the heldout retain examples and duplicate token sequences.

Prefer the existing retain source split where feasible. Current retain8 comes from WikiText-2 **test**; moving to fresh **train** examples also changes the source split. If needed for sufficient data, declare this as a retain-data-policy followup. The paired objective contrast remains controlled, but old-to-new changes cannot be attributed solely to refresh. A 1,000-example stream supplies 992 examples to parameter-changing steps because the final outer learning rate is zero.

Teacher KL should wait: it changes the retention objective and adds teacher choice and gradient-scale questions. Fresh CE directly tests the observed fixed-cohort memorization. If utility still fails, KL can then be compared on the identical fresh-data stream.
