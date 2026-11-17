# Controlled TAR hyperparameters, geometry, and attack behavior

The learning-rate interaction changes the random-plane geometry toward the collaborator-like tower. It does not remove the attacker's immediate descent direction. The joint recipe also reduces WMDP recovery under the standard 1000-step attack, with a utility difference that prevents treating this as a shape-only effect.

Training, benchmarks, native landscape measurements, and final comparison figures are complete. The [figure index](figure-index.md) links the matched comparisons, paired 3D views, first-update planes, and projection checks.

B is the completed local K16 control: retain coefficient 2.5, simulated attack LR 1e-6, outer defense LR 5e-6. R changes retain to 4; I changes the simulated attack LR to 1e-5; O changes the outer LR to 1e-5. RI, RO, IO, and RIO combine those changes. All use K16 for 125 defense updates, the original local sampler and exact observed batch contents/order, and the remaining source/settings unchanged. B/R/I/O/RIO have the standard LR1e-6, 1000-update attacks. The three additional pair cells have endpoint measurements only.

The [eight-condition comparison](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/findings.md) identifies the shaping effect. Raising inner LR alone produces a lower bowl. Raising outer LR alone preserves the plateau. Raising both produces pronounced, asymmetric shoulders at retain 2.5. Raising retain to 4 in that joint-LR setting lifts center CE by 1.9013 nats and reduces the maximum immediate-neighbor rise by 0.9123 nats. At radius 125, IO has six immediate neighbors above center and two slightly below; RIO has all eight higher. Those neighbors are spaced 31.25 units away, so these observations establish neither a local minimum nor an all-around barrier between samples.

The conditional inner/outer interaction on center CE at retain 2.5 is 5.5591 nats; the extra three-way coefficient is 3.0760 nats. These are descriptive contrasts at the selected levels from one training seed, without uncertainty estimates or a fitted dose response. The [full report](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/report.md) contains all eight radii, exact definitions and contrasts. The [six-checkpoint context comparison](static/comparison/report.md) includes the uploaded model. Matching the published hyperparameters keeps the collaborator's legacy data pipeline and implementation outside the controlled comparison.

All percentages below come from complete benchmark evaluations. Static CE uses the original fixed-ten-document BF16/default-attention recipe.

| Condition | Static center CE | WMDP at 0 | WMDP at 1000 | Highest measured WMDP through 1000 | MMLU at 0 | MMLU at 1000 |
|---|---:|---:|---:|---:|---:|---:|
| B | 11.2384 | 33.06 | 53.57 | 53.69 | 54.02 | 58.06 |
| R | 10.9572 | 33.64 | 54.61 | 54.61 | 54.91 | 57.84 |
| I | 6.9325 | 51.84 | 50.12 | 52.19 | 54.62 | 54.94 |
| O | 11.2568 | 27.07 | 51.27 | 51.38 | 48.88 | 55.02 |
| RIO | 14.4114 | 36.29 | 44.59 | 44.93 | 52.62 | 53.96 |

The joint recipe's highest measured WMDP score is 44.93%, versus B's 53.69%. Those peaks occur at saved attack steps 256 and 192, respectively. This is improved suppression under this particular attack budget and LR, while substantial recovery remains. Its starting WMDP accuracy is higher than B's despite its higher fixed-ten CE, and its utility is lower before and after attack. I starts with much weaker benchmark unlearning; its lower final score than R is not evidence of stronger resistance. O initially suppresses WMDP more strongly while losing substantially more utility.

The original base MMLU reference is 58.5873%. A five-percentage-point initial utility limit would require at least 53.5873%: B/R/I satisfy it; O and RIO do not. RIO is just above that threshold after attack, but remains about 4.10 percentage points below B's final utility. This screen identifies conditional geometry and behavioral effects of the recipes; it does not identify an effect of geometry at matched starting capability and utility.

[Validated scores and coverage](evaluation-report.md), [raw score table](evaluation-scores.csv), and [highest measured scores](max-observed-wmdp.csv) retain every milestone. Each new attack has 27 WMDP checkpoints and MMLU at 0/128/512/1000, for 31 task/checkpoint pairs; the final collector verified all 124 results and rejected none. The collector's utility-matching CSV covers defense-role evaluation jobs. This study evaluates defense endpoints as attack step zero instead and shows their validated utility directly here.

B is the completed baseline-lr1e-6 case in the [control score table](../../tar-compute/20261110T1815Z/evaluation-scores.csv), [control peak table](../../tar-compute/20261110T1815Z/max-observed-wmdp.csv), and [control actual-checkpoint CE table](../../tar-compute/20261110T1815Z/cases/baseline-lr1e-6/surfaces/actual-losses.csv). The main-study score CSV contains the four new cases.

The saved first updates establish immediate descent even in the joint tower-like condition:

| Condition | Fixed-ten CE before → after update 1 | This update's batch CE before → after |
|---|---:|---:|
| B | 11.2385 → 11.1502 | 11.1336 → 11.0407 |
| R | 10.9573 → 10.9024 | 10.8770 → 10.8095 |
| I | 6.9328 → 6.7806 | 7.6508 → 6.9210 |
| O | 11.2568 → 11.1581 | 11.2046 → 11.0949 |
| RIO | 14.4078 → 13.8177 | 14.6569 → 12.3253 |

The joint condition drops 0.5901 nats on the fixed probe and 2.3317 on the actual first minibatch. No positive barrier appears at the sampled fractions between its saved update endpoints; this is a finite sampling statement, not a proof of a smooth barrier-free path. Its first update has raw saved-weight L2 distance 0.10984 and evaluated-endpoint distance 0.04353, both far smaller than the static grid spacing. Raw and evaluated distances differ because of the loading/casting map. The [joint first-update plane](cases/hparam-rio-lr1e-6/first-update-landscape.png) and [eight update segments](cases/hparam-rio-lr1e-6/early-update-profiles.png) show descent on both fixed and actual-batch probes.

These first-update measurements use the existing eager-attention evaluator and FP32 direction convention, with separately cast saved endpoints. Keep them distinct from the static BF16/default-attention measurements. Their slightly different CE centers are measured, rather than forced to agree. The narrow first-update axis and broad random axis also have different plotted ranges. A fitted attack plane selects directions from the recorded path; its clear descent is not an unbiased random-plane estimate.

The complete actual-checkpoint pass also measures a residual loss difference at the end of attack:

| Condition | Fixed-ten CE at 0 | Fixed-ten CE at 1000 | Attack-held-out-32 CE at 0 | Attack-held-out-32 CE at 1000 |
|---|---:|---:|---:|---:|
| B | 11.2385 | 1.9495 | 11.1930 | 2.0196 |
| R | 10.9573 | 1.9497 | 10.9212 | 2.0253 |
| I | 6.9328 | 2.6065 | 7.3294 | 2.7523 |
| O | 11.2568 | 2.0052 | 11.2392 | 2.0882 |
| RIO | 14.4078 | 5.9654 | 14.4811 | 5.9989 |

The 32 attack-held-out probes were excluded from the recorded attacker minibatches; no claim is made about exclusion from defense or pretraining. The joint condition combines a larger immediate loss drop with less eventual CE recovery under this attack schedule. The residual appears on both fixed and attack-held-out probes. This is a budget-specific optimization result: the linear LR schedule approaches zero, using LR1e-9 for the final update and reaching zero afterward. Its endpoint does not establish convergence or removal of all viable descent directions. The hyperparameters change how far this ordinary attack progresses, even though its first updates already find descent.

The joint recipe's [broad random-plane progression](cases/hparam-rio-lr1e-6/random-landscape-evolution.png) moves the center through defense and attack while keeping the random directions and fixed-ten objective unchanged. Its early basin flattens around defense step 32, the raised center develops by step 125, and attack recentering produces a lower bowl by step 1000. These are slices around different checkpoints of one high-dimensional objective.

In the [fixed-slice view](cases/hparam-rio-lr1e-6/surfaces-and-checkpoints.png), reference denotes the original K16 128-update attack plane, reused across conditions. Attack denotes this condition's own 1000-update path fit. The joint case's two fitted axes explain 98.21% of the sum of squared raw snapshot displacements, but capture only 4.83% of its first update. Their largest fixed-probe projected-loss error is 1.958 nats at attack step 8; at step 1000 capture is 99.75% and the error is 0.057 nats. The original reference plane captures only 0.206% of its final displacement and misses its actual loss by 8.443 nats. See [projection fidelity](cases/hparam-rio-lr1e-6/projection-fidelity.png) and the [numeric checks](cases/hparam-rio-lr1e-6/projection-quality.csv).

A small step relative to the broad plot radius can still leave the plotted two-dimensional plane. Points drawn at projected x/y with their actual CE can therefore lie below the surface. The uncentered PCA energy fraction summarizes the sampled path, while per-checkpoint capture and loss error assess individual projections. Neither metric measures all protected directions or a global barrier.

The prerequisite [attack/defense basis analysis](../../tar-structure/20261116T2200Z/report.md) covers B and the uploaded model's attack and was completed before these training runs. The attack planes are concentrated but still involve hundreds of millions of parameters, with broad token-row support. Output-head Euclidean energy is large, while frozen attention-plus-MLP checkpoint splices reproduce most measured forget-CE recovery. Some apparent unseen input-embedding movement is a loading-cast offset. B's defense and attack planes are about 85 degrees apart; its full recorded defense span captures about 0.03% of the first attack displacement and 0.9% of its final displacement. That distinguishes the observed paths, without equating the defense path span with all protected loss directions.

The present evidence supports separating three questions: which recipe shapes the random slice, whether ordinary updates immediately lower forget CE, and how much benchmark recovery occurs at a specified attack budget and utility level. The raised shoulders, immediate descent, and incomplete recovery can coexist.
