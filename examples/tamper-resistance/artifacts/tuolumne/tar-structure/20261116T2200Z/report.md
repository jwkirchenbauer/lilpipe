# Where the attack moves, and how it differs from defense

The fitted attack plane is concentrated relative to isotropic random directions, but its energy still covers hundreds of millions of coordinates. It is not localized to a few embedding tokens. More importantly, the local attack follows directions almost orthogonal to the observed defense trajectory. Replaying saved changes shows that the transformer body produces most of the fixed and held-out probe cross-entropy (CE) decrease, despite the output head's large share of Euclidean basis energy.

This analysis uses the completed local K16 defense and its standard 1000-update LR1e-6 attack, plus the same attack on the collaborator's uploaded checkpoint. No new defense training was needed. The collaborator checkpoint has no defense history, so its defense-preferred basis cannot be fitted. See the [original reproduction plan](../../../../../../TAR_REPRODUCTION_PLAN.md) and [checkpoint comparison](../../../../../../TAR_COLLABORATOR_CHECKPOINT.md).

**What the basis measures.** Each phase uses an uncentered principal-component fit to saved displacements from that phase's starting checkpoint, matching our existing landscape analysis. The defense has nine saves at updates 1, 2, 4, 8, 16, 32, 64, 96, 125; its first displacement is zero because the first learning rate is zero. The attack fit uses 13 saves through update 1000. Two defense axes capture 98.18% of summed squared saved displacement; two attack axes capture 95.80%. These are sampled path fits, with no centering or correction for uneven checkpoint spacing. They describe the realized optimizer path rather than the gradient covariance, Hessian, or all directions the defense protects.

**Three concentration measures.** Let $\Delta_t=\theta_t-\theta_0$ denote a saved displacement from the phase's starting model. Each measure below is normalized by its total energy across all parameter coordinates.

| Measure | Coordinate energy before normalization | What it describes |
|---|---|---|
| Endpoint delta | $E_i=\Delta_{T,i}^2$ | Net weight change at the phase's end |
| All saved displacements | $E_i=\sum_t\Delta_{t,i}^2$ | Full recorded trajectory energy, preserving displacement magnitudes |
| Equal-axis fitted plane | $E_i=(u_i^2+v_i^2)/2$, for orthonormal fitted axes | Coordinate support of the two-dimensional subspace, with equal weight on each axis |

The trajectory measure includes every listed save once, without PCA truncation. It is not a sum of adjacent optimizer updates or a time average: uneven checkpoint spacing affects its weighting. Larger displacements contribute more energy; an excursion can contribute even if it cancels in the endpoint delta. The plane measure is invariant to rotations within its plane and weights its two axes equally, regardless of the trajectory's eigenvalues. Gram matrices accumulate in FP64; plane coordinates are reconstructed in FP32. Here the defense trajectory means the **outer defense parameter path**, not TAR's simulated inner attacks.

**Parameter concentration.** Fractions below refer to all 7.24 billion parameters. Coverage bounds come from energy histograms, rather than exact global sorting.

| Plane | Coordinates needed for 90% of energy | Exact nonzero coordinates in saved-weight fit | Output-head share of plane energy |
|---|---:|---:|---:|
| Local defense | 13.884–13.904% | 27.94% | 24.78% |
| Local attack | 5.310–5.312% | 13.68% | 34.36% |
| Collaborator attack | 2.393–2.395% | 14.54% | 69.79% |

For comparison, an isotropic Gaussian two-dimensional plane needs about 59% of coordinates for 90% of energy. The attack planes are therefore compressible, but even the collaborator's 2.39% means about 173 million coordinates. Exact zeros are a property of these serialized checkpoints and their precision, not proof that the underlying gradients were zero.

**Endpoint and full-trajectory comparison.** The earlier 13.90% defense / 5.31% local attack figures describe the equal-axis planes. Direct weight-delta and trajectory measurements give the following bounds, using the same energy histogram method and all 7.24 billion coordinates:

| Phase, native saved weights | Endpoint delta: coordinates for 90% energy | All saved displacements: coordinates for 90% energy | Equal-axis plane: coordinates for 90% energy |
|---|---:|---:|---:|
| Local outer defense | 10.9304–10.9418% | 12.4266–12.4426% | 13.8840–13.9039% |
| Local attack | 4.2295–4.2357% | 5.2485–5.2539% | 5.3102–5.3124% |
| Collaborator attack | 1.2856–1.2865% | 1.4348–1.4359% | 2.3927–2.3950% |

The endpoint delta is more concentrated than the equally weighted plane in all three cases. Full trajectory energy lies between them at the 90% threshold. Thus the numerical concentration depends on the object being measured, while the ordering—local defense more diffuse than local attack, and collaborator attack most concentrated—holds across all three definitions. These are still large coordinate sets: the collaborator's endpoint alone needs about 93 million coordinates for 90% of its squared change. A concentration fraction does not identify the same coordinates across phases or establish their functional importance.

The [local concentration comparison](cases/local-defense-attack/displacement-concentration/coordinate-comparison.png) and [collaborator concentration comparison](cases/collaborator-attack/displacement-concentration/coordinate-comparison.png) show all three measures directly. A curve reaching 90% farther left means more concentrated energy. The upper panels use native saved values; the lower panels cast each checkpoint and anchor separately to BF16 before subtraction. This changes the endpoint/trajectory 90% fractions by less than 0.07 percentage points. The collaborator's existing external plane was fitted from native saved differences, so it appears only in the upper panel; no BF16 plane fit is implied. Both attacks use saves at 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1000. The defense uses the nine saves listed above. [Local values](cases/local-defense-attack/displacement-concentration/summary.csv) and [collaborator values](cases/collaborator-attack/displacement-concentration/summary.csv) retain every precision and measure.

The [local module comparison](cases/local-defense-attack/displacement-concentration/module-comparison.png) and [collaborator module comparison](cases/collaborator-attack/displacement-concentration/module-comparison.png) apply the same three definitions to module energy shares. Using full saved-displacement energy, local defense is 61.23% MLP, 16.41% attention, and 21.57% output head; local attack is 54.54%, 13.85%, and 30.38%, respectively. The collaborator's head share rises from 69.79% in the equal-axis plane to 84.55% for its full trajectory and 87.36% for its endpoint. This is a substantial weighting difference, but it does not change the functional replay result below: large parameter energy in the head does not imply that head changes cause most of the CE recovery.

The untied output head contains only 1.81% of parameters: its plane energy is enriched 19.0-fold in the local attack and 38.6-fold in the collaborator attack. Local attack energy is 50.65% MLP and 13.32% attention, versus 59.56% and 14.86% during defense. Their different parameter counts explain most of this difference: there is no strong MLP-specific enrichment over attention. Within the body, the last layer's MLP down projection has the largest enrichment: 2.00-fold locally and 4.41-fold for the collaborator attack. The layer plots show broad support, with weaker query/key changes and stronger late-layer changes in the collaborator case. Normalization parameters are strongly enriched per parameter but carry little total energy. [Local module and layer plots](figure-index.md) include parameter-count normalization.

**Token and hidden-coordinate structure.** The collaborator attack's input embedding carries only 0.80% of plane energy. Within that matrix, 11,635 of 32,000 token rows are needed for 90% of its energy; the top 100 rows carry 2.44%. The output head needs 24,504 rows for 90%; its top 100 carry only 0.76%. The corresponding local attack counts are 9,779 embedding rows and 24,370 head rows. This is broad vocabulary support, without evidence for a small set of biology tokens explaining escape.

The hidden dimension is also broadly involved: 90% of collaborator embedding and head energy needs 3,660 and 3,284 of 4,096 columns. The top 100 columns hold 3.59% and 4.66%, respectively. The [token audit](token-audit/report.md) also separates PC1, PC2, equal-axis energy, and eigenvalue-weighted energy. Weighting by displacement eigenvalues changes the collaborator head share to 84.9% within the top-two fit, but does not reveal a narrow token or column support.

First-step raw embedding changes initially appear to touch rows absent from the attack minibatch. For both attacks, those unseen-row changes equal the candidate conversion BF16(anchor) minus anchor exactly; subtracting that conversion leaves zero unseen-row embedding motion. This algebraic loading-offset check explains why raw coordinate support should not be read directly as minibatch gradient support. Softmax head updates can involve classes absent from the input and need a separate interpretation. [All loading checks](token-audit/loading-offset-rows.csv) include padding IDs when defining unseen rows.

**Defense-preferred versus attack-preferred directions.** The local planes have principal angles 85.01° and 85.39°. Their mean squared cosine is 0.0070. Even the full rank-eight span of the recorded defense path captures only 0.031% of the first attack displacement and 0.902% of the final displacement. This is substantially more overlap than unrelated random subspaces in billions of dimensions would have, but it is a small fraction of attack motion. [Projection and overlap plots](figure-index.md) show the entire sampled attack history.

The endpoint defense and attack displacements have cosine −0.0716. Thus the attack is largely taking another parameter route rather than undoing the defense along its endpoint direction. Neither this fact nor the plane angles prove that these other directions escaped the defense objective: movement during defense is not equivalent to the set of protected loss directions.

Two sensitivity checks preserve this result. Recasting each saved weight to the original all-BF16 evaluation representation before differencing gives plane angles 85.15° and 85.38°, and full-defense-span capture 0.033% at attack update 1 and 0.888% at update 1000. Refitting defense displacements from its ending checkpoint gives angles 84.84° and 85.38°. Fitting late defense intervals alone also gives low overlap. Changing the anchor preserves the full recorded defense span. See [anchor sensitivity](anchor-sensitivity/anchor-sensitivity.png) and its [numeric results](anchor-sensitivity/anchor-analysis.json).

**Which changes actually lower CE?** Large Euclidean energy does not imply large functional effect. We substituted selected saved attack parameters into the defended checkpoint and evaluated the original fixed ten examples and attack-held-out 32 examples, using exactly the native BF16 evaluator. Original anchors and full attack endpoints reproduce their native probe losses exactly.

| Fixed-ten-example CE decrease by attack update 1000, nats | Local K16 | Collaborator checkpoint |
|---|---:|---:|
| All saved changes | 9.289 | 11.494 |
| Output head only | 0.173 | 1.743 |
| Attention and MLP only | 9.260 | 9.994 |

At attack update 1, attention and MLP alone also reproduce nearly the full CE decrease in both models. This supports broad transformer-body adaptation as the dominant functional change on these probes. It does not support a simple readout-only or few-token patch explanation. These are frozen checkpoint composites: effects interact and need not add. They do not establish the outcome of training an attacker restricted to a module group, or partial-composite WMDP recovery. See [replay figure](module-replay/module-only-ce-replay.png), [all values](module-replay/losses.csv), and [native-loss validation](module-replay/validation.json).

**Implication for the next runs.** The useful follow-up is to measure how retain weight, simulated-attack LR, and outer defense LR change both loss geometry and the attack route, keeping data/order fixed. The present analysis rules out some simple explanations, but cannot distinguish objective misspecification from inadequate simulated-attack coverage by itself. The new study compares each single change with the existing local baseline, then all three jointly; it does not identify every interaction or reproduce the collaborator's different legacy data sampler.

Both primary analyses and all three auxiliary checks completed. Seven focused basis tests, 23 compute/preparation tests, and 41 comparison/report tests passed; the auxiliary anchor and token self-checks also passed. Numeric provenance, frozen execution sources, input identities, and job records are retained alongside this report. Checkpoint equality at the local defense-to-attack bridge was exact. The local attack Gram matrix matched the original native fit. No hyperparameter training was launched before these results were complete and interpreted.

The endpoint/trajectory addition completed on 2026-11-18 using existing checkpoints. Ten focused tests passed, including a dense synthetic integration check against exact coordinate sorting. Newly streamed endpoint energies matched the original scans; full trajectory energies matched the original Gram traces; local trajectory module shares matched the prior per-parameter Gram analysis. [Addition job records](concentration-jobs.json), per-case validation files, frozen sources, and the refreshed artifact manifest preserve this computation. This addition did not rerun training or attacks.
