# Where the attack moves, and how it differs from defense

The fitted attack plane is concentrated relative to isotropic random directions, but its energy still covers hundreds of millions of coordinates. It is not localized to a few embedding tokens. More importantly, the local attack follows directions almost orthogonal to the observed defense trajectory. Replaying saved changes shows that the transformer body produces most of the fixed and held-out probe cross-entropy (CE) decrease, despite the output head's large share of Euclidean basis energy.

This analysis uses the completed local K16 defense and its standard 1000-update LR1e-6 attack, plus the same attack on the collaborator's uploaded checkpoint. No new defense training was needed. The collaborator checkpoint has no defense history, so its defense-preferred basis cannot be fitted. See the [original reproduction plan](../../../../../../TAR_REPRODUCTION_PLAN.md) and [checkpoint comparison](../../../../../../TAR_COLLABORATOR_CHECKPOINT.md).

**What the basis measures.** Each phase uses an uncentered principal-component fit to saved displacements from that phase's starting checkpoint, matching our existing landscape analysis. The defense has nine saves at updates 1, 2, 4, 8, 16, 32, 64, 96, 125; its first displacement is zero because the first learning rate is zero. The attack fit uses 13 saves through update 1000. Two defense axes capture 98.18% of summed squared saved displacement; two attack axes capture 95.80%. These are sampled path fits, with no centering or correction for uneven checkpoint spacing. They describe the realized optimizer path rather than the gradient covariance, Hessian, or all directions the defense protects.

For a two-axis orthonormal basis, coordinate energy is the sum of squared entries across both axes, divided by two. This is invariant to rotations within the plane. It gives each axis equal weight; it is distinct from eigenvalue-weighted displacement energy. Both definitions are preserved in the numeric outputs. Gram matrices accumulate in FP64; plane coordinates are reconstructed in FP32.

**Parameter concentration.** Fractions below refer to all 7.24 billion parameters. Coverage bounds come from energy histograms, rather than exact global sorting.

| Plane | Coordinates needed for 90% of energy | Exact nonzero coordinates in saved-weight fit | Output-head share of plane energy |
|---|---:|---:|---:|
| Local defense | 13.884–13.904% | 27.94% | 24.78% |
| Local attack | 5.310–5.312% | 13.68% | 34.36% |
| Collaborator attack | 2.393–2.395% | 14.54% | 69.79% |

For comparison, an isotropic Gaussian two-dimensional plane needs about 59% of coordinates for 90% of energy. The attack planes are therefore compressible, but even the collaborator's 2.39% means about 173 million coordinates. Exact zeros are a property of these serialized checkpoints and their precision, not proof that the underlying gradients were zero.

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
