# Figure index

Start with the [findings and measured results](report.md).

The matched comparison figures place B, R, I, O, and RIO together. Each figure shares CE/color limits and a camera; random and reference planes also share physical directions and bounds. The own-attack panel uses each condition's fitted directions and explicitly labeled radius, so its horizontal axes are not a common parameter plane.

| View | PNG | PDF |
|---|---|---|
| End of defense, fixed random slice | [PNG](aggregate-comparison-v2/final/figures/end-defense-fixed-random.png) | [PDF](aggregate-comparison-v2/final/figures/end-defense-fixed-random.pdf) |
| End of defense, shared attack-reference slice | [PNG](aggregate-comparison-v2/final/figures/end-defense-fixed-reference.png) | [PDF](aggregate-comparison-v2/final/figures/end-defense-fixed-reference.pdf) |
| End of defense, each condition's fitted attack slice | [PNG](aggregate-comparison-v2/final/figures/end-defense-fixed-attack.png) | [PDF](aggregate-comparison-v2/final/figures/end-defense-fixed-attack.pdf) |
| Broad random slices at attack 0/128/1000, radius 125 | [PNG](aggregate-comparison-v2/final/figures/broad-random-r125.png) | [PDF](aggregate-comparison-v2/final/figures/broad-random-r125.pdf) |
| Shared local reference at attack 0/1/8/32/128/512/1000 | [PNG](aggregate-comparison-v2/final/figures/local-reference-matched-attack.png) | [PDF](aggregate-comparison-v2/final/figures/local-reference-matched-attack.pdf) |
| Shared local reference at defense 0/8/32/125 | [PNG](aggregate-comparison-v2/final/figures/defense-history-local-reference.png) | [PDF](aggregate-comparison-v2/final/figures/defense-history-local-reference.pdf) |

The [paired comparison report](checkpoint-report.md) provides larger B-versus-condition 3D/contour views, center-relative views, and capability/actual-loss/projection trajectories. The [presentation provenance](checkpoint-comparisons/layout-v2-publication.json) verifies that wrapping the titles changed no plotted numerical data. Earlier layouts remain archived for provenance; the links here use the corrected layouts.

The joint recipe's native set is complete: [broad random-plane defense/attack progression](cases/hparam-rio-lr1e-6/random-landscape-evolution.png), [fixed slices and actual checkpoints](cases/hparam-rio-lr1e-6/surfaces-and-checkpoints.png), [projection fidelity](cases/hparam-rio-lr1e-6/projection-fidelity.png), and [minibatch dynamics](cases/hparam-rio-lr1e-6/minibatch-dynamics.png). Reference denotes the shared plane fitted to the original K16 128-update attack; attack denotes this condition's own 1000-update path fit. Actual-loss dots can lie off the projected surface; inspect fidelity before reading the projected path as an actual loss trajectory.

The controlled eight-condition endpoint view isolates the hyperparameter interaction using one physical random plane and shared scales within each radius:

- [All radii, definitions and contrasts](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/report.md).
- [Radius 125: absolute CE](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/radius-125-absolute.png) / [PDF](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/radius-125-absolute.pdf).
- [Radius 125: center-relative CE](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/radius-125-delta.png) / [PDF](../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/radius-125-delta.pdf).
- [Five controlled cells plus the uploaded checkpoint](static/comparison/report.md), for the original collaborator-like tower context.

The first-update plane uses the actual saved update direction and a residualized random direction. The update axis spans +/-1 raw L2 unit; the broad random axis spans +/-125. The two axes therefore have different displayed scales. Segment curves evaluate fixed and actual-batch CE between saved optimizer states, with shaded interpolation and labeled extrapolation.

| Condition | First-update plane | Eight update segments |
|---|---|---|
| R | [PNG](cases/hparam-r-lr1e-6/first-update-landscape.png) / [PDF](cases/hparam-r-lr1e-6/first-update-landscape.pdf) | [PNG](cases/hparam-r-lr1e-6/early-update-profiles.png) / [PDF](cases/hparam-r-lr1e-6/early-update-profiles.pdf) |
| I | [PNG](cases/hparam-i-lr1e-6/first-update-landscape.png) / [PDF](cases/hparam-i-lr1e-6/first-update-landscape.pdf) | [PNG](cases/hparam-i-lr1e-6/early-update-profiles.png) / [PDF](cases/hparam-i-lr1e-6/early-update-profiles.pdf) |
| O | [PNG](cases/hparam-o-lr1e-6/first-update-landscape.png) / [PDF](cases/hparam-o-lr1e-6/first-update-landscape.pdf) | [PNG](cases/hparam-o-lr1e-6/early-update-profiles.png) / [PDF](cases/hparam-o-lr1e-6/early-update-profiles.pdf) |
| RIO | [PNG](cases/hparam-rio-lr1e-6/first-update-landscape.png) / [PDF](cases/hparam-rio-lr1e-6/first-update-landscape.pdf) | [PNG](cases/hparam-rio-lr1e-6/early-update-profiles.png) / [PDF](cases/hparam-rio-lr1e-6/early-update-profiles.pdf) |

The standard attack's [complete benchmark tables](evaluation-report.md), [all milestone scores](evaluation-scores.csv), and [highest measured scores through each budget](max-observed-wmdp.csv) cover all four new attacks. Defense endpoints are attack step zero.

The small CE matrices behind the minibatch-dynamics views retain their model-step and minibatch-step axes: [R](cases/hparam-r-lr1e-6/surfaces/minibatch-losses.npz), [I](cases/hparam-i-lr1e-6/surfaces/minibatch-losses.npz), [O](cases/hparam-o-lr1e-6/surfaces/minibatch-losses.npz), and [RIO](cases/hparam-rio-lr1e-6/surfaces/minibatch-losses.npz). These contain measured losses, rather than model or basis tensors.

The prerequisite [basis-structure figures](../../tar-structure/20261116T2200Z/figure-index.md) cover coordinate/module/token concentration, defense/attack overlap, the loading-cast audit and frozen module replay. The [basis report](../../tar-structure/20261116T2200Z/report.md) defines these measurements and their limits.
