# Initial five-cell defense endpoint screen

The joint published hyperparameters change the random-plane geometry toward the collaborator's raised-rim shape. The single changes do not reproduce that joint geometry. This points to an interaction, rather than identifying retain weight alone as the explanation.

All five controlled cells share the original local sampler, 125 defense updates with 16 simulated steps each, seeds, optimizer/precision settings, and exact observed batch contents/order. The uploaded checkpoint is an additional context comparison; its legacy sampling and unpublished training implementation remain outside this controlled screen.

| Condition | Retain coefficient | Inner LR | Outer LR | End-defense fixed-ten CE, nats |
|---|---:|---:|---:|---:|
| B | 2.5 | 1e-6 | 5e-6 | 11.2384 |
| R | 4 | 1e-6 | 5e-6 | 10.9572 |
| I | 2.5 | 1e-5 | 5e-6 | 6.9325 |
| O | 2.5 | 1e-6 | 1e-5 | 11.2568 |
| RIO | 4 | 1e-5 | 1e-5 | 14.4114 |

At half-widths 100–125, retain weight and outer LR changes individually preserve a fairly flat raised center. Raising the inner LR alone produces a lower central bowl. Changing all three produces a higher center with a raised rim around a shallow central depression. The uploaded model has the same broad qualitative feature but a stronger rim in these sampled views. See the [matched absolute-CE view](radius-125-absolute.png), [center-relative view](radius-125-delta.png), and [all eight radii](report.md).

These effects are strongly non-additive. The additive prediction for joint center CE from the three single changes is R + I + O − 2B = 6.6697 nats; the measured joint value is 14.4114, a 7.7417-nat residual. This is a descriptive contrast from one training seed, without an uncertainty estimate. It does not establish that all three changes are necessary. The subsequent pairwise endpoint measurements resolve that question, as linked below. The rim is a visual feature of an asymmetric sampled surface; the defended point need not be a local minimum, and a high neighboring point does not establish a closed barrier.

The comparison uses the original static convention: the same seed-42 BF16 random directions, sequential BF16 perturbations, default attention, ten fixed documents, batch size two, and truncation 512. Each radius shares CE/color/camera limits across checkpoints. Center-relative figures subtract each checkpoint's measured center. These are finite two-dimensional slices, without a claim about a full-space barrier or attack resistance. Standard attack dynamics are evaluated separately using the existing FP32 orthonormal direction convention and eager attention.

All 648 grid points per checkpoint passed validation. The comparison recovered identical FP32 coordinate bits despite CSV decimal differences of at most 6.25e-17; the repair reused the raw loss grids without resampling. [Validation](validation.json) records the measured inputs and output hashes. The [completed eight-condition comparison](../../../../tar-hparams-factorial/20261117T0130Z/static/full-factorial/findings.md) now resolves the interaction: increasing both learning rates produces the raised rim without increasing retain, and retain changes its height and depth. Attack and benchmark claims for those three additional pair cells remain outside this endpoint screen.
