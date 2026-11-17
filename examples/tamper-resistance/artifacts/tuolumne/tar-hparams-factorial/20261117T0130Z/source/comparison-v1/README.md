# Controlled eight-cell static comparison

This profile combines B/R/I/O/RIO from the completed main hyperparameter study with RI/RO/IO from this defense-only extension. It reads JSON/CSV metadata and cached grids. It loads no model, dataset, or basis tensors and requires no attack or benchmark results. The uploaded HF checkpoint remains a separate contextual comparison in the main study.

The three binary factors are retain coefficient R (2.5→4), inner learning rate I (1e-6→1e-5), and outer learning rate O (5e-6→1e-5). The figure layout is B/R/I/RI on the top row and O/RO/IO/RIO on the bottom row. Every panel states all three actual values.

`inputs.json` fixes the eight sources, prior completed artifact hashes, and legacy B model file identities. `design.json` records the metric definitions before reading any new pair-control grid. The frozen wrapper checks its manifest and validates all eight cases before rendering.

For each 9×9 grid at half-width r:

- Center CE is L[4,4].
- Rim rise is the largest CE among the eight immediate neighbors of index(4,4), minus center CE. Neighbors have coordinate spacing r/4.
- Center elevation is center CE minus the mean CE of the 32 perimeter cells.

Negative values remain negative. These are finite-grid diagnostics, without a global barrier or curvature interpretation. Shared absolute z/color limits and camera apply across all eight cells within each radius; center-relative contour colors use one symmetric scale per radius. All eight radii are reported, with 125 and 100 linked first. The recipe is the exact original seed 42 BF16 random-direction/default-attention convention, separate from the saved FP32 orthonormal dynamic bases.

For each metric, f(r,i,o) is its value at the three binary factor settings. The anchored coefficients form the exact multilinear expansion:

```
f(r,i,o) = f(0,0,0) + c_R*r + c_I*i + c_O*o
           + c_RI*r*i + c_RO*r*o + c_IO*i*o + c_RIO*r*i*o
```

Coefficients are unscaled differences around the other factors set to 0. For example, c_RI=f(1,1,0)−f(1,0,0)−f(0,1,0)+f(0,0,0). The three-way coefficient is f111−f110−f101−f011+f100+f010+f001−f000. All conditional main differences and conditional pair differences are retained. Marginal main contrasts average four conditional differences; marginal pair contrasts average two. The CSV normalization column records these divisors explicitly. One seed and no replication support descriptive differences only; no p-values are computed.

The comparison requires matching get_ll source hashes, Torch/HIP, seed 42, fixed texts/revision, token batch hashes, all eight radii and all grid points. CSV coordinates are recovered to the original FP32 execution coefficients; their bits must match exactly. No interpolation or loss adjustment is performed. A coordinate change of one FP32 unit in the last place is rejected. Each new defense must have completed training, a passed 125-batch exact-order guard against B, a complete static status, and valid source/config/protocol/checkpoint identities. B uses its existing passed legacy 125/K16 proof plus the recorded defense's exact metric/endpoint equality; its current model file identities are bound in the input specification. The guarded static workers explicitly establish tokenizer/collation equality to B.

The CLI is `factorial_static_report.py validate RUN` for metadata/CSV validation, or `factorial_static_report.py report RUN` for final rendering. The latter requires an allocated launcher job. Submit the frozen `run.sh RUN` entry only after all eight numerical caches complete. It starts separate validation and rendering processes with empty GPU visibility and a fresh Matplotlib cache. `dryrun.json` records the reviewed 10-minute pdebug/verival launcher command; preparation does not submit it.

Outputs will be `RUN/static/full-factorial/{report.md,grid-metrics.csv,factorial-contrasts.csv,validation.json}` and 16 PNG/PDF figure pairs. Its own completion marker is `RUN/status/static-factorial-report.json`; it does not mark the whole experiment complete.
