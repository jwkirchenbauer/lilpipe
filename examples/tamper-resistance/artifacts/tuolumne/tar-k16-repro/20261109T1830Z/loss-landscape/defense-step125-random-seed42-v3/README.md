# TAR loss landscape: revised radii

The end-of-defense checkpoint is unchanged. The eight-panel plot now uses
**1000, 500, 200, 125, 100, 10, 1, and 0.1**. Radii 10 and 1 replace 0.01 and 0.001.

- [Updated 3D overview](overview/landscape-3d.png) · [PDF](overview/landscape-3d.pdf)
- [Focused radius-125 view](radius-125/landscape-3d.png) · [PDF](radius-125/landscape-3d.pdf)
- [Combined grid](grid.csv) · [New measurements](new-radii/grid.csv)
- [Settings and source hashes](provenance.json) · [Validation](validation.json)
- [Comparison with a shared vertical scale](centering-check/shared-scale.png) · [PDF](centering-check/shared-scale.pdf)

We computed 162 new points and reused 486 points from the
[previous measurement](../defense-step125-random-seed42-v2/README.md).
Each radius has a 9 × 9 grid. The same two random directions use seed 42 and
global L2 normalization in BF16. The checkpoint, ten forget documents,
512-token limit, batch size 2, and padding-label masking match the previous run.
Compatibility checks covered the data hashes, measurement code, and runtime.

All 648 losses are finite and the repeated center losses agree at 11.2384 nats.
The center checkpoint projects to (0, 0). The surface and colors use a separate
vertical scale for each panel. Flux allocation `f3jAy9mDVeUF` completed the new
measurements and plotting successfully in about four minutes.

The apparent slope at radius 10 comes from a small local trend magnified by
independent vertical scaling. Its loss span is 0.0219 nats, versus 2.3488 at
radius 100. A separate [centering check](centering-check/centering-check.json)
recomputed 18 points, including 16 nonzero coordinates shared with the saved
radius-100 and radius-10 grids; all losses matched exactly. Every grid's center
loss is 11.238433265686036. A plane explains 98.6% of the radius-10 variation.
