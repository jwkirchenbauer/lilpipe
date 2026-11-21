# RIO K comparison: completed CUDA controls

All four conditions use CUDA training, the same frozen source, 125 outer
defense updates, and the standard 1000-update attack at LR 1e-6. This adds the
missing K16/K64 controls to the completed CUDA K4/K256 runs. All eleven new
phases and strict report checks passed on 2026-11-21 UTC.

Start with these figures:

- [Attack trajectories and benchmark recovery](figures/fourway-trajectories.png):
  held-out CE falls in every condition, but benchmark recovery is not monotone
  in K. K16 has the lowest final WMDP score, alongside declining attack MMLU.
- [Broad random-plane landscapes](figures/broad-random.png): columns are K values;
  rows are attack steps 0, 128, and 1000. The physical directions, radius, camera,
  and absolute CE/color scale are shared. Higher K produces a much taller central
  region at defense completion in these slices, which falls during attack.
- [Center-relative random landscapes](figures/broad-delta.png): subtracting each
  panel's central CE separates shape from the changing central height.
- [Local reference-plane landscapes](figures/local-reference.png): a narrow view
  in the same fixed directions across cases. These reference directions come
  from the early baseline, rather than any RIO defender or each attack's fit.

The [full results table and per-condition figures](rio-cuda-report.md) link each
defense progression, fitted attack plane, actual versus projected attack path,
and early-update profile. The [original report](../../tar-rio/20261120T2120Z/report.md)
contains the separate data/order and simulator-fidelity experiments.

Every defender fails the initial MMLU gate of 53.5798%. Later crossings of 45%
WMDP occur at steps 384 and 512 for K64/K256, but their initial WMDP scores differ,
and K16 never reaches that threshold. These are single-seed trajectories, not
evidence of a systematic robustness gain under the utility constraint. Increasing
K also increases compute and repeated exposure to the same 16-example simulation
pool; total defense compute is not held fixed.

The K sweep preserves the legacy simulator's enclosing mixed-precision cache,
which failed multi-update trajectory parity in the separate pilot. The repaired
matched simulator passed that tested prefix; it was not substituted into these
K runs. See the [configuration/data-order audit](ops/comparability-audit.md),
[provenance](report-provenance.json), and [final accounting](status/final-accounting.json).
