# RIO K and simulation-fidelity results

The training, attacks, evaluations, landscapes and early-update profiles are
complete. The strict final report passed every declared measurement and
provenance check.

- [Full report and result table](rio-report.md)
- [Four-way K trajectories](rio-figures/rio-k-four-way-trajectories.png)
- [Four-way K landscapes](rio-figures/rio-k-four-way-broad-random.png)
- [Data and optimizer comparison](rio-figures/rio-data-optimizer-2x2-trajectories.png)
- [Final references and runtime provenance](report-matrix-references.json)
- [Geometry reuse validation](geometry-reuse-validation.json)
- [Final figure rendering and unchanged-input validation](plot-rendering-validation.json)

Interpret the K comparison with its utility and backend controls: all four K
conditions fail the initial five-percentage-point MMLU utility gate. The paired
K4 training-backend control differs by up to 5.76 percentage points in WMDP,
so the CUDA-trained K256 comparison carries a material backend uncertainty.
These are single paired-seed measurements. The full report also distinguishes
the legacy simulator from the matched update implementation.
