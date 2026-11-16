# TAR attack and defense basis structure

Completed analysis: [report and figures](examples/tamper-resistance/artifacts/tuolumne/tar-structure/20261116T2200Z/report.md).

The fitted attack directions are concentrated, but spread over hundreds of millions of parameters and broad token support. The local defense and attack planes are nearly orthogonal. Frozen module replay shows that attention and MLP changes account for most of the measured forget-set CE decrease, despite the output head's large Euclidean energy share. The report separates these observations from causal claims about what the defense protects.

This builds on the [reproduction](TAR_REPRODUCTION_PLAN.md), [landscape/compute study](TAR_COMPUTE_EXTENSION.md), and [collaborator checkpoint comparison](TAR_COLLABORATOR_CHECKPOINT.md).
