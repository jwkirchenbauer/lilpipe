# RIO K comparison on CUDA

This follow-up adds CUDA K16 and K64 defense/attack runs to the completed CUDA
K4 and K256 runs. It addresses the material backend variation measured in the
[original completed study](../../tar-rio/20261120T2120Z/report.md).

All four conditions share numerical source, environment, data order, probes and
plotting directions. Only K and its required inner epochs differ in the defense
recipe. Each defense has 125 outer updates; each attack has 1,000 updates at 1e-6.
The legacy simulator behavior and the utility gate remain part of the analysis.
This is one paired seed.

Completed on 2026-11-21 UTC. All eleven declared phases and strict validations
passed. Start with [results and key figures](report.md), or read the
[full report](rio-cuda-report.md). Recovery is not monotone in K in this seed,
and every defender misses the initial MMLU utility gate.

See [protocol](protocol.json),
[configuration/data-order audit](ops/comparability-audit.md),
[final accounting](status/final-accounting.json), and
[execution history](OPERATOR_STATE.md).
