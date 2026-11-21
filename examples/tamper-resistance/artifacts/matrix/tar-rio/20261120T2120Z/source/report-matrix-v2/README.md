# Final Matrix report: geometry reuse

This version reconciles the original K4/K16 geometry protocol with the Matrix
score-output path and CPU anchor storage option. The geometry implementation
does not read either option. The adapter runs the original validator with
those saved operational values and records the differences in the report.

All other settings, protocol hashes, input identities, stage completion,
accumulators, fitted basis identities and measured losses retain their original
checks. The original geometry and progress files stay byte-identical. Fresh
CUDA scores and current profile settings use their existing strict checks.

The final report records the adapter/worker hashes and writes
`geometry-reuse-validation.json` alongside `report-matrix-references.json`.
