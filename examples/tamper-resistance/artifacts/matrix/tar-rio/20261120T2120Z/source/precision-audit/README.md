# Matrix RIO saved-tensor precision audit

This is a CUDA-port of the original read-only saved-tensor audit. It uses
bounded 1,000,000-element chunks and writes only `results/precision-audit.json`
and `.csv`. It does not load a model, run inference, or train.

The forward-matched CUDA parity run must finish first. The valid native input
is the exact original forward-matched defender root file used by the pilot's
`base_model`; the anchor is the saved post-reload CUDA state from that pilot.
The audit also needs the five simulated states (updates 1, 2, 4, 8, and 16) and
the archived AMD K16 checkpoints. Matrix parity must have `state=complete`,
exact `simulated_vs_actual` equality with zero difference at every checked
state, and exactly 16 zero actual-loss errors. Those same-backend requirements
remain strict.

Two anchor families are reported with explicit backend fields in each CSV row
and in JSON provenance:

- `archived_rio_v2`: archived AMD anchor, AMD simulated states, and AMD archive
  checkpoints. This is a historical same-backend reference pair, re-analyzed
  with the CUDA audit environment.
- `forward_matched_new_anchor`: fresh CUDA anchor and simulation states, with
  archived AMD checkpoints as reference. The checkpoint comparison is
  cross-backend, so floating-point differences are expected and diagnostic;
  it does not gate same-backend parity.

Measures remain separated by saved tensor dtype and module family:

- `native_bf16_roundtrip`: native defender versus BF16 cast-back values.
- `native_to_reload`: native defender versus the saved post-reload anchor, with
  relative drift normalized by the simulated attack update at states 1 and 16.
- `straight_through_reconstruction`: saved-dtype `anchor + (endpoint-anchor)`
  reconstruction error at each checked simulated endpoint.
- `reference_vs_simulated`: archived reference state versus the simulated
  endpoint. Backend fields identify whether this is AMD-versus-AMD or
  CUDA-versus-AMD.

The straight-through measure is arithmetic at the saved anchor. It does not
reconstruct native training-time inner endpoints or run a model forward pass.
The output provenance records the CUDA analysis device, package versions, file
paths, sizes, mtimes, source hashes, and the validation's complete reference
comparison metrics for every checked step.
