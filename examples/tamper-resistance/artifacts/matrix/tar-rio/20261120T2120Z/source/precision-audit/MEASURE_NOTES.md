# Reading the precision measures

The submitted code and its original README remain unchanged for provenance.
This note clarifies the roundtrip field's meaning.

- `native_to_reload` is the loaded anchor minus the exact native file supplied
  to the loader.
- `native_bf16_roundtrip` is the loaded anchor minus native values cast through
  BF16 and back to the native dtype. Zero difference proves that this cast
  explains the reload. It does not mean the native parameters were unchanged.
- `straight_through_reconstruction` compares saved-dtype
  `anchor + (endpoint - anchor)` with the saved simulated endpoint. The exact
  replay check makes the simulated and actual updates interchangeable here.

Raw displacements measured from a native defense file include the reload
offset. The first optimizer update is measured from the loaded anchor instead.
Keep this distinction when interpreting early trajectory directions and
coordinate concentration. These saved-tensor calculations contain no forward
passes; their norms alone establish neither a CE change nor a robustness effect.
