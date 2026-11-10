# TAR reproduction plan for JWK

Run the collaborator's named configurations: one Zephyr-7B tamper-resistant
unlearning (TAR) defense, followed by three independent 128-step relearning
attacks at learning rates `1e-5`, `1e-6`, and `5e-6`. Each attack starts from the
same defense checkpoint. Run `1e-5` first, then the two additional rates.

All config paths below are relative to `examples/tamper-resistance`:

| Stage | Directory containing `selected.yml` |
|---|---|
| Defense | `configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16` |
| Attack, `1e-5` | `configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr1e-5-ms128` |
| Attack, `1e-6` | `configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr1e-6-ms128` |
| Attack, `5e-6` | `configs/stages-wmdp/2_relearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft-lr5e-6-ms128` |

1. **Preserve the experiment settings.** The defense runs 125 steps with learning
   rate `5e-6`, retain coefficient `2.5`, and seed/data seed `42`. TAR evaluates
   maximum-entropy unlearning after a simulated attack; `k16` means 16 simulated
   attack steps per defense step, with inner learning rate `1e-6`. External
   attacks use the named configs' `data.wmdp_zephyr` loader, batch size 1,
   sequence length 512, seed `1`, data seed `42`, and checkpoints every 16 steps.
   Change only runtime paths, logging destinations, and pinned input revisions.

2. **Use JWK's Tuolumne workflow.** Use the separate locked ROCm environments at
   `/usr/workspace/wsb/kirchenb/lilpipe-tar-train-rocm72-v4` and
   `/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2`.
   They preserve Axolotl `0.18.0`, lm-eval `0.4.11`, Transformers `5.14.1`, and
   PyTorch `2.12.1` with ROCm `7.2`. Every Python command follows absolute-prefix
   `mm_activate` inside interactive Bash with `set -e`. Launch compute through
   `/p/vast1/kirchenb/llnl-tools/launch_tuo.py`, bank `verival`, one node and four
   GPUs. Training uses one process with the configs' four-GPU layer map.
   Keep `torchrun=false`. Set `TORCH_BLAS_PREFER_CUBLASLT=0` (and its HIP alias)
   to use hipBLAS/rocBLAS after observing a hipBLASLt backward stall. Log and
   verify the original and copied model's layer placement and first batch devices.

3. **Validate before the full run.** Cache gated data and pinned Zephyr revision
   `892b3d7a7b1cf10c7a701c60881cd93df615734c` under `tar-root`. Verify all four
   MI300A GPUs, review launcher dry-runs, and complete a separate one-step defense
   test. Start the full defense fresh from Zephyr. Check untouched-model benchmark
   scores before training; investigate a mismatch above one percentage point.

4. **Evaluate each trajectory.** Use repository task definitions for
   `wmdp_bio_robust` (biology knowledge; lower is better) and `mmlu_no_bio`
   (Massive Multitask Language Understanding with biology tasks excluded; higher
   is better). Preserve zero-shot evaluation, BF16, batch size 32, four evaluation
   processes, and no sample limit. Evaluate untouched Zephyr, the shared defense
   as attack step 0, and steps `16,32,48,64,80,96,112,128` for each attack:
   26 model states in total.

5. **Save a comparison-ready result.** Preserve configs, source checksums,
   dependency locks, input revisions, checkpoints, and raw evaluation JSONs.
   Produce a CSV and a two-panel figure showing both metrics against attack step
   for all three learning rates. Report available matching reference scores
   explicitly: untouched Zephyr `55.18% / 58.59%` and the selected defense
   `31.34% / 53.45%` (WMDP / MMLU). The selected defense's documented capability
   drop is 5.14 percentage points, promoted by exception to the nominal `5pp`
   budget. Share the named configs and measured attack trajectories for direct
   comparison with the collaborator.

Execution artifacts live under
`examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/`.

Completed on 2026-11-09: Flux job `f3j8PtnfdGXy` finished all training and all 26
evaluations. The defense scored WMDP `33.06%` / MMLU `54.02%`, versus the
collaborator's `31.34%` / `53.45%`. All device checks, training metrics, config
comparisons, and evaluation checks passed. Relearning recovered most or all of
the measured WMDP reduction.

[Results and reference comparison](examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/report.md)
· [Trajectory figure](examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/trajectory.pdf)
· [All scores](examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/comparison.csv).

The same K16 defense and `1e-6` attack were repeated with intermediate checkpoints
to study how the attacker leaves the defense.
[Landscape dynamics, figures, and movies](examples/tamper-resistance/artifacts/tuolumne/tar-k16-repro/20261109T1830Z/dynamics/baseline-k16-20261110T0400Z/report.md)
records the baseline results. The follow-up compute and intervention experiments
are described in the [compute-extension plan](TAR_COMPUTE_EXTENSION.md).
