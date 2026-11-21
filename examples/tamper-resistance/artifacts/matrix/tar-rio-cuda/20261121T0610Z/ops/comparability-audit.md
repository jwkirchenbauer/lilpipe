# RIO CUDA K16/K64 comparability audit

Read-only audit of the archived RIO K16 control and the fresh CUDA K16/K64 configs staged under this run. No training, evaluation, or source files were changed for this audit.

## Archived K16 versus fresh CUDA K16

The archived control is `artifacts/tuolumne/tar-hparams/20261116T2310Z/cases/hparam-rio-defense` plus `hparam-rio-lr1e-6`. Its defense config has `attack_epochs: 1` and omits `attack_max_steps`; the archived trainer default is 16. The staged CUDA K16 config makes that cap explicit at 16. Both use the same defense recipe: Zephyr snapshot `892b3d7a7b1cf10c7a701c60881cd93df615734c`; forget/retain coefficients 1/4; defense forget/retain batch sizes 4/4; inner forget batch size 16 with per-iteration size 1; inner LR 1e-5; 125 outer updates; AdamW at 1e-5, zero weight decay, linear schedule, 12 warmup updates; BF16, TF32 off; seeds 42/42; and the same WMDP and WikiText revisions. The fresh K16 differs from that archived control in backend/runtime and config/code provenance, plus explicit K metadata and run-local paths/names. The prepared cache directory changes from the hparam run to the RIO run; source revisions and observed batch order are held fixed and checked below.

The archived training source and current Matrix source are not byte-identical. The current source adds replay/optimizer options and an explicit inner-step cap. The prepared primary K runs use `RecordedTARMETrainer`, with no replay trace or matched optimizer options, so the new options are inactive; the current `adamw_attack` legacy defaults retain AdamW's standard defaults. Treat this as source provenance to record, not as proof of bitwise equivalence.

| Source file | Archived K16 SHA-256 | Current Matrix SHA-256 |
|---|---|---|
| `source/misc/tar_dynamics_training.py` | `8fbfd35bd251cbc831bb6dc2fdfdb9183caa33983993b2430d27e3cf858ed9a9` | `de44ad18b93267f41761aef0a7a53bb70b3320ff6d325ff1532ff684ca4b90a9` |
| `source/src/trainers/tr_tar.py` | `17af0ecc0a164edf2547053a0df249854fda3e4cfe18e4882f06cd3b9f585188` | `58fa7e4fb677e1a62d4088146798fdeaa084a93865f260edee8c86a1347fb069` |

Config SHA-256 values pin the exact inputs reviewed:

| Config | SHA-256 |
|---|---|
| Archived `hparam-rio-defense/config.yml` | `10211244bdf197473f16efa59fbd30a5b165362ec504f4ddbb91a5d3bba1997e` |
| Matrix K4 CUDA defense template | `2d0844ce90236e18afe4dcd5a29ff6dfca3a5d1ca56dd62112152c2b10dba879` |
| Archived `hparam-rio-lr1e-6/config.yml` | `e1a228a247c56be79a53d5286e6c3f32d47d42b5639ba6eaa8be4b44e3cf3853` |
| Matrix K4 CUDA attack template | `998f74adc6993736c708c780a4c04fe01ba90aa31d1f9ca6aecc00df6eab5074` |
| Staged CUDA K16 defense | `deca874c1ec84aa923e4e13f931073bd61ff95f755bf3d94825cda479930a191` |
| Staged CUDA K16 attack | `f1bffded2ae5271246c4e3935bff209a75da119b6cd921b679718a58d833608c` |
| Staged CUDA K64 defense | `32cc7909f0d310648707c55adaca1f6951f1e0141809aadb2ab924b9e6c4c522` |
| Staged CUDA K64 attack | `3ea15595ac8a6b9d117f98c6a2a460b9c9874031dc4191dd5177b51249a99228` |

## K4-derived CUDA settings

The staged K16/K64 defense configs were compared directly to the successful Matrix K4 CUDA config. Outside `output_dir`, `wandb_name`, and the K-specific inner settings, all fields match. K16 has `attack_epochs=1`, `attack_max_steps=16`; K64 has `attack_epochs=4`, `attack_max_steps=64`. Both retain the complete 125-update defense schedule and state saves at 32/64/125. This is consistent with the original K16 default and the existing K64 config (`attack_epochs=4`, cap 64).

Each attack config was compared to the Matrix K4 CUDA attack config. Differences are only the fresh matching CUDA defense as `base_model`, the unique attack `output_dir`, and `wandb_name`. Both retain the full 1000-step attack, LR 1e-6, AdamW, zero weight decay, linear scheduler, no warmup, BF16, TF32 off, seed 1/data seed 42, and optimizer-state saves at 128/512/1000. The `case.json` records the fresh K-specific defense path and K value; the old AMD cases remain separate.

## Data order, probes, and directions

The old K16 and Matrix CUDA K4 `observed-batches.jsonl` files were compared by their ordered per-update `batch_sha256` values. Both defense sequences contain 125 updates and have sequence digest `50a8b55c0d86c856dcf44905434151d2be090ac0b86da7b7cf52f88b3634ec72`; both attack sequences contain 1000 updates and have digest `7d7aa4490c442b247642a36c7dd7fb26273b8fd12a95cd9a88a7978766d1512a`. This direct sequence comparison avoids comparing `training-validation.json` summary hashes, which use different aggregation formats across the two study kinds. The staged K16/K64 use the same RIO prepared-cache paths and seeds as the K4 CUDA template; their full observed batch sequences should be checked against these K4 CUDA sequences after completion.

The staged root reuses the fixed probe set: forget revision `5a786ed1041e56fef2d4ed26c1239f12b73a68eb`, fixed source indices 0–9, and the same 32 held-out indices in `probes.json`. Keep those probe and held-out identities unchanged. Keep the shared random basis, shared reference basis, direction seed 42, local radius, and broad radius shared across all four K conditions. The shared reference basis comes from `tar-k16-repro/dynamics/baseline-k16-20261110T0400Z`; that baseline used retain coefficient 2.5 and defense LR 5e-6, so it is a fixed common plotting direction source, not the RIO K16 control or any fresh CUDA K16 defender. Each new geometry must use its own fresh CUDA defense as `attack_anchor` and fresh CUDA attack checkpoints, while preserving the same shared axes.

For the primary K comparison, retain fresh attack batches and the legacy optimizer path. Do not enable `RecordedReplayTARMETrainer`, reverse order, replay traces, or matched optimizer mode; those are separate fidelity conditions. If `DYNAMICS_SOURCE_MANIFEST` is enabled, give each training run a unique output manifest path; it is written by the trainer. Leave `DYNAMICS_BOOTSTRAP_SEED` unset.

## Checkpoint reload and backend scope

Train each defense from the same base snapshot, then point its attack config at that run's saved CUDA defense checkpoint. Preserve the checkpoint's native stored dtype as provenance, and use the same standard BF16 reload and analysis arithmetic across the CUDA cases; distinguish native/saved dtype from the loaded dtype in any precision record. Do not replace fresh CUDA checkpoints with archived ROCm weights. Keep saved-checkpoint reload measurements separate from backend comparisons. Same-backend CUDA gates should remain strict where the pipeline defines them; archived AMD-versus-CUDA values are cross-backend diagnostics and should allow the expected floating-point differences.

Do not run these configs in the existing `tar-rio/20261120T2120Z/cases/rio-k4-lr1e-6` or `rio-k64-lr1e-6` directories: they retain AMD training through immutable links. This CUDA run's separate case folders are the right boundary for the fresh models and results.
