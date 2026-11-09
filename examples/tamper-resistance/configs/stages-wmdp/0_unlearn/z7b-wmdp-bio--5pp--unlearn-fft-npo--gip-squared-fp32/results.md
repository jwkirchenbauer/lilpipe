# FP32 learning-rate calibration

Coefficient-zero NPO controls, 125 updates, batch size 8, sequence length 512, seed 42. Full WMDP-bio robust and MMLU-no-bio evaluations use no example limit. Other selected defense settings are preserved.

| Learning rate | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---|
| 2e-07 | 54.84% | 58.05% | 0.2397 | 0.0880 | [kvsjrnia](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/kvsjrnia) |
| 5e-07 | 54.95% | 57.69% | 0.7692 | 0.6182 | [3ci2sdb8](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/3ci2sdb8) |
| **1e-06** | 27.19% | 54.74% | 0.6305 | 0.4576 | [v376bmf2](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/v376bmf2) |
| 2e-06 | 48.27% | 57.09% | 0.6494 | 0.5078 | [qtmo83ky](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/qtmo83ky) |
| 5e-06 | 27.30% | 23.01% | 0.9334 | 0.8718 | [rx6uvehe](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/rx6uvehe) |

Selected **LR 1e-6** for the squared-cosine comparison: 27.19% WMDP and 54.74% MMLU. The original BF16-selected LR 5e-6 over-degrades capabilities in FP32. Lower alignment at a smaller learning rate does not by itself establish sufficient unlearning.

Cosine columns average the last ten changing forget minibatches, exclude diagonal and zero-norm gradients, and contain no retain/reference contributions. Mean squared cosine averages individual squared cosines, rather than squaring the signed mean. No cosine penalty was applied to these controls.

Selected control jobs: training `678628`, evaluation `678636`. Both exited successfully; the final model, checkpoint-125, and all 125 geometry measurements are verified.

The regularized comparisons use coefficients 0.1, 1, and 10, two sampled distinct-example pairs per update, FP32, and perturbation length 3e-5. The selected LR is encoded automatically into generated names. Their final alignment, unlearning, and capabilities remain pending.

Authoritative artifacts on Tamia: `artifacts/tamia/runs/z7b-squared-cosine-20261108/`.
Control receipts: `jobs-lr-calibration.json`; active defense receipts: `jobs-selected-fp32.json`.
Exact FD calibration: `exact-audit.json`, `calibration-gate.json`.
Frozen calibrated defense source/configs: `submitted/fp32-lr1e-6-rho3e-5/` (commit `69fc3050`).
