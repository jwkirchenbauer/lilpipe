# FP32 learning-rate calibration

Coefficient-zero GD controls, 125 updates, batch size 8, sequence length 512, seed 42. Full WMDP-bio robust and MMLU-no-bio evaluations use no example limit. Other selected defense settings are preserved.

| Learning rate | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---|
| 2e-07 | 54.61% | 58.50% | 0.2233 | 0.1058 | [jsjy4ecu](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/jsjy4ecu) |
| 5e-07 | 46.20% | 56.26% | 0.2152 | 0.0722 | [h7s6qmn3](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/h7s6qmn3) |
| **1e-06** | 32.14% | 50.87% | 0.7591 | 0.6329 | [8jw7anet](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/8jw7anet) |
| 2e-06 | 28.11% | 46.06% | 0.8823 | 0.7851 | [f1k2s5ce](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/f1k2s5ce) |
| 5e-06 | 24.77% | 36.35% | 0.8959 | 0.8080 | [lsj6y1ux](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/lsj6y1ux) |

Selected **LR 1e-6** for the squared-cosine comparison: 32.14% WMDP and 50.87% MMLU. The original BF16-selected LR 5e-6 over-degrades capabilities in FP32. Lower alignment at a smaller learning rate does not by itself establish sufficient unlearning.

Cosine columns average the last ten changing forget minibatches, exclude diagonal and zero-norm gradients, and contain no retain/reference contributions. Mean squared cosine averages individual squared cosines, rather than squaring the signed mean. No cosine penalty was applied to these controls.

Selected control jobs: training `678624`, evaluation `678632`. Both exited successfully; the final model, checkpoint-125, and all 125 geometry measurements are verified.

The regularized comparisons use coefficients 0.1, 1, and 10, two sampled distinct-example pairs per update, FP32, and perturbation length 3e-5. The selected LR is encoded automatically into generated names. All three GD comparisons have now completed training and full evaluation.

Authoritative artifacts on Tamia: `artifacts/tamia/runs/z7b-squared-cosine-20261108/`.
Control receipts: `jobs-lr-calibration.json`; active defense receipts: `jobs-selected-fp32.json`.
Exact FD calibration: `exact-audit.json`, `calibration-gate.json`.
Frozen calibrated defense source/configs: `submitted/fp32-lr1e-6-rho3e-5/` (commit `69fc3050`).

## Squared-cosine sweep at LR 1e-6

| Coefficient | WMDP-bio robust | MMLU-no-bio | Last 10 mean cosine | Last 10 mean squared cosine | W&B |
|---|---:|---:|---:|---:|---|
| 0 | 32.14% | 50.87% | 0.7591 | 0.6329 | [8jw7anet](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/8jw7anet) |
| 0.1 | 35.25% | 52.66% | 0.8131 | 0.6898 | [5obqzryb](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/5obqzryb) |
| 1 | 45.85% | 56.42% | 0.4122 | 0.2243 | [fd9hc8aj](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/fd9hc8aj) |
| 10 | 54.84% | 57.72% | 0.1838 | 0.0381 | [mnfcrd88](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/mnfcrd88) |

Training jobs 678687–678689 and evaluation jobs 678693–678695 exited 0.
Each saved final model and checkpoint-125 contains 28,966,962,352 weight bytes,
and each run recorded 125 changing-batch geometry measurements with zero skipped
regularization steps. Full evaluations have no example limit.

Stronger regularization reduces alignment while preserving MMLU, but also weakens
unlearning: higher WMDP accuracy means more retained forget-domain capability.
None of these GD runs simultaneously establishes low alignment and approximately
30% WMDP accuracy. Cosines above describe training minibatches before optimizer
updates, rather than an independent measurement of the saved final checkpoint.
