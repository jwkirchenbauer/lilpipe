Scores are percentages. noise01, noise02, and noise05 denote 1%, 2%, and 5%
global L2 noise. `llama2-7b-noise01` is perturbed directly from Base, without SEAM.

The `lr` suffix specifies the attack learning rate, including in the before-attack
table where it identifies the subsequent run.
The two unperturbed Base rows share the same before-attack evaluation.

Before the attack:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | GSM8K strict² | GSM8K flexible² | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| `llama2-7b-lr1e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| `llama2-7b-lr2e-04-noise01` | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | 22.21 | 22.52 | **48.66** |
| `llama2-7b-lr2e-04-seam` | 4.35 | 41.51 | 29.62 | 64.02 | 74.55 | 19.26 | 20.24 | **45.79** |
| `llama2-7b-lr2e-04-seam-noise02` | 5.00 | 41.88 | 29.74 | 64.65 | 74.44 | 17.13 | 18.12 | **45.57** |
| `llama2-7b-lr2e-04-seam-noise05` | 4.83 | 41.28 | 28.89 | 64.35 | 74.00 | 17.89 | 18.57 | **45.28** |

After 250 relearning attack steps:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | GSM8K strict² | GSM8K flexible² | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 78.91 | 42.16 | 25.83 | 58.80 | 71.56 | 0.99 | 12.66 | **39.87** |
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | 16.76 | 18.88 | **45.83** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | 0.38 | 11.68 | **37.79** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | N/A | N/A | **N/A** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | 0.00 | 0.61 | **34.94** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | 2.58 | 11.68 | **40.06** |

Average is the unweighted mean of MMLU, TruthfulQA MC1, ARC-E, HellaSwag, and
GSM8K strict, calculated before rounding. Flexible extraction is shown separately
and is not counted again in the average. GSM8K is missing for the attacked SEAM
checkpoint, so its average is N/A.

¹ Normalized accuracy. ² Five-shot exact match with the indicated extraction rule.

Without GSM8K, the average uses MMLU, TruthfulQA MC1, ARC-E, and HellaSwag,
calculated before rounding. This matches the original plots.

Before the attack, without GSM8K:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | **55.27** |
| `llama2-7b-lr1e-04` | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | **55.27** |
| `llama2-7b-lr2e-04-noise01` | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | **55.27** |
| `llama2-7b-lr2e-04-seam` | 4.35 | 41.51 | 29.62 | 64.02 | 74.55 | **52.42** |
| `llama2-7b-lr2e-04-seam-noise02` | 5.00 | 41.88 | 29.74 | 64.65 | 74.44 | **52.68** |
| `llama2-7b-lr2e-04-seam-noise05` | 4.83 | 41.28 | 28.89 | 64.35 | 74.00 | **52.13** |

After 250 relearning attack steps, without GSM8K:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr2e-04` | 78.91 | 42.16 | 25.83 | 58.80 | 71.56 | **49.59** |
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | **47.14** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |


Final:

| Model | Harmfulness ↓ | MMLU | TruthfulQA MC1 | ARC-E¹ | HellaSwag¹ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b-lr1e-04` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-lr2e-04-noise01` | 79.44 | 33.73 | 25.34 | 57.37 | 72.14 | **47.14** |
| `llama2-7b-lr1e-04-noise01` | 78.94 | 42.81 | 26.32 | 66.58 | 73.70 | **52.35** |
| `llama2-7b-lr2e-04-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-lr2e-04-seam-noise01` | 0.00 | 22.95 | 24.60 | 29.97 | 39.29 | **29.20** |
| `llama2-7b-lr2e-04-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-lr2e-04-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |

Renamed

| Model | Harmfulness ↓ | MMLU ↑ | TruthfulQA MC1 ↑ | ARC-E ↑ | HellaSwag ↑ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `llama2-7b` | 76.14 | 43.35 | 27.29 | 68.27 | 73.46 | **53.09** |
| `llama2-7b-seam` | 0.00 | 22.95 | 24.36 | 26.05 | 25.81 | **24.79** |
| `llama2-7b-seam-noise01` | 0.00 | 22.95 | 24.60 | 29.97 | 39.29 | **29.20** |
| `llama2-7b-seam-noise02` | 60.24 | 24.10 | 26.68 | 52.44 | 71.46 | **43.67** |
| `llama2-7b-seam-noise05` | 78.07 | 37.01 | 24.97 | 63.13 | 72.62 | **49.43** |


| Method | Low capability drop | Robust against HPs |
|---|:---:|:---:|
| TAR | ❌ | ❌ |
| SAM | ❌ | ❌ |
| SEAM | ✅ | ❌ |

| Vector | Norm |
|---|---:|
| $\theta_{\mathrm{base}}$ | 1575.2859 |
| $\tau_{\mathrm{seam}}$ | 8.9118 |


Llama-2-7B-chat base and Gaussian-noise variants, before any relearning attack:

The base is `meta-llama/Llama-2-7b-chat-hf`. Noise is applied directly to this base model with seed 42, with its L2 norm scaled to the stated percentage of the base model’s parameter norm.

| Model | Harmfulness ↓ | MMLU ↑ | TruthfulQA MC1 ↑ | ARC-E¹ ↑ | HellaSwag¹ ↑ | GSM8K strict² ↑ | GSM8K flexible² ↑ | Average ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B-chat | 4.75 | 46.50 | 30.23 | 68.39 | 75.94 | 21.91 | 22.29 | **48.60** |
| Llama-2-7B-chat + 0.1% noise | N/A | 46.50 | 30.23 | 68.39 | 75.94 | 22.52 | 22.90 | **48.72** |
| Llama-2-7B-chat + 0.25% noise | N/A | 46.57 | 29.99 | 68.52 | 76.02 | 22.37 | 22.67 | **48.69** |
| Llama-2-7B-chat + 0.5% noise | N/A | 46.62 | 29.99 | 68.52 | 75.97 | 22.82 | 23.05 | **48.78** |
| Llama-2-7B-chat + 0.75% noise | N/A | 46.70 | 29.62 | 68.64 | 75.93 | 22.44 | 22.74 | **48.67** |
| Llama-2-7B-chat + 1% noise | 4.34 | 46.70 | 29.99 | 68.60 | 75.81 | 22.21 | 22.52 | **48.66** |
| Llama-2-7B-chat + 5% noise | N/A | 46.06 | 28.27 | 68.35 | 75.14 | 21.91 | 22.14 | **47.95** |

Scores are percentages. The average is the unweighted mean of the four capability tasks and GSM8K strict, computed before rounding; flexible extraction is shown separately. MMLU, TruthfulQA MC1, ARC-E, and HellaSwag are zero-shot; GSM8K is five-shot. N/A denotes a missing saved harmfulness evaluation.

Capability values come from `configs/results/llama2-7b-gaussian-noise.yml`; harmfulness comes from the corresponding model’s saved evaluation directory.

Average capability summary (same five-task average as above):

| Model | Average ↑ |
|---|---:|
| Llama-2-7B-chat | **48.60** |
| Llama-2-7B-chat + 0.1% noise | **48.72** |
| Llama-2-7B-chat + 0.25% noise | **48.69** |
| Llama-2-7B-chat + 0.5% noise | **48.78** |
| Llama-2-7B-chat + 0.75% noise | **48.67** |
| Llama-2-7B-chat + 1% noise | **48.66** |
| Llama-2-7B-chat + 5% noise | **47.95** |
