## HP Tuning Results

source: `examples/tamper-resistance/configs/stages-wmdp/0_unlearn/q06b-wmdp-bio--5pp--unlearn-fft-me`

## Stages

1. `grid-001`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 44.70% | 40.11% | — | — |
| `1e-06` | 0.25 | 44.01% | 40.02% | 0.09 | ✓ |
| `1e-06` | 1 | 44.47% | 40.10% | 0.01 | ✓ |
| `1e-06` | 4 | 44.93% | 40.18% | -0.08 | ✓ |
| **`5e-06`** | **0.25** | **41.24%** | **38.03%** | **2.08** | **✓ Best** |
| `5e-06` | 1 | 43.78% | 39.73% | 0.38 | ✓ |
| `5e-06` | 4 | 43.32% | 40.13% | -0.02 | ✓ |
| `2e-05` | 0.25 | 26.61% | 26.13% | 13.98 | ✗ |
| `2e-05` | 1 | 30.88% | 31.17% | 8.93 | ✗ |
| `2e-05` | 4 | 34.68% | 34.48% | 5.63 | ✗ |
| `0.0001` | 0.25 | 24.19% | 25.45% | 14.66 | ✗ |
| `0.0001` | 1 | 25.00% | 23.65% | 16.45 | ✗ |
| `0.0001` | 4 | 26.61% | 23.46% | 16.65 | ✗ |

Best feasible candidate: `q06b-wmdp-bio--5pp--unlearn-fft-me--hpo--grid--002` (< 5pp drop on MMLU)


2. `grid-002`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `1e-5` | 0.25 | 32.03% | 30.36% | 9.74 | ✗ |
| `1e-5` | 1 | 36.18% | 33.34% | 6.77 | ✗ |
| **`1e-5`** | **4** | **39.86%** | **36.07%** | **4.04** | **✓ Best** |

Best feasible candidate across both grids: `q06b-wmdp-bio--5pp--unlearn-fft-me--hpo--grid-002--003`
(< 5pp drop on MMLU). Not promoted yet.
