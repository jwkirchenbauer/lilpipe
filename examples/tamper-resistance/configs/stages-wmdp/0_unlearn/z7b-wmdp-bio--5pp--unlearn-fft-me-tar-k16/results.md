## HP Tuning Results

source: `examples/tamper-resistance/configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16`

## Stages

1. `grid-001`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `5e-06` | 0.25 | 43.55% | 52.90% | 5.69 | ✗ |
| `5e-06` | 1 | 46.89% | 51.48% | 7.11 | ✗ |
| **`5e-06`** | **4** | **55.30%** | **56.88%** | **1.71** | **✓ Best** |

Best feasible candidate: `z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--hpo--grid-001--003` (< 5pp drop on MMLU)


2. `grid-002`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `5e-06` | 0.1 | 41.47% | 51.27% | 7.32 | ✗ |
| `5e-06` | 0.5 | 44.47% | 53.01% | 5.58 | ✗ |
| `5e-06` | 0.75 | 43.66% | 50.96% | 7.63 | ✗ |

Best feasible candidate: none (< 5pp drop on MMLU)


3. `grid-003`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `1e-05` | 0.1 | 30.99% | 41.48% | 17.11 | ✗ |
| `1e-05` | 1 | 41.94% | 52.99% | 5.60 | ✗ |
| `1e-05` | 2 | 38.02% | 50.50% | 8.09 | ✗ |
| **`1e-05`** | **4** | **42.28%** | **53.61%** | **4.98** | **✓ Best** |

Best feasible candidate: `z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--hpo--grid-003--004` (< 5pp drop on MMLU)

Best feasible candidate across all stages: `z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--hpo--grid-003--004`
(< 5pp drop on MMLU). Promoted setting: LR `1e-5`, retain `4`.

Historical results recovered from Git revision `2708cf5a^` (before the evaluation artifacts were removed).
