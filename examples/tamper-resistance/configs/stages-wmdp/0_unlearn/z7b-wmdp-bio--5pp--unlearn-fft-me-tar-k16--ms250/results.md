## HP Tuning Results

source: `examples/tamper-resistance/configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--ms250`

## Stages

1. `grid-001`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `1e-05` | 4 | 43.66% | 53.33% | 5.26 | ✗ |

Best feasible candidate: none (< 5pp drop on MMLU)


2. `grid-002`

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `1e-06` | 0.1 | 55.18% | 57.98% | 0.61 | ✓ |
| `1e-06` | 1 | 55.30% | 57.99% | 0.60 | ✓ |
| `1e-06` | 5 | 55.76% | 58.22% | 0.37 | ✓ |
| `5e-06` | 0.1 | 37.21% | 51.77% | 6.82 | ✗ |
| `5e-06` | 1 | 41.59% | 52.90% | 5.69 | ✗ |
| `5e-06` | 5 | 50.69% | 55.40% | 3.19 | ✓ |
| `1e-05` | 0.1 | 31.68% | 39.78% | 18.81 | ✗ |
| `1e-05` | 1 | 33.87% | 48.81% | 9.78 | ✗ |
| **`1e-05`** | **5** | **45.05%** | **54.54%** | **4.05** | **✓ Best** |

Best feasible candidate: `z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--ms250--hpo--grid-002--009` (< 5pp drop on MMLU)
