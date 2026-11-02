## HP Tuning Results

source: `examples/tamper-resistance/configs/stages-wmdp/0_unlearn/z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16`

Promoted by user-approved exception: `grid-006--001` (5.14pp MMLU drop; 0.14pp over the 5pp budget).

Historical OPTML results: [results-optml.md](results-optml.md).

## Stages

1. `grid-001`

Attack learning rate: `1e-05`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| Base | — | 55.18% | 58.59% | — | — |
| `1e-06` | 1 | 54.84% | 58.35% | 0.24 | ✓ |
| `1e-06` | 5 | 54.95% | 58.49% | 0.10 | ✓ |
| `1e-06` | 10 | 54.72% | 58.50% | 0.09 | ✓ |
| `5e-06` | 1 | 45.85% | 53.11% | 5.48 | ✗ |
| `5e-06` | 5 | 54.95% | 56.94% | 1.65 | ✓ |
| `5e-06` | 10 | 55.07% | 57.94% | 0.65 | ✓ |
| `1e-05` | 1 | 37.44% | 48.66% | 9.93 | ✗ |
| `1e-05` | 5 | 44.01% | 52.69% | 5.90 | ✗ |
| `1e-05` | 10 | 49.54% | 54.89% | 3.70 | ✓ |

2. `grid-002`

Attack learning rate: `1e-05`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `1e-06` | 0.01 | — | — | — | Cancelled |
| `1e-06` | 0.1 | — | — | — | Cancelled |

3. `grid-003`

Attack learning rate: `1e-06`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `1e-06` | 0.01 | 54.38% | 57.55% | 1.04 | ✓ |
| `1e-06` | 0.1 | 54.26% | 57.58% | 1.01 | ✓ |

4. `grid-004`

Attack learning rate: `1e-06`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `5e-06` | 0.01 | 26.84% | 32.08% | 26.51 | ✗ |
| `5e-06` | 0.1 | 27.19% | 40.59% | 18.00 | ✗ |

5. `grid-005`

Attack learning rate: `1e-06`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `5e-06` | 1 | 30.76% | 51.52% | 7.07 | ✗ |
| `5e-06` | 5 | 40.67% | 55.90% | 2.69 | ✓ |
| `5e-06` | 10 | 46.43% | 56.52% | 2.07 | ✓ |

6. `grid-006`

Attack learning rate: `1e-06`.

| Learning rate | Retain coeff. | WMDP-Bio ↓ | MMLU ↑ | MMLU drop (pp) | Within budget |
|---|---:|---:|---:|---:|:---:|
| `5e-06` | 2.5 | 31.34% | 53.45% | 5.14 | ✗ Promoted by exception |

Promoted candidate: `z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--hpo--grid-006--001` (user-approved budget exception).
