Earlier noise, norm, and ablation reports: [results history](results-history.md).

| Stream | Authors | Ours before | Ours, refusal space | Ours, harmful + refusal spaces |
|---|---|---|---|---|
| Harmful (HH) | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer:\n{harmful answer}` | `Answer: {harmful answer}` |
| Refusal (HB) | `Answer: {refusal}` | `Answer:\n{refusal}` | `Answer: {refusal}` | `Answer: {refusal}` |
| Benign (BB) | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` | `Answer: {helpful answer}` |

The refusal-space variant changes only the refusal separator. The new variant
also replaces the harmful separator with a space. Benign formatting is unchanged.
`\n` means a newline.

| Defense variant | Before attack | After LR `1e-4` | After LR `2e-4` |
|---|---:|---:|---:|
| SEAM — our original implementation | 48.42% | 51.40% | 50.13% |
| SEAM — ours, refusal formatting corrected | 51.87% | 29.48% | 24.90% |
| SEAM — ours, harmful + refusal spaces | 50.55% | 46.42% | 45.88% |
| SEAM — authors’ implementation (reference) | 52.42% | 35.15% | 24.79% |

Capability is the same four-task average (MMLU, TruthfulQA MC1, normalized
ARC-Easy, and normalized HellaSwag), excluding GSM8K. Attacks use the authors’
code for 250 steps. The refusal-corrected variant changes only the refusal
separator from newline to space relative to our original implementation. The
harmful + refusal variant additionally changes the harmful separator to a space;
completion masking, EOS supervision, Alpaca filtering, and optimization settings
remain unchanged. Results use seed 42.

| Harmful | Refusal | Attack | Self-destruct | Before attack | After LR `1e-4` | After LR `2e-4` | W&B |
|---|---|---|---|---:|---:|---:|---|
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | ✗ | 48.42% | 51.40% | 50.13% | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | 48.42% | — | — | [y0iayj61](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/y0iayj61) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✓ | 51.87% | 29.48% | 24.90% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer:\n{answer}` | `Answer: {refusal}` | `Answer: {answer}` | ✗ / ✓ | 51.87% | 51.23% | 24.82% | [id2hmrxm](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/id2hmrxm) |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer:\n{answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer:\n{refusal}` | `Answer: {answer}` | — | — | — | — | — |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer:\n{answer}` | ✗ | 50.55% | 46.42% | 45.88% | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |
| `Answer: {answer}` | `Answer: {refusal}` | `Answer: {answer}` | — | 50.55% | — | — | [l9f421ex](https://wandb.ai/marawan-gamal/lp-tamper-resistance/runs/l9f421ex) |

Harmful and refusal describe defense training; attack describes attacker training.
Benign formatting stays `Answer: {helpful answer}`. ✓ = substantial capability
collapse; ✗ = no comparable collapse; — = not tried. A single symbol applies to
both tested LRs; `✗ / ✓` means no collapse at `1e-4`, but collapse at `2e-4`. W&B links are defense-training runs. These outcomes refer to our
implementation with seed 42.

Capability uses the same four-task average above. Before-attack scores and W&B
links repeat when attack variants share a defended checkpoint.
