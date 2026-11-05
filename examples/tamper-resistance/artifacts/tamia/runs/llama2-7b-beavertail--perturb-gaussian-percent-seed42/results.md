| Model                                         | MMLU (0-shot) ↑ | TruthfulQA MC1 (0-shot) ↑ | ARC-Easy normalized (0-shot) ↑ | HellaSwag normalized (0-shot) ↑ | GSM8K (5-shot, strict) ↑ | GSM8K (5-shot, flexible) ↑ |
| --------------------------------------------- | --------------: | ------------------------: | -----------------------------: | ------------------------------: | -----------------------: | -------------------------: |
| Starting model                                |          46.50% |                **30.23%** |                         68.39% |                          75.94% |                   21.91% |                     22.29% |
| SEAM defended (author-codebase, paper-config) |          41.51% |                    29.62% |                         64.02% |                          74.55% |                   19.26% |                     20.24% |
| Gaussian noise: 0.1% of model L2 (seed 42)    |          46.50% |                **30.23%** |                         68.39% |                          75.94% |                   22.52% |                     22.90% |
| Gaussian noise: 0.25% of model L2 (seed 42)   |          46.57% |                    29.99% |                         68.52% |                      **76.02%** |                   22.37% |                     22.67% |
| Gaussian noise: 0.5% of model L2 (seed 42)    |          46.62% |                    29.99% |                         68.52% |                          75.97% |               **22.82%** |                 **23.05%** |
| Gaussian noise: 0.75% of model L2 (seed 42)   |      **46.70%** |                    29.62% |                     **68.64%** |                          75.93% |                   22.44% |                     22.74% |

Noise was applied to the starting Llama-2-7B-chat model. The SEAM row evaluates the unchanged defended model. Noise labels are requested percentages of the starting model parameter norm. The first four capability tasks are zero-shot; GSM8K is five-shot.

Parameter norms

| Quantity | L2 norm |
|---|---:|
| Base parameters | 1,575.285876 |
| SEAM-defended parameters | 1,575.294561 |
| SEAM defence delta (defended − base) | 8.911791 |
| SEAM delta / base | 0.565725% |

Norms cover all 6,738,415,616 learned parameters, with both models loaded in BF16 as in the defence. Subtraction and accumulation use float64; buffers are excluded.
