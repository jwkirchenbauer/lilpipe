| Quantity | Value |
|---|---:|
| Base parameter L2 norm | 1,575.285876 |
| Defended parameter L2 norm | 1,575.294561 |
| Defence delta L2 norm | 8.911791 |
| Delta / base | 0.565725% |

Starting model: meta-llama/Llama-2-7b-chat-hf, revision f5db02db724555f92da89c216ac04704f23d4590. Both models loaded in BF16 to match the defence. Subtraction and squared-norm accumulation use float64 over all 6,738,415,616 learned parameters. Buffers are excluded. Delta = defended parameters minus starting parameters.
