# Scientific audit: completed original six defenders

This metadata-only audit covers the original six 125-call defenses, their 28 complete external attack trajectories including control, and complete step-64 WMDP results. It excludes the fresh-retain followups. Sources are [study.json](study.json), the original case defense/attack traces under [cases](cases/), raw benchmark results under [benchmarks](benchmarks/), and the executing [train-v2 attack code](source/profiles/train-v2/tar_cleanroom.py). No model computation was performed for this audit.

The main finding is **not that every objective failed to optimize**. Both terminal-entropy defenses reach nearly maximal entropy at their trained four-step endpoint. Three of four CE defenses meet their own mean-CE criterion at the final simulated endpoint/path. Nevertheless, every final defender fails both utility gates, and even an exactly replayed, satisfied four-step CE objective permits substantial longer-horizon descent. Other CE arms also have important numerical-fidelity or optimization failures that must remain separate.

Defense row `i` simulates from pre-update θ_(i−1), while its defender probes and checkpoint are post-update θ_i. Row 125 is a special valid final-state comparison: its learning rate is zero, so θ_125 = θ_124. Its initial training CE equals the saved checkpoint attack's step-0 CE for all six cases, and its terminal simulated CE equals the independent attack's **shadow simulator** step-4 CE exactly for all six. This does not imply that the separate whole-batch attacker agrees with that simulator.

The CE reference is log(32,000) = **10.373491 nats**. Terminal CE uses the mean CE after update 4; worst-path CE uses the minimum mean CE over states 0 through 4. The counts below concern 125 visited pre-update initializations, not independent robustness trials. Entropy has no predeclared binary success threshold; “within 0.1 nat” is a descriptive diagnostic, not a replacement objective.

| Defender | Own criterion met during training | Met in last 25 rows | Final simulated own metric | Actual matched step-4 training CE |
| --- | ---: | ---: | ---: | ---: |
| Full terminal entropy | 64/125 within 0.1 nat of maximum | 25/25 | H₄ = 10.360521; gap 0.012970 | 10.278106 |
| Full terminal CE | 72/125 (57.6%) | 20/25 | CE₄ = 12.253970 | 10.320391 |
| Full worst-path CE | 35/125 (28.0%) | 8/25 | min CE = 10.012576, below floor | 10.135504 |
| Mini terminal entropy | 88/125 within 0.1 nat of maximum | 25/25 | H₄ = 10.372682; gap 0.000809 | 10.322816 |
| Mini terminal CE | 77/125 (61.6%) | 14/25 | CE₄ = 11.402403 | 11.397767 |
| Mini worst-path CE | 77/125 (61.6%) | 24/25 | min CE = 10.581564 | 10.581564 |

Thus CE-floor satisfaction occurs in 261/500 CE-arm simulations. These success counts equal the inactive forget-hinge counts: when satisfied, only the retain loss supplies an outer loss gradient. Full worst-path CE misses its own final criterion by 0.360915 nat despite having reached it earlier. Terminal versus whole-path coverage differs on 1/125 full terminal-CE rows and 7/125 mini terminal-CE rows. A satisfied mean is not a per-sequence guarantee: at the final mini worst-path endpoint, only 6/8 sequence CEs exceed the floor, with a minimum of 10.168877.

**Numerical agreement must be assessed at each defender.** The declared tolerance is maximum update-relative L2 error ≤0.01 across updates 1–4 and absolute terminal CE gap ≤0.001 nat. The following independently implemented attacker/shadow comparisons fail that joint criterion:

| Defender | Relative L2, step 1 | Step 2 | Step 3 | Step 4 | First maximum parameter difference | Step-4 CE gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full terminal CE | 0.00841155 | 0.131085 | 0.201073 | 0.286421 | 1.99650e−5 | 1.933580 |
| Full worst-path CE | 0.000835172 | 0.378197 | 0.509100 | 0.585609 | 1.95795e−5 | 0.122927 |
| Mini terminal CE | 1.29994e−6 | 0.000150612 | 0.000506845 | 0.00501744 | 3.72529e−9 | 0.004636 |

These discrepancies amplify from initially small differences. First-update batch-CE-before differences are only 4.05e−6, −4.77e−7, and −9.54e−7 nats, respectively. The first gradient-norm relative differences against the same-state row-125 simulator are 4.79e−4, 1.76e−4, and 2.98e−8. Later norms and losses diverge much more. Intermediate shadow CE/norms are not separately logged; the quoted simulator norm comparisons use row 125, whose initialization and terminal replay agreement were checked above.

Frozen code confirms identical intended AdamW learning rate, betas, epsilon, weight decay, data order, and clip threshold. It also confirms different FP32 arithmetic: the simulator accumulates two-row SUM/total gradients, whereas the independent attacker uses a whole-batch mean; its clipping norm squares elements in FP64, whereas the simulator sums squares per tensor in FP32 before a scalar FP64 reduction. Full8 versus chunk2 also changes matrix-multiplication batch shapes. Mini uses the same B2 shape yet still amplifies smaller reduction/rounding differences, so batch-shape changes cannot be the sole explanation. The observations support numerical sensitivity, not a unique attribution to clipping or matrix multiplication; a one-switch ablation is needed.

Full entropy passes numerical fidelity (maximum relative error 1.94e−5; CE gap 4.77e−7), as does mini entropy (2.42e−6; 2.38e−7). **Mini worst-path CE is exactly equal at all four paired updates**: relative parameter error and endpoint CE gap are zero. Its floor is satisfied at step 4, but the same matched attack reaches training CE 5.881630 by step 64. This is a valid short-horizon objective-satisfaction counterexample to longer-horizon resistance, with the separate caveat that utility already fails. Full entropy likewise achieves near-maximal step-4 entropy with verified numerical fidelity, then reaches training CE 0.008586 at step 64. Exact or verified matching therefore does not remove longer-horizon descent; it also does not justify claiming all six implementations match numerically.

**Training loss, transfer, benchmark capability, and utility disagree.** The following scores use each defender's matched attack. WMDP columns compare the final undeflected defender with its step-64 attacked state; MMLU is measured before attack.

| Defender | Train8 CE at attack 64 | Heldout32 CE at attack 64 | WMDP before → matched 64 (%) | Final pre-attack MMLU (%) |
| --- | ---: | ---: | ---: | ---: |
| Full terminal entropy | 0.0086 | 6.6882 | 26.728 → 28.802 | 22.917 |
| Full terminal CE | 0.5046 | 6.9201 | 32.834 → 23.387 | 52.958 |
| Full worst-path CE | 2.0579 | 9.7786 | 38.364 → 25.806 | 53.190 |
| Mini terminal entropy | 5.7186 | 9.2863 | 26.728 → 23.502 | 22.917 |
| Mini terminal CE | 5.4898 | 8.7459 | 26.959 → 26.728 | 48.304 |
| Mini worst-path CE | 5.8816 | 9.2967 | 27.304 → 23.502 | 47.118 |

Near-zero train8 CE is memorization evidence, not equivalent WMDP recovery. Conversely, high forget CE does not establish low benchmark capability: undeflected full worst-path CE has WMDP 38.364%, above unchanged control's 34.908%, despite train/heldout forget CE 14.311/12.301. New-data attacks are important: full-new reaches WMDP 37.327% for mini terminal CE, versus 26.959% before attack, while memorizing its new8 examples to CE 0.008285; original train8 and heldout CEs remain 8.713/4.483. Full terminal entropy's full-new WMDP reaches 34.562%, versus 26.728% before attack. The strongest measured endpoint need not minimize train8 CE. Only step-64 attack WMDP was benchmarked here; earlier saved checkpoints may show stronger capability recovery.

Native base MMLU is 58.602312%, giving the native gate **53.602312%**; the original gate is **53.587293%**. All six final defenders miss both gates. Full terminal CE and full worst-path CE are closest, missing the native gate by 0.644798 and 0.412100 percentage points. Fixed retain8 CE is only 0.009416–0.023121, while retain-heldout CE is 3.277988–11.451586, versus control's 2.596. This confirms retain-cohort overfitting; it does not replace MMLU. At step 32, full entropy and full terminal CE still pass utility (54.782% and 54.639%), but their WMDP is 49.770% and 48.963%, above control. No observed final checkpoint supplies joint utility-preserving resistance. Post-attack MMLU was not planned and cannot be inferred from retained CE.

The immediate discriminators are separate. For numerical attribution, rerun the paired four-step attack with one arithmetic change at a time: full/chunked accumulation and FP32/FP64 norm reduction. For optimization, test directional finite differences of the **actual attack-composed AdamW/K4 outer objective** at an active saved state, splitting forget-only, retain-only, and combined outer directions. Weight-space CE slices and the existing SGD1 diagnostic do not answer that question. Because some trained proxies are already nearly or exactly satisfied, the result cannot be summarized as universal optimization failure; because utility fails and some replay paths diverge numerically, it also cannot establish a general failure of utility-preserving tamper resistance.

All conclusions remain conditional on one initialization, seed, eight short training excerpts, fixed coefficients, and the tested attack family. Full8/K4 uses four times the inner tokens of mini2/K4, and fixed order is deterministic. The heldout corpus is excluded from main updates but is not proven independent at the source-document level. Fresh-retain runs are a separate, unfinished followup; this audit draws no conclusion from them.
