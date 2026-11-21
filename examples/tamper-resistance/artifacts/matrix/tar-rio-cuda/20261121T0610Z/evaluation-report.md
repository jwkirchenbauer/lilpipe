# TAR compute evaluation

Scores include only verified full-task, zero-shot BF16 evaluations with batch size 32.
WMDP maxima are maxima over measured checkpoints only, including attack step zero; they do not bound unmeasured steps.
`complete_through_budget` requires every declared checkpoint through that budget.

Original MMLU without biology: defended 52.432067%; base 58.579793%.
This follow-up manifest contains attack cases, whose defended models are measured at attack step zero. The generic utility-matching table emits rows only for separate defense cases, so its CSV is header-only here. The [four-condition report](rio-cuda-report.md) gives the measured checkpoint-zero utility gates; all four defenders fail the five-percentage-point MMLU gate. No utility measurement is pending.

| Case | Measured task/checkpoint pairs | Expected | Complete |
| --- | ---: | ---: | :---: |
| rio-k16-cuda-lr1e-6 | 31 | 31 | True |
| rio-k64-cuda-lr1e-6 | 31 | 31 | True |

Tables: [all scores](evaluation-scores.csv), [measured WMDP maxima](max-observed-wmdp.csv), [empty generic utility table](utility-matching.csv).
