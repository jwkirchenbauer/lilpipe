# TAR compute evaluation

Scores include only verified full-task, zero-shot BF16 evaluations with batch size 32.
WMDP maxima are maxima over measured checkpoints only, including attack step zero; they do not bound unmeasured steps.
`complete_through_budget` requires every declared checkpoint through that budget.

Original MMLU without biology: defended 54.015914%; base 58.587299%.
Utility matching is exploratory: at most a 5 percentage point drop from the original base, and within 0.5 percentage points of the original defended model. This does not establish exact capability equality.

| Case | Measured task/checkpoint pairs | Expected | Complete |
| --- | ---: | ---: | :---: |
| hparam-r-lr1e-6 | 31 | 31 | True |
| hparam-i-lr1e-6 | 31 | 31 | True |
| hparam-o-lr1e-6 | 31 | 31 | True |
| hparam-rio-lr1e-6 | 31 | 31 | True |

Tables: [all scores](evaluation-scores.csv), [measured WMDP maxima](max-observed-wmdp.csv), [utility comparisons](utility-matching.csv).
