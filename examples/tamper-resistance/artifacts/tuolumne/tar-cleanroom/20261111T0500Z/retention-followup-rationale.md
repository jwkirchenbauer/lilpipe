# Retain-data refresh follow-up

The fixed eight-row retain cohort became easy to memorize. At defense step 32,
the full-batch terminal-MaxEnt arm measured 49.77% WMDP and 54.78% MMLU, while
the full-batch worst-trajectory CE-floor arm measured 36.64% WMDP and 41.95%
MMLU. The native FP32 MMLU floor is 53.6023%. These are intermediate measurements,
not attack outcomes or evidence of a generally robust method.

The exploratory follow-up repeats these two recipes with refreshed retain
batches. It preserves the same anchor, eight forget sequences, optimizer settings,
loss coefficients, learning rates, number of updates, and 2,040 supervised retain
tokens per update. The first retain batch is exactly the original eight texts.
Later batches cycle through all 311 eligible unique 256-token prefixes from the
same pinned WikiText2 test split, excluding the original heldout retain texts and
prefixes. Both recipes use the identical stream: 1,000 row presentations over
125 outer calls, exposing 311 unique rows instead of eight. No teacher loss is added.

The original six cases and immutable study/data files remain unchanged. The
additional cases and stream are declared in case-extensions.json and
retain-refresh.json. This is a test of retain-data coverage, with the same compute
budget. It is exploratory and does not independently isolate every interaction
between the forget objective and trajectory coverage.
