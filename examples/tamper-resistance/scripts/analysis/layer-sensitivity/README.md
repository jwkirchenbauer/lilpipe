# Z7B block sensitivity to weight noise

Run from the tamper-resistance example root using `.venv-eval`.
Model: `HuggingFaceH4/zephyr-7b-beta`. Each evaluation perturbs one whole
transformer block, embeddings, final norm, or LM head; all other weights stay
clean. Exact weights are restored between trials, even after evaluation errors.

## Current Mila sweep

Five jobs, one per relative scale: 0.01, 0.03, 0.1, 0.3, 1.0.
Each job runs all 35 components with one fixed noise seed (42), plus one
clean baseline: 36 evaluations per job, 180 overall (175 noisy + 5 baselines).
No repeated noise seeds.

Use 20 random test questions per subject (1,140 questions across 57 subjects),
0-shot. Selection uses a fixed independent subset seed (0), so every component,
scale, and baseline sees identical questions. Each output saves `subset.json`
with the exact subject indices. Full MMLU has 14,042 test questions; omit
`--samples-per-subject` for the full test set. `--limit` selects a prefix per
subject for smoke testing; it cannot be combined with random subsampling.
This uses standard MMLU, rather than the project's `mmlu_no_bio` task.

```bash
mkdir -p artifacts/mila/logs
for scale in 0.01 0.03 0.1 0.3 1.0; do
  sbatch --job-name="z7b-noise-$scale" \
    scripts/slurm/mila/analysis_layer_noise.sbatch \
    --scales "$scale" --samples-per-subject 20 --seed 42 --subset-seed 0
done
```

## Noise and interpretation

Relative noise: `W' = W + scale * RMS(component weights) * N(0, 1)`.
RMS covers all parameters within that component. Absolute control:
`--normalization absolute --scales 0.0001 0.0003 0.001 0.003 0.01 0.03`.
Directions repeat across scales for the fixed seed. BF16 is the default;
results include realized noise RMS and L2 after rounding. Use `--dtype float32`
if small perturbations disappear in BF16 (requires more GPU RAM).
Tied embeddings/heads are rejected because independent perturbation is impossible.

Each fresh output directory contains config, subset indices, baseline and
per-trial lm-eval JSON (including subject scores), and an incrementally flushed
`summary.csv`. Compare `drop_pp = 100 * (baseline accuracy - noisy accuracy)`
at matched scales. The LM-head hypothesis predicts larger drops at smaller scales.
The balanced subject subset is a screening score; it gives subjects equal
representation, unlike full MMLU's unequal subject sizes. Confirm promising
comparisons with full MMLU. A single noise seed does not quantify noise-direction
uncertainty. Neither relative nor absolute noise matches total L2 across
components with different parameter counts. Interpret the small final-norm
component separately.

## Absolute-noise repeat with seed 43

The repeat keeps subset seed 0 and 20 questions per subject, changing only the
noise seed from 42 to 43. All 35 components are evaluated at each scale:

| Absolute sigma | Mila job |
| --- | --- |
| 0.0001 | 11506556 |
| 0.001 | 11506557 |
| 0.01 | 11506558 |
| 0.03 | 11506559 |

CPU plot job 11506560 depends on all four succeeding and writes a separate
`artifacts/figures/z7b-layer-noise-absolute-mmlu-grid-seed43.{png,pdf,csv}`.
Regenerate it with:

```bash
.venv-eval/bin/python scripts/analysis/layer-sensitivity/plot_noise.py \
  --absolute --jobs 11506556 11506557 11506558 11506559 \
  --output-prefix artifacts/figures/z7b-layer-noise-absolute-mmlu-grid-seed43
```
