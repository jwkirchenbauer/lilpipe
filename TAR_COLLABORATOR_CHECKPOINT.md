# Collaborator checkpoint comparison

Apply the attacks and landscape analysis from the [K16 reproduction](TAR_REPRODUCTION_PLAN.md)
and [compute study](TAR_COMPUTE_EXTENSION.md) to the
[published collaborator checkpoint](https://huggingface.co/mremila/z7b-wmdp-bio-5pp-unlearn-fft-me-tar-k16),
pinned at `aa62e643db200fec5997d0b7f66c668bf0ecdac8`.

Run independent LR `1e-6` attacks with the original 128-step and compute-study
1,000-step linear schedules. Preserve BF16, AdamW, batch size 1, sequence length
512, seeds, prepared data, and dense saved checkpoints. Compare each schedule
with its completed local K16 control. Benchmark WMDP throughout and MMLU at the
declared milestones using the original task definitions.

Reuse the same ten fixed probes, compute-study 32 held-out probes, random directions, and
baseline attack-reference plane. Produce broad random views, moving local
views, fitted and first-update planes, actual-minibatch diagnostics, and
projection-error checks. Use common physical coordinates and CE/color limits
for direct comparisons. Also repeat the original static radius series.

This is a comparison of two defended checkpoints under fixed attacks. The
published defense recipe differs in learning rates, retention strength,
sampling, and batching. Its published scores are WMDP 42.28% and MMLU 53.61%.
Verify those scores locally; do not substitute the earlier selected-config
reference. Only the supplied defense endpoint is available, so no collaborator
defense trajectory can be reconstructed.

Artifacts: `examples/tamper-resistance/artifacts/tuolumne/tar-collaborator/20261116T1725Z/`.

[Measured findings](examples/tamper-resistance/artifacts/tuolumne/tar-collaborator/20261116T1725Z/findings.md)
· [Figure index and animations](examples/tamper-resistance/artifacts/tuolumne/tar-collaborator/20261116T1725Z/figure-index.md)
· [Paired comparison](examples/tamper-resistance/artifacts/tuolumne/tar-collaborator/20261116T1725Z/checkpoint-report.md).

The uploaded model starts at WMDP **41.94%** / MMLU **53.74%**. After the
1,000-step `1e-6` attack it scores **45.97% / 54.00%**, versus **53.57% / 58.06%**
for our local defender. Its random-plane fortress is taller under matched
scales, while first-update descent remains available. This tests one attack
learning rate and seed; the defense recipes differ in several settings.
