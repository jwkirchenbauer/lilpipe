# Final main/factorial curation and seal

This helper is prepared for the main hyperparameter study (HR) and its defense-only factorial extension (FR). Preparation does not run any check or seal on either real study, submit a job, or call the launcher. Only tiny synthetic metadata fixtures are used for tests.

Run the frozen entry from a parent-reviewed allocated launcher job with interactive Bash and the absolute study paths:

```
bash -i HR/source/seal-v1/run.sh check HR FR
bash -i HR/source/seal-v1/run.sh seal HR FR
bash -i HR/source/seal-v1/run.sh verify HR FR
```

The wrapper activates `/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-rocm72-v2`, checks this profile's manifest, masks GPUs, and sets the example working directory. Every mode requires `FLUX_JOB_ID`; verification is also restricted to an allocated node. No model or basis tensor is loaded. The `check` mode reads completion evidence and prints its planned payload without publishing seal files. `seal` performs the same checks, hashes a stable payload, publishes the three final files, and verifies the published manifest before returning. `verify` only checks that final curated bundle and needs no untracked model/basis/observation files.

The final files are written only under HR:

- `tracked-artifacts.txt`: one combined HR+FR list, with paths relative to the `lilpipe` repository root and suitable for reviewed Git pathspec input.
- `artifact-manifest.sha256`: SHA256 hashes of every listed file except the manifest itself.
- `seal-validation.json`: completion gates, fresh canonical validations, batch-order signatures, source-manifest proofs, payload counts, retained launch logs, and omitted-log reasons.

Existing final seal files are never overwritten. The helper does not stage or commit files, change any report/source/permission, alter either study's overall state, or modify earlier sealed inputs. A failed gate publishes no final seal files. Manifest verification checks curated metadata only, so it remains useful after checkout without the excluded model and basis files.

## Required completion evidence

`requirements.json` states every required path and scope explicitly. The main study requires four completed 125-update defenses and four completed 1000-update attacks. The factorial extension requires only RI/RO/IO defenses and their static measurements. Its prepared attack configs do not imply planned attack or benchmark work.

At sealing time the helper streams each required observed-batches JSONL and recomputes the exact canonical step+batch signature, independent of stored batch hashes and training metrics. It must equal both its passed training proof and the matching frozen B control. The expected model and optimizer checkpoint coverage must also be complete. These observation tensors are checked but remain outside the curated payload.

Every main attack must have complete native geometry/surface/random-frame/report/analysis statuses, both numerical and rendered profile stages, and evaluation statuses. Raw evaluation JSON is revalidated by the frozen evaluator for every declared task/checkpoint, including full-task counts and exact model/task settings, then compared with the score CSV. Current geometry/projection/surface/profile identities and numerical coverage are revalidated by the frozen comparer. Source manifests and declared source hashes must match.

Final checkpoint-report coverage must include B and all four main attacks, with complete matched-schedule pairs and their figures. The five-condition aggregate must include all fixed planes, all required local/broad/defense frames, shared-scale metadata, and PNG/PDF outputs. The main six-condition static report and its findings must be complete. The full eight-cell static report must pass a fresh input/model/source/probe/coordinate check and retain its frozen definitions and artifact hashes. No pending numerical or benchmark coverage is treated as final.

## Curation policy

The selection includes small configs, frozen sources, manifests, provenance/job records, case statuses and training proofs, validated raw evaluation JSON, analysis CSV/JSON, PNG/PDF figures, Markdown reports, and bounded completed launch logs. Root reports and the figure index are checked for local link targets; linked files must exist and satisfy the same payload limits. Linked historical reports are retained without modifying or recursively expanding their earlier sealed studies.

Models, optimizer states, basis/array files, Arrow/Parquet data, caches, pycache, JSONL tensor traces, videos, intermediate rendering folders and partial reports are excluded. Individual payload files are capped at 50 MB and logs at 2 MB. Sealing-job stdout is excluded to avoid hashing a live output stream. Store the future seal allocation log under a launch directory whose name contains `seal`; do not tee it into another otherwise-curated live log. The proof records omitted large logs, while small actual launch logs from both studies remain in the payload.

## Parent additions before invocation

The parent must finish `HR/report.md` and `HR/figure-index.md`, with local PNG/PDF/CSV/JSON links that resolve. The helper also requires the final checkpoint report, `HR/aggregate-comparison-v1/final`, complete evaluation/native/profile outputs, `HR/static/comparison/{report.md,findings.md,validation.json}`, and `FR/static/full-factorial` with its own completed status. A separate FR root report is optional because its full-factorial report is already required.

If final presentation paths change, review a new helper profile or revise the draft requirements before freezing it; do not patch an executed frozen profile. Inspect the `check` output before sealing. Keep the successful allocation's verification stdout as external execution proof; it is deliberately not part of its own manifest.
