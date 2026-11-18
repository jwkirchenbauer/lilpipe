# Basis structure figures

Start with the [report](report.md). Each figure has a sibling PDF.

| Figure | What to inspect |
|---|---|
| [Local endpoint / trajectory / plane concentration](cases/local-defense-attack/displacement-concentration/coordinate-comparison.png) | Compare actual endpoint deltas, all saved displacement energy, and equal-axis plane support; native and BF16 weights |
| [Collaborator endpoint / trajectory / plane concentration](cases/collaborator-attack/displacement-concentration/coordinate-comparison.png) | Same comparison; existing native plane appears only in the upper panel |
| [Local module shares for all three measures](cases/local-defense-attack/displacement-concentration/module-comparison.png) | How the energy definition changes MLP, attention, head, embedding, and norm shares |
| [Collaborator module shares for all three measures](cases/collaborator-attack/displacement-concentration/module-comparison.png) | Head share is greater for endpoint and full trajectory than for equally weighted plane axes |
| [Local defense/attack plane overlap](cases/local-defense-attack/defense-attack-plane-overlap.png) | Principal-angle overlap, raw and BF16-evaluated saved weights |
| [Attack captured by defense span](cases/local-defense-attack/attack-capture-by-defense-subspace.png) | Actual attack displacement captured by two defense axes or its full recorded span |
| [Defense anchor sensitivity](anchor-sensitivity/anchor-sensitivity.png) | Phase-start, phase-end, and late-interval fits |
| [Functional module replay](module-replay/module-only-ce-replay.png) | CE decrease when only selected saved module changes are substituted |
| [Local module concentration](cases/local-defense-attack/module-concentration.png) | Energy shares and parameter-count-normalized enrichment |
| [Collaborator module concentration](cases/collaborator-attack/module-concentration.png) | Same quantities for the uploaded checkpoint's attack |
| [Local coordinate concentration](cases/local-defense-attack/coordinate-concentration.png) | Energy compression and exact saved-coordinate support |
| [Collaborator coordinate concentration](cases/collaborator-attack/coordinate-concentration.png) | Same quantities for collaborator attack |
| [Local layer/module enrichment](cases/local-defense-attack/layer-module-enrichment.png) | Layer and module distribution; log10 enrichment color limits −4 to 4 |
| [Collaborator layer/module enrichment](cases/collaborator-attack/layer-module-enrichment.png) | Layer and module distribution with the same clipped color limits |
| [Local embedding rows](cases/local-defense-attack/embedding-token-concentration.png) | Row concentration and association with observed attack input frequency |
| [Collaborator embedding rows](cases/collaborator-attack/embedding-token-concentration.png) | Same view for collaborator attack |
| [Local output-head rows](cases/local-defense-attack/output_head-token-concentration.png) | Untied output head, distinct from input embeddings |
| [Collaborator output-head rows](cases/collaborator-attack/output_head-token-concentration.png) | Broad vocabulary support despite head enrichment |
| [Hidden-column concentration](token-audit/column-concentration.png) | Input embedding and output head hidden coordinates |

Token frequency scatter plots display exact zero energy at 1e-30 for logarithmic plotting. Frequency is measured from actual attack batches; local defense role counts are also preserved in CSV. Concentration depends on saved precision; the report includes loading-offset and BF16-evaluation checks. The new comparison counts each saved phase-start displacement once, retaining its magnitude and all trajectory directions; it does not correct for uneven save spacing or measure adjacent optimizer updates. The defense curves describe the outer defense path. All new figures have sibling PDFs and per-case summary CSVs.
