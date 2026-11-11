# Landscape figure index

All planes are centered at the saved defender. Canonical first-update and fitted
planes use the full-same attacker. All-case exports share CE limits 0–82.
Each radius is a per-axis L2 half-width; square corners extend farther.
The reference basis is literally shared; fitted and first-update bases are
specific to each attack. Numerical JSON records contain actual checkpoint losses,
projection capture, and CE reconstruction errors.

| Defender | 3D | Heatmap | Equal-radius sections | Numerical data |
| --- | --- | --- | --- | --- |
| Starting K16 | [PNG](control/slices-3d.png) / [PDF](control/slices-3d.pdf) | [PNG](control/slices.png) | [PNG](control/matched-radius-sections.png) | [JSON](control/geometry.json) |
| Full terminal entropy, fixed retain | [PNG](full-maxent-terminal/slices-3d.png) / [PDF](full-maxent-terminal/slices-3d.pdf) | [PNG](full-maxent-terminal/slices.png) | [PNG](full-maxent-terminal/matched-radius-sections.png) | [JSON](full-maxent-terminal/geometry.json) |
| Full terminal CE, fixed retain | [PNG](full-ce-terminal/slices-3d.png) / [PDF](full-ce-terminal/slices-3d.pdf) | [PNG](full-ce-terminal/slices.png) | [PNG](full-ce-terminal/matched-radius-sections.png) | [JSON](full-ce-terminal/geometry.json) |
| Full worst-path CE, fixed retain | [PNG](full-ce-worst/slices-3d.png) / [PDF](full-ce-worst/slices-3d.pdf) | [PNG](full-ce-worst/slices.png) | [PNG](full-ce-worst/matched-radius-sections.png) | [JSON](full-ce-worst/geometry.json) |
| Mini terminal entropy, fixed retain | [PNG](mini-maxent-terminal/slices-3d.png) / [PDF](mini-maxent-terminal/slices-3d.pdf) | [PNG](mini-maxent-terminal/slices.png) | [PNG](mini-maxent-terminal/matched-radius-sections.png) | [JSON](mini-maxent-terminal/geometry.json) |
| Mini terminal CE, fixed retain | [PNG](mini-ce-terminal/slices-3d.png) / [PDF](mini-ce-terminal/slices-3d.pdf) | [PNG](mini-ce-terminal/slices.png) | [PNG](mini-ce-terminal/matched-radius-sections.png) | [JSON](mini-ce-terminal/geometry.json) |
| Mini worst-path CE, fixed retain | [PNG](mini-ce-worst/slices-3d.png) / [PDF](mini-ce-worst/slices-3d.pdf) | [PNG](mini-ce-worst/slices.png) | [PNG](mini-ce-worst/matched-radius-sections.png) | [JSON](mini-ce-worst/geometry.json) |
| Full terminal entropy, refreshed retain | [PNG](full-maxent-terminal-fresh-retain/slices-3d.png) / [PDF](full-maxent-terminal-fresh-retain/slices-3d.pdf) | [PNG](full-maxent-terminal-fresh-retain/slices.png) | [PNG](full-maxent-terminal-fresh-retain/matched-radius-sections.png) | [JSON](full-maxent-terminal-fresh-retain/geometry.json) |
| Full worst-path CE, refreshed retain | [PNG](full-ce-worst-fresh-retain/slices-3d.png) / [PDF](full-ce-worst-fresh-retain/slices-3d.pdf) | [PNG](full-ce-worst-fresh-retain/slices.png) | [PNG](full-ce-worst-fresh-retain/matched-radius-sections.png) | [JSON](full-ce-worst-fresh-retain/geometry.json) |

The utility-passing comparison shares CE limits 0–32 across all three models
and the additional full-new view. The range is calculated from their measured
grids, without per-model vertical rescaling.

- [Starting K16: 3D](control/utility-comparison/slices-3d.png) / [PDF](control/utility-comparison/slices-3d.pdf).
- [Full terminal entropy, refreshed retain: 3D](full-maxent-terminal-fresh-retain/utility-comparison/slices-3d.png) / [PDF](full-maxent-terminal-fresh-retain/utility-comparison/slices-3d.pdf).
- [Full worst-path CE, refreshed retain: 3D](full-ce-worst-fresh-retain/utility-comparison/slices-3d.png) / [PDF](full-ce-worst-fresh-retain/utility-comparison/slices-3d.pdf).

The [strong new-data attacker](full-maxent-terminal-fresh-retain/attack-variants/full-new/utility-comparison/slices-3d.png)
has first-update and fitted planes with three loss cohorts, including the eight
examples it actually updates on. [Heatmap](full-maxent-terminal-fresh-retain/attack-variants/full-new/utility-comparison/slices.png)
· [PDF](full-maxent-terminal-fresh-retain/attack-variants/full-new/utility-comparison/slices-3d.pdf)
· [Numerical data](full-maxent-terminal-fresh-retain/attack-variants/full-new/geometry.json).
These examples were excluded from this continuation; their earlier training
history is not ruled out. A projected checkpoint is a counterfactual model;
its plotted loss need not match the actual checkpoint loss.

[Findings](../findings.md) · [All capability measurements](../report/report.md)
· [Rendering provenance](rendering.json).
