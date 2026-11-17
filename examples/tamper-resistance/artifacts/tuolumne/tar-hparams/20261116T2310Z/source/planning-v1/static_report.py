"""One common-scale static comparison of B/R/I/O/RIO plus the uploaded context."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from static_worker import CONVENTION, RADII, dump, read, sha, validate_grid


def common_limits(values, symmetric=False):
    assert values and all(math.isfinite(value) for value in values)
    if symmetric:
        bound = max(max(map(abs, values)) * 1.05, 1e-5)
        return -bound, bound
    low, high = min(values), max(values)
    pad = max((high - low) * 0.05, 1e-5)
    return low - pad, high + pad


def load_cases(run):
    program = read(run / "study.json")
    original = (
        Path(program["original_run"])
        / "loss-landscape/defense-step125-random-seed42-v3"
    )
    collaborator = (
        run.parents[3] / "artifacts/tuolumne/tar-collaborator/20261116T1725Z/static"
    )
    baseline = read(original / "provenance.json")
    entries = [("B", "B | retain 2.5, inner 1e-6, outer 5e-6", original, baseline)]
    for condition in ("R", "I", "O", "RIO"):
        name = f"hparam-{condition.lower()}-defense"
        assert (
            read(run / "cases" / name / "status/static.json")["state"] == "complete"
        ), name
        folder = run / "static/cases" / name
        validation = read(folder / "validation.json")
        assert validation["state"] == "passed" and validation["points"] == 648
        assert validation["grid_sha256"] == sha(folder / "grid.csv")
        assert validation["protocol_sha256"] == sha(folder / "protocol.json")
        metadata = read(folder / "protocol.json")
        assert metadata["baseline_tokenizer_and_collation_match"]
        assert metadata["default_dataset_fixed10_matches_pinned"]
        assert metadata["n_pts"] == 9 and metadata["seed"] == 42
        assert metadata["batch_size"] == 2 and metadata["max_seq_len"] == 512
        assert (
            metadata["source_sha256"]["get_ll.py"]
            == baseline["source_sha256"]["get_ll.py"]
        )
        values = program["hparam_matrix"][condition]
        title = f"{condition} | retain {values['retain_coefficient']:g}, inner {values['inner_learning_rate']:g}, outer {values['outer_learning_rate']:g}"
        entries.append((condition, title, folder, metadata))
    supplied = read(collaborator / "provenance.json")
    assert supplied["get_ll_source_sha256"] == baseline["source_sha256"]["get_ll.py"]
    entries.append(
        (
            "HF context",
            "Uploaded HF | multifactor checkpoint contrast",
            collaborator,
            supplied,
        )
    )
    records = []
    original_keys = None
    baseline_batches = None
    for label, title, folder, metadata in entries:
        assert metadata["document_sha256"] == baseline["document_sha256"]
        assert metadata["dataset_revision"] == baseline["dataset_revision"]
        assert metadata["radii"] == RADII and metadata["n_pts"] == 9
        assert metadata["batch_size"] == 2 and metadata["max_seq_len"] == 512
        assert (
            metadata["torch"] == baseline["torch"]
            and metadata["hip"] == baseline["hip"]
        )
        rows = validate_grid(folder / "grid.csv")
        keys = {(row["radius"], row["c0"], row["c1"]) for row in rows}
        if original_keys is None:
            original_keys = keys
        assert keys == original_keys, label
        if "evaluated_batch_sha256" in metadata:
            baseline_batches = baseline_batches or metadata["evaluated_batch_sha256"]
            assert metadata["evaluated_batch_sha256"] == baseline_batches
        records.append(
            {
                "label": label,
                "title": title,
                "folder": folder,
                "metadata": metadata,
                "rows": rows,
                "center_ce": next(
                    row["loss"] for row in rows if row["c0"] == row["c1"] == 0
                ),
            }
        )
    return records


def render(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, colors

    records = load_cases(run)
    output = run / "static/comparison"
    output.mkdir(parents=True, exist_ok=True)
    validation = {
        "state": "complete",
        "cases": {},
        "scales": {},
        "convention": CONVENTION,
        "scale_policy": "Common across all six checkpoints within each radius, including the uploaded HF context. Radii have distinct scales. Delta CE subtracts each checkpoint's measured center; common symmetric color limits.",
        "camera": {"elevation": 27, "azimuth": -58},
        "all_inputs_complete": True,
    }
    links = []
    for radius in RADII:
        selections = [
            [row for row in record["rows"] if row["radius"] == radius]
            for record in records
        ]
        absolute = common_limits([row["loss"] for rows in selections for row in rows])
        relative = common_limits(
            [
                row["loss"] - record["center_ce"]
                for record, rows in zip(records, selections, strict=True)
                for row in rows
            ],
            symmetric=True,
        )
        validation["scales"][str(radius)] = {
            "absolute_ce": absolute,
            "center_relative_ce": relative,
            "coordinate_limits": [-radius, radius],
        }
        for kind, limits, cmap in (
            ("absolute", absolute, "viridis"),
            ("delta", relative, "coolwarm"),
        ):
            subplot_kw = {"projection": "3d"} if kind == "absolute" else {}
            figure, axes = plt.subplots(
                2, 3, figsize=(19.5, 11.8), subplot_kw=subplot_kw
            )
            figure.subplots_adjust(
                left=0.045, right=0.91, bottom=0.085, top=0.87, wspace=0.25, hspace=0.35
            )
            for axis, record, rows in zip(axes.flat, records, selections, strict=True):
                coordinates = sorted({row["c0"] for row in rows})
                lookup = {(row["c0"], row["c1"]): row["loss"] for row in rows}
                x, y = np.meshgrid(coordinates, coordinates)
                z = np.array(
                    [[lookup[(c0, c1)] for c0 in coordinates] for c1 in coordinates]
                )
                if kind == "absolute":
                    axis.plot_surface(
                        x,
                        y,
                        z,
                        cmap=cmap,
                        vmin=limits[0],
                        vmax=limits[1],
                        linewidth=0,
                        antialiased=True,
                    )
                    axis.view_init(elev=27, azim=-58)
                    axis.set_zlim(*limits)
                    axis.set_zlabel("Forget CE (nats)", labelpad=9)
                    axis.tick_params(axis="z", labelsize=8, pad=2)
                    axis.set_box_aspect((1, 1, 0.75))
                else:
                    z -= record["center_ce"]
                    levels = np.linspace(*limits, 17)
                    axis.contourf(
                        x,
                        y,
                        z,
                        levels=levels,
                        cmap=cmap,
                        vmin=limits[0],
                        vmax=limits[1],
                    )
                    if np.ptp(z) > 1e-8:
                        axis.contour(
                            x,
                            y,
                            z,
                            levels=levels,
                            linewidths=0.4,
                            colors="black",
                            alpha=0.4,
                        )
                    axis.set_aspect("equal")
                axis.set_xlim(-radius, radius)
                axis.set_ylim(-radius, radius)
                axis.set_xlabel("Static random coordinate 1", labelpad=7)
                axis.set_ylabel("Static random coordinate 2", labelpad=7)
                axis.set_title(
                    f"{record['title']}\ncenter CE {record['center_ce']:.4f}",
                    fontsize=10,
                    pad=12,
                )
                axis.tick_params(labelsize=8)
            description = (
                "Absolute CE | shared z and color limits"
                if kind == "absolute"
                else "CE minus each measured center | shared symmetric color limits"
            )
            figure.suptitle(
                f"Original static BF16 recipe | half-width {radius:g}\n{description}",
                fontsize=16,
                y=0.965,
            )
            figure.text(
                0.48,
                0.025,
                "Same seed42 random recipe and orientation; independent from FP32 orthonormal dynamic slices. Uploaded HF checkpoint is contextual, not a controlled hparam cell.",
                ha="center",
                fontsize=10,
            )
            color_axis = figure.add_axes([0.94, 0.19, 0.014, 0.58])
            figure.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(*limits), cmap=cmap),
                cax=color_axis,
                label=(
                    "Forget CE (nats)"
                    if kind == "absolute"
                    else "Center-relative CE (nats)"
                ),
            )
            stem = f"radius-{radius:g}-{kind}"
            for extension in ("png", "pdf"):
                figure.savefig(
                    output / f"{stem}.{extension}",
                    dpi=170,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
            plt.close(figure)
        links.append(
            f"| {radius:g} | [absolute CE](radius-{radius:g}-absolute.png) / [PDF](radius-{radius:g}-absolute.pdf) | [center-relative CE](radius-{radius:g}-delta.png) / [PDF](radius-{radius:g}-delta.pdf) |"
        )
    summary = [
        "# Original static radius-series comparison",
        "",
        "B is the completed local baseline. R, I, O and RIO change retain coefficient, inner learning rate, outer learning rate, or all three while holding the local sampler and original BF16 training path fixed. The uploaded HF checkpoint also differs in sampling/batching and is shown only as context.",
        "",
        CONVENTION,
        "",
        "Every panel uses the same fixed10 documents, truncation512, batch size2 and seed42. Absolute CE and z scales match across all six checkpoints within each radius. Center-relative panels subtract each measured center and share a symmetric scale; small relief therefore remains distinguishable from a vertical loss shift. These sampled translated slices describe a fixed forget CE objective, not a change to the global loss function.",
        "",
        "| Checkpoint | Center CE |",
        "|---|---:|",
    ]
    for record in records:
        validation["cases"][record["label"]] = {
            "folder": str(record["folder"]),
            "grid_sha256": sha(record["folder"] / "grid.csv"),
            "center_ce": record["center_ce"],
            "points": len(record["rows"]),
        }
        summary.append(f"| {record['label']} | {record['center_ce']:.6f} |")
    summary.extend(
        [
            "",
            "| Static half-width | Absolute CE | Center-relative CE |",
            "|---:|---|---|",
            *links,
            "",
            "One defender seed42 and one optimizer seed1 limit the inference. R/I/O are paired deltas around B; RIO alone does not identify all interactions. Random directions are regenerated by the exact original BF16 recipe rather than read from the separate saved dynamic bases.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(summary))
    validation["artifacts"] = {
        path.name: sha(path)
        for path in sorted(output.iterdir())
        if path.suffix in {".png", ".pdf", ".md"}
    }
    dump(output / "validation.json", validation)
    dump(
        run / "status/static-comparison.json",
        {"state": "complete", "figures": 16, "checkpoints": 6},
    )
    print(json.dumps(validation, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert os.environ.get("FLUX_JOB_ID"), "Allocated launcher job required"
    render(args.run.resolve())


if __name__ == "__main__":
    main()
