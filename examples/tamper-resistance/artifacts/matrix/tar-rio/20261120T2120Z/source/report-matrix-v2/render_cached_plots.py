"""Render compact comparison titles from already validated numerical tables."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.titlesize"] = 9

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/misc"
sys.path.insert(0, str(SOURCE))
import tar_compute_compare as comparison

LABELS = {
    "rio-k16": "RIO K16",
    "rio-k4-lr1e-6": "K4",
    "rio-k64-lr1e-6": "K64",
    "rio-k256-lr1e-6": "K256",
    "rio-prefix-forward-legacy-lr1e-6": "Forward / legacy",
    "rio-prefix-reverse-legacy-lr1e-6": "Reverse / legacy",
    "rio-fresh-matched-lr1e-6": "Fresh / matched",
    "rio-prefix-forward-matched-lr1e-6": "Forward / matched",
    "rio-prefix-reverse-matched-lr1e-6": "Reverse / matched",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    validation = comparison.read_json(ROOT / "rio-comparison-validation.json")
    assert validation["coverage"] == "complete"
    assert not any(row.get("issues") for row in validation["cases"].values())
    references = comparison.read_json(ROOT / "report-matrix-references.json")
    assert references["state"] == "complete"
    program = comparison.read_json(ROOT / "study.json")
    records = {
        name: comparison.checkpoint_record(
            name, item["folder"], item["schedule_steps"], item
        )
        for name, item in program["comparison_controls"].items()
    }
    attacks = {
        name: case
        for name, case in program["cases"].items()
        if case["role"] == "attack"
    }
    for name, case in attacks.items():
        records[name] = comparison.checkpoint_record(
            name, ROOT / "cases" / name, case["max_steps"]
        )
    inputs = {
        ROOT / "rio-comparison-validation.json",
        ROOT / "rio-summary.csv",
        ROOT / "rio-trajectories.csv",
    }
    for name, record in records.items():
        record["label"] = LABELS[name]
        folder = record["folder"]
        inputs.update(
            folder / filename
            for filename in (
                "study.json",
                "geometry/geometry.json",
                "surfaces/actual-losses.csv",
                "surfaces/local-grids.csv",
                "surfaces/random-evolution.csv",
                "evaluation-scores.csv",
                "projection-quality.csv",
            )
        )
    before = {str(path): digest(path) for path in inputs}
    pairs = comparison.checkpoint_pairs(records, list(attacks), ["rio-k16"])
    for names, label in (
        (
            ["rio-k16", "rio-k4-lr1e-6", "rio-k64-lr1e-6", "rio-k256-lr1e-6"],
            "rio-k-four-way",
        ),
        (
            [
                "rio-k16",
                "rio-prefix-forward-legacy-lr1e-6",
                "rio-fresh-matched-lr1e-6",
                "rio-prefix-forward-matched-lr1e-6",
            ],
            "rio-data-optimizer-2x2",
        ),
    ):
        selected = [records[name] for name in names]
        pairs.append(
            dict(
                name=label,
                records=selected,
                schedule_steps=1000,
                local_frames=sorted(
                    set.intersection(*(set(r["local_frames"]) for r in selected))
                ),
                broad_frames=sorted(
                    set.intersection(*(set(r["broad_frames"]) for r in selected))
                ),
            )
        )
    outputs = []
    for pair in pairs:
        for kind in ("local-reference", "local-delta", "broad-random", "broad-delta"):
            output = ROOT / "rio-figures" / (pair["name"] + "-" + kind)
            comparison.checkpoint_surface_figure(
                pair, validation["cases"], kind, output
            )
            outputs.extend(output.with_suffix(ext) for ext in (".png", ".pdf"))
    assert before == {str(path): digest(path) for path in inputs}
    manifest = dict(
        state="complete",
        numerical_inputs_unchanged=True,
        input_sha256=before,
        display_labels=LABELS,
        renderer_source_sha256=digest(SOURCE / "tar_compute_compare.py"),
        presentation_source_sha256=digest(Path(__file__)),
        outputs={str(path): digest(path) for path in outputs},
    )
    target = ROOT / "plot-rendering-validation.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    references["plot_rendering_validation"] = str(target)
    (ROOT / "report-matrix-references.json").write_text(
        json.dumps(references, indent=2) + "\n"
    )
    print(
        "Completed compact titles for",
        len(outputs),
        "figure files; numerical inputs unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    main()
