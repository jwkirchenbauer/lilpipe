"""One-time label-only paired-figure repair using unchanged canonical renderers."""

import hashlib
import json
import os
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import aggregate


SURFACE_KINDS = ("local-reference", "local-delta", "broad-random", "broad-delta")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_identity(record):
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in record.items() if key != "label"},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def wrapped_record(record):
    result = dict(record)
    result["label"] = (
        textwrap.fill(
            record["label"], width=30, break_long_words=False, break_on_hyphens=False
        )
        + "\n"
    )
    assert record_identity(result) == record_identity(record)
    return result


def surface_identity(data):
    grids = [
        {
            "key": list(key),
            "coordinates": grid["coordinates"].tolist(),
            "values": grid["values"].tolist(),
            "center": grid["center"],
        }
        for key, grid in sorted(data["grids"].items())
    ]
    value = {key: data[key] for key in ("broad", "delta", "frames", "radius", "limits")}
    return hashlib.sha256(
        json.dumps(
            {**value, "grids": grids},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def present_titles(figure):
    wrapped = False
    for axis in figure.axes:
        title = axis.get_title()
        if ", attack " in title:
            prefix, suffix = title.rsplit(", attack ", 1)
            axis.set_title(prefix.rstrip() + "\nAttack " + suffix, fontsize=9)
            wrapped = True
    if wrapped:
        figure._suptitle.set_y(0.995)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    titles = [
        axis.title.get_window_extent(renderer)
        for axis in figure.axes
        if axis.get_title()
    ]
    for index, first in enumerate(titles):
        for second in titles[index + 1 :]:
            if first.overlaps(second):
                raise ValueError("Paired panel titles overlap after wrapping")
        if first.overlaps(figure._suptitle.get_window_extent(renderer)):
            raise ValueError("Paired title overlaps figure header")
        if (
            first.x0 < figure.bbox.x0
            or first.x1 > figure.bbox.x1
            or first.y0 < figure.bbox.y0
            or first.y1 > figure.bbox.y1
        ):
            raise ValueError("Paired title extends outside figure bounds")
    header = figure._suptitle.get_window_extent(renderer)
    if not figure.bbox.contains(header.x0, header.y0) or not figure.bbox.contains(
        header.x1, header.y1
    ):
        raise ValueError("Paired figure header extends outside figure bounds")
    return {
        "panel_titles": len(titles),
        "title_overlap": False,
        "title_fontsize": 9,
        "record_label_wrap_columns": 30,
        "global_header_y": figure._suptitle.get_position()[1],
    }


def expected_names(pairs):
    return {
        f"{pair['name']}-{kind}.{extension}"
        for pair in pairs
        for kind in (*SURFACE_KINDS, "trajectories")
        for extension in ("png", "pdf")
    }


def image_hashes(folder, names):
    assert folder.is_dir() and not folder.is_symlink()
    assert {path.name for path in folder.iterdir()} == set(
        names
    ), "Paired final must contain exactly the expected 40 images"
    for name in names:
        path = folder / name
        assert path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    return {name: digest(folder / name) for name in sorted(names)}


def load_complete_pairs(run):
    program = aggregate.read_json(run / "study.json")
    probes = aggregate.read_json(run / "probes.json")
    source = run / "source/misc/tar_compute_compare.py"
    assert (
        digest(source) == program["source_sha256"]["source/misc/tar_compute_compare.py"]
    )
    compare = aggregate.load_compare_module(run)
    records = {}
    for name, control in program["comparison_controls"].items():
        record = compare.checkpoint_record(
            name, control["folder"], control["schedule_steps"], control
        )
        record["label"] = control.get("label", "Reproduced")
        records[name] = record
    attacks = []
    for name, case in program["cases"].items():
        if case["role"] != "attack":
            continue
        attacks.append(name)
        defense = run / "cases" / f"hparam-{case['condition'].lower()}-defense"
        aggregate.validate_training_guard(defense, program["cases"][defense.name])
        record = compare.checkpoint_record(
            name, run / "cases" / name, case["max_steps"]
        )
        record["label"] = case["label"]
        _, _, profile = compare._profile_adapter_record(run, name)
        assert (
            profile["step_profiles_complete"]
            and profile["step_profiles_comparable"]
            and not profile["issues"]
        )
        records[name] = record
    validations = {
        name: aggregate.validate_dynamics(record, run, program, probes, compare)
        for name, record in records.items()
    }
    saved = aggregate.read_json(run / "checkpoint-comparison-validation.json")
    assert saved["coverage"] == "complete" and set(saved["cases"]) == set(records)
    assert all(
        not value["issues"] and value["physical_comparable"]
        for value in saved["cases"].values()
    )
    pairs = compare.checkpoint_pairs(
        records, attacks, list(program["comparison_controls"])
    )
    assert len(attacks) == len(pairs) == 4
    assert {item["name"] for item in saved["pairs"]} == {pair["name"] for pair in pairs}
    return compare, pairs, validations, digest(source)


def publish_directories(final, staging, archive):
    """Archive the complete old directory, then atomically install the staged set."""
    assert not archive.exists(), "Refuse to overwrite original-layout archive"
    os.rename(final, archive)
    try:
        os.rename(staging, final)
    except BaseException:
        os.rename(archive, final)
        raise


def render_and_publish(run):
    import matplotlib

    matplotlib.use("Agg")

    compare, pairs, validations, source_sha = load_complete_pairs(run)
    parent = run / "checkpoint-comparisons"
    final, archive = parent / "final", parent / "layout-v1-original"
    provenance = parent / "layout-v2-publication.json"
    assert (
        not archive.exists() and not provenance.exists()
    ), "Paired layout repair is one-time; preserve its archive/provenance"
    assert not (
        run / "aggregate-comparison-v2/final"
    ).exists(), "Refuse to overwrite aggregate-v2 output"
    names = expected_names(pairs)
    old_hashes = image_hashes(final, names)
    metadata = {
        "state": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source_sha256": source_sha,
        "change": "Only copied record.label wrapping, drawn panel-title font/linebreaks and global header position. Canonical numerical grids/scales/camera and trajectory functions are unchanged.",
        "archive": str(archive),
        "final": str(final),
        "numeric_csv_json_reports_modified": False,
        "old_control_outputs_modified": False,
        "images": {},
        "surface_data_identity": {},
        "title_audits": {},
    }
    original_save = compare.save_figure

    def readable_save(figure, output):
        metadata["title_audits"][Path(output).name] = present_titles(figure)
        original_save(figure, output)

    with tempfile.TemporaryDirectory(
        prefix=".layout-v2-stage-", dir=parent
    ) as temporary:
        staging = Path(temporary) / "figures"
        staging.mkdir()
        compare.save_figure = readable_save
        try:
            with matplotlib.rc_context({"axes.titlesize": 9}):
                for pair in pairs:
                    before = {
                        record["name"]: record_identity(record)
                        for record in pair["records"]
                    }
                    presented = {
                        **pair,
                        "records": [
                            wrapped_record(record) for record in pair["records"]
                        ],
                    }
                    for kind in SURFACE_KINDS:
                        original_data = surface_identity(
                            compare.checkpoint_surface_data(pair, validations, kind)
                        )
                        presented_data = surface_identity(
                            compare.checkpoint_surface_data(
                                presented, validations, kind
                            )
                        )
                        assert (
                            original_data == presented_data
                        ), "Labels changed numerical surface data or scales"
                        metadata["surface_data_identity"][
                            f"{pair['name']}-{kind}"
                        ] = original_data
                        compare.checkpoint_surface_figure(
                            presented,
                            validations,
                            kind,
                            staging / f"{pair['name']}-{kind}",
                        )
                    compare.checkpoint_trajectory_figure(
                        presented, validations, staging / f"{pair['name']}-trajectories"
                    )
                    assert before == {
                        record["name"]: record_identity(record)
                        for record in pair["records"]
                    }
        finally:
            compare.save_figure = original_save
        new_hashes = image_hashes(staging, names)
        assert (
            image_hashes(final, names) == old_hashes
        ), "Original paired images changed while staging"
        metadata["images"] = {
            name: {"original_sha256": old_hashes[name], "new_sha256": new_hashes[name]}
            for name in sorted(names)
        }
        publish_directories(final, staging, archive)
    assert (
        image_hashes(archive, names) == old_hashes
        and image_hashes(final, names) == new_hashes
    )
    temporary = provenance.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, provenance)
    print(
        json.dumps(
            {
                "state": "complete",
                "paired_images": len(names),
                "archive": str(archive),
                "provenance": str(provenance),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    import sys

    assert os.environ.get("FLUX_JOB_ID"), "Allocated launcher job required"
    render_and_publish(Path(sys.argv[1]).resolve())
