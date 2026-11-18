"""Analyze per-call trajectory Gram matrices without loading model weights.

Usage: tar_inner_geometry.py report --spec experiment.json --output results
The version-1 spec has aggregates (paths to recorder JSON files) and actual_views
with two lists, matched and long. Entries in each list have the recorder's
reference name and a positive integer step. actual_states lists all archived
references with the same fields, including projection-only states. Paths resolve from the working
directory. The matched view must have the same steps as the recorded inner call.
"""

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

import tar_basis_structure as basis


CAVEAT = (
    "Each inner trajectory starts immediately before its own outer update. The "
    "archived attack starts at the final defender, after the last outer update. "
    "Even the last inner call therefore has a different anchor. Observed "
    "directional similarity cannot establish how both would move from the same state."
)
REPRESENTATIONS = {"raw_saved": "raw", "bf16_evaluated": "bf16"}
TABLES = {
    "anchor_distance": ("outer_call", "representation", "squared_distance", "distance"),
    "comparisons": (
        "outer_call",
        "representation",
        "motion",
        "normalization",
        "actual_view",
        "requested_rank",
        "inner_rank",
        "actual_rank",
        "inner_overlap_fraction",
        "actual_overlap_fraction",
        "actual_energy_captured",
        "mean_actual_vector_capture",
        "first_step_cosine",
    ),
    "angles": (
        "outer_call",
        "representation",
        "motion",
        "normalization",
        "actual_view",
        "requested_rank",
        "component",
        "degrees",
    ),
    "capture": (
        "outer_call",
        "representation",
        "motion",
        "normalization",
        "actual_view",
        "requested_rank",
        "reference",
        "start_step",
        "end_step",
        "squared_norm",
        "capture_fraction",
    ),
    "spectrum": (
        "outer_call",
        "representation",
        "motion",
        "normalization",
        "view",
        "numerical_rank",
        "component",
        "eigenvalue",
        "energy_fraction",
    ),
    "inner_similarity": (
        "outer_call",
        "representation",
        "motion",
        "left_start",
        "left_end",
        "right_start",
        "right_end",
        "cosine",
    ),
}


TABLES["family_comparisons"] = ("family", "coordinates", *TABLES["comparisons"])
TABLES["family_basis_energy"] = (
    "outer_call",
    "representation",
    "motion",
    "normalization",
    "actual_view",
    "requested_rank",
    "basis_side",
    "basis_rank",
    "family",
    "coordinates",
    "equal_axis_energy_share",
)


def validate_record(record, actual_views, actual_states):
    """Verify row identity, layout, and view membership before indexing blocks."""
    if record["status"] != "complete" or record["schema_version"] != 1:
        raise ValueError("Expected a complete version-1 recorder aggregate")
    steps = record["inner_steps"]
    names = record["reference_names"]
    if [r["name"] for r in record["references"]] != names or len(set(names)) != len(
        names
    ):
        raise ValueError("Reference names and metadata disagree")
    if type(record["outer_call"]) is not int or record["outer_call"] < 1:
        raise ValueError("Expected a positive integer outer_call")
    if type(record["coordinates"]) is not int or record["coordinates"] < 1:
        raise ValueError("Expected a positive coordinate count")
    if (
        not steps
        or steps != sorted(set(steps))
        or any(type(s) is not int or s <= 0 for s in steps)
    ):
        raise ValueError("Inner steps must be increasing positive integers")
    if set(actual_views) != {"matched", "long"}:
        raise ValueError("Specify matched and long actual views")
    selected = set()
    for name, rows in {**actual_views, "all_states": actual_states}.items():
        numbers = [r["step"] for r in rows]
        refs = [r["name"] for r in rows]
        if (
            not rows
            or any(type(s) is not int or s <= 0 for s in numbers)
            or numbers != sorted(set(numbers))
            or len(set(refs)) != len(refs)
        ):
            raise ValueError(f"Invalid saved steps or references in actual view {name}")
        if name == "matched" and numbers != steps:
            raise ValueError("Matched view must use the recorded inner steps")
        selected.update((r["name"], r["step"]) for r in rows)
    states = {(r["name"], r["step"]) for r in actual_states}
    if selected != states or {r["name"] for r in actual_states} != set(names):
        raise ValueError(
            "Actual states must cover all recorded references and agree with fit-view steps"
        )
    if any(
        "step" in r and (r["name"], r["step"]) not in states
        for r in record["references"]
    ):
        raise ValueError(
            "Declared actual steps differ from recorder reference metadata"
        )


def combined_gram(record, representation):
    n, m = len(record["inner_steps"]), len(record["reference_names"])
    blocks = record["representations"][REPRESENTATIONS[representation]]
    inner, cross, actual = (
        np.asarray(blocks[k], dtype=np.float64)
        for k in ("inner_gram", "cross_gram", "reference_gram")
    )
    if inner.shape != (n, n) or cross.shape != (n, m) or actual.shape != (m, m):
        raise ValueError("Gram block dimensions differ from the recorded row counts")
    gram = np.block([[inner, cross], [cross.T, actual]])
    if not np.isfinite(gram).all():
        raise ValueError("Nonfinite Gram matrix")
    scale = float(np.max(np.abs(gram)))
    if not np.allclose(gram, gram.T, rtol=1e-12, atol=scale * 1e-13):
        raise ValueError("Asymmetric Gram matrix")
    gram = (gram + gram.T) * 0.5
    if np.linalg.eigvalsh(gram).min() < -scale * 1e-10:
        raise ValueError("Gram matrix is not positive semidefinite")
    return gram


def transform(indices, size, intervals):
    """Rows select cumulative displacements, or adjacent net updates per view."""
    matrix = np.eye(size)[indices]
    if intervals:
        matrix[1:] -= matrix[:-1].copy()
    return matrix


def fit(gram, equal_unit):
    """Return orthonormal axes in trajectory coordinates and their spectrum."""
    norms = np.sqrt(np.maximum(np.diag(gram), 0))
    weights = np.ones_like(norms)
    if equal_unit:
        np.divide(1, norms, where=norms > 0, out=weights)
    weighted = weights[:, None] * gram * weights[None, :]
    scale = float(np.max(np.diag(weighted)))
    if scale <= 0:
        return np.empty((len(gram), 0)), np.zeros(len(gram))
    coeff, info = basis.fit_subspace(weighted / scale, np.arange(len(gram)))
    return (
        weights[:, None] * coeff / np.sqrt(scale),
        np.array(info["eigenvalues"]) * scale,
    )


def capture_fractions(inner, cross, energies):
    fractions = np.divide(
        np.square(inner.T @ cross).sum(axis=0),
        energies,
        where=energies > 0,
        out=np.full(len(energies), np.nan),
    )
    if np.any(fractions > 1 + 1e-5):
        raise ValueError("Projection capture exceeds target energy")
    return np.clip(fractions, 0, 1)


def capture_and_overlap(inner, actual, cross, actual_gram, rank):
    n_inner, n_actual = inner.shape[1], actual.shape[1]
    if rank != "full":
        n_inner = n_actual = min(int(rank), n_inner, n_actual)
    inner, actual = inner[:, :n_inner], actual[:, :n_actual]
    cosines = np.linalg.svd(inner.T @ cross @ actual, compute_uv=False)
    if np.any(cosines > 1 + 1e-5):
        raise ValueError("Principal-angle cosine outside numerical tolerance")
    cosines = np.clip(cosines, 0, 1)
    mass = float(sum(cosines**2))
    energies = np.maximum(np.diag(actual_gram), 0)
    positive = energies > 0
    fractions = capture_fractions(inner, cross, energies)
    return (
        {
            "inner_rank": n_inner,
            "actual_rank": n_actual,
            "inner_overlap_fraction": mass / n_inner if n_inner else None,
            "actual_overlap_fraction": mass / n_actual if n_actual else None,
            "actual_energy_captured": (
                float(np.dot(energies[positive], fractions[positive]) / energies.sum())
                if positive.any()
                else None
            ),
            "mean_actual_vector_capture": (
                float(fractions[positive].mean()) if positive.any() else None
            ),
        },
        np.degrees(np.arccos(cosines)),
        energies,
        fractions,
    )


def cosine(dot, left_energy, right_energy):
    norm = np.sqrt(max(0.0, left_energy)) * np.sqrt(max(0.0, right_energy))
    return float(np.clip(dot / norm, -1, 1)) if norm else None


def capture_rows(scope, refs, intervals, energies, fractions):
    for i, (ref, energy, fraction) in enumerate(
        zip(refs, energies, fractions, strict=True)
    ):
        yield dict(
            **scope,
            reference=ref["name"],
            start_step=refs[i - 1]["step"] if intervals and i else 0,
            end_step=ref["step"],
            squared_norm=float(energy),
            capture_fraction=float(fraction) if np.isfinite(fraction) else None,
        )


def family_basis_energy(scope, side, axes, transformed_grams, families):
    """Partition the squared norms of fixed whole-model orthonormal axes."""
    rank = axes.shape[1]
    rows = []
    for family, gram in transformed_grams.items():
        share = float(np.sum(axes * (gram @ axes)) / rank) if rank else None
        if rank:
            if not np.isfinite(share) or not -1e-5 <= share <= 1 + 1e-5:
                raise ValueError(f"Invalid whole-model basis energy in family {family}")
            share = float(np.clip(share, 0, 1))
        rows.append(
            dict(
                **scope,
                basis_side=side,
                basis_rank=rank,
                family=family,
                coordinates=families[family]["coordinates"],
                equal_axis_energy_share=share,
            )
        )
    if rank and not np.isclose(
        sum(r["equal_axis_energy_share"] for r in rows), 1, rtol=1e-5, atol=0
    ):
        raise ValueError("Family shares of the whole-model basis do not sum to one")
    return rows


def analyze_call(record, actual_views, actual_states, family_only=False):
    validate_record(record, actual_views, actual_states)
    inner_steps = record["inner_steps"]
    n = len(inner_steps)
    index = {name: i + n for i, name in enumerate(record["reference_names"])}
    selections = {
        "inner": list(range(n)),
        **{
            name: [index[r["name"]] for r in rows]
            for name, rows in actual_views.items()
        },
    }
    tables = {name: [] for name in TABLES}
    families = {} if family_only else record.get("families", {})
    if (
        families
        and sum(f["coordinates"] for f in families.values()) != record["coordinates"]
    ):
        raise ValueError(
            "Family coordinate counts do not sum to the full parameter count"
        )
    for representation in REPRESENTATIONS:
        gram = combined_gram(record, representation)
        family_grams = {
            family: combined_gram(
                {**record, "representations": data["representations"]}, representation
            )
            for family, data in families.items()
        }
        if family_grams:
            np.testing.assert_allclose(
                sum(family_grams.values()),
                gram,
                rtol=1e-10,
                atol=np.max(np.abs(gram)) * 1e-13,
                err_msg="Family Gram matrices do not sum to the whole model",
            )
        gap = record["representations"][REPRESENTATIONS[representation]].get(
            "anchor_gap_squared_norm"
        )
        if gap is not None and not family_only:
            if not np.isfinite(gap) or gap < 0:
                raise ValueError("Invalid anchor gap squared norm")
            tables["anchor_distance"].append(
                dict(
                    outer_call=record["outer_call"],
                    representation=representation,
                    squared_distance=gap,
                    distance=float(np.sqrt(gap)),
                )
            )
        for motion in ("displacement", "interval_update"):
            intervals = motion == "interval_update"
            context = dict(
                outer_call=record["outer_call"],
                representation=representation,
                motion=motion,
            )
            transforms = {
                v: transform(selection, len(gram), intervals)
                for v, selection in selections.items()
            }
            within = {v: t @ gram @ t.T for v, t in transforms.items()}
            within_family = {
                v: {family: t @ block @ t.T for family, block in family_grams.items()}
                for v, t in transforms.items()
            }
            inner_gram = within["inner"]
            all_transform = transform(
                [index[r["name"]] for r in actual_states], len(gram), intervals
            )
            all_cross = transforms["inner"] @ gram @ all_transform.T
            all_energies = np.maximum(
                np.diag(all_transform @ gram @ all_transform.T), 0
            )
            starts = [0, *inner_steps[:-1]] if intervals else [0] * n
            for i in range(n):
                for j in range(i, n):
                    tables["inner_similarity"].append(
                        dict(
                            **context,
                            left_start=starts[i],
                            left_end=inner_steps[i],
                            right_start=starts[j],
                            right_end=inner_steps[j],
                            cosine=cosine(
                                inner_gram[i, j], inner_gram[i, i], inner_gram[j, j]
                            ),
                        )
                    )
            for normalization in ("norm_weighted", "equal_unit"):
                labels = dict(**context, normalization=normalization)
                bases = {}
                for v, local in within.items():
                    coeff, values = fit(local, normalization == "equal_unit")
                    bases[v] = coeff
                    total = float(values.sum())
                    for i, value in enumerate(values):
                        tables["spectrum"].append(
                            dict(
                                **labels,
                                view=v,
                                numerical_rank=coeff.shape[1],
                                component=i + 1,
                                eigenvalue=float(value),
                                energy_fraction=float(value / total) if total else None,
                            )
                        )
                for v, refs in actual_views.items():
                    cross = transforms["inner"] @ gram @ transforms[v].T
                    first = (
                        cosine(cross[0, 0], inner_gram[0, 0], within[v][0, 0])
                        if inner_steps[0] == refs[0]["step"] == 1
                        else None
                    )
                    for rank in ("1", "2", "full"):
                        scope = dict(**labels, actual_view=v, requested_rank=rank)
                        measures, angles, energies, capture = capture_and_overlap(
                            bases["inner"], bases[v], cross, within[v], rank
                        )
                        tables["comparisons"].append(
                            dict(**scope, **measures, first_step_cosine=first)
                        )
                        if family_only:
                            continue
                        if families:
                            for side, view in (("inner", "inner"), ("actual", v)):
                                axes = bases[view][:, : measures[side + "_rank"]]
                                tables["family_basis_energy"].extend(
                                    family_basis_energy(
                                        scope, side, axes, within_family[view], families
                                    )
                                )
                        for i, angle in enumerate(angles):
                            tables["angles"].append(
                                dict(**scope, component=i + 1, degrees=float(angle))
                            )
                        tables["capture"].extend(
                            capture_rows(scope, refs, intervals, energies, capture)
                        )
                if not family_only:
                    for rank in ("1", "2", "full"):
                        axes = (
                            bases["inner"]
                            if rank == "full"
                            else bases["inner"][:, : int(rank)]
                        )
                        capture = capture_fractions(axes, all_cross, all_energies)
                        scope = dict(
                            **labels, actual_view="all_states", requested_rank=rank
                        )
                        tables["capture"].extend(
                            capture_rows(
                                scope, actual_states, intervals, all_energies, capture
                            )
                        )
    if families:
        for family, data in families.items():
            restricted = {
                **record,
                "coordinates": data["coordinates"],
                "representations": data["representations"],
            }
            measures = analyze_call(
                restricted, actual_views, actual_states, family_only=True
            )
            for row in measures["comparisons"]:
                tables["family_comparisons"].append(
                    {"family": family, "coordinates": data["coordinates"], **row}
                )
    return tables


def figures(folder, tables, steps, actual_views, calls):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    files = []

    def save(fig, stem):
        for ext in ("pdf", "png"):
            path = folder / f"{stem}.{ext}"
            tmp = path.with_suffix(f".tmp.{ext}")
            fig.savefig(tmp, dpi=220)
            tmp.replace(path)
            files.append(path.name)
        plt.close(fig)

    def heat(ax, matrix, xs, ys, title, signed=False):
        cmap = plt.get_cmap("RdBu_r" if signed else "viridis").with_extremes(
            bad="#eeeeee"
        )
        image = ax.imshow(
            np.asarray(matrix, float),
            vmin=-1 if signed else 0,
            vmax=1,
            cmap=cmap,
            aspect="auto",
        )
        ax.set_xticks(range(len(xs)), xs)
        ax.set_yticks(range(len(ys)), ys)
        ax.set_title(title)
        for (y, x), val in np.ndenumerate(np.asarray(matrix, float)):
            dark = abs(val) > 0.6 if signed else val < 0.55
            ax.text(
                x,
                y,
                "—" if not np.isfinite(val) else f"{val:.2g}",
                ha="center",
                va="center",
                color="white" if dark else "#151515",
                fontsize=7,
            )
        return image

    def selected(row):
        return (
            row["motion"] == "displacement" and row["normalization"] == "norm_weighted"
        )

    comparison = {
        (r["representation"], r["outer_call"], r["actual_view"], r["requested_rank"]): r
        for r in tables["comparisons"]
        if selected(r)
    }
    horizons = {
        name: f"{len(refs)} saves\nto step {refs[-1]['step']}"
        for name, refs in actual_views.items()
    }
    fig, axes = plt.subplots(
        2, 3, figsize=(10, max(4, 1.2 * len(calls))), layout="constrained"
    )
    for i, rep in enumerate(REPRESENTATIONS):
        for j, (rank, field, title) in enumerate(
            [
                ("2", "actual_overlap_fraction", "Rank 2 · actual fraction"),
                ("full", "actual_overlap_fraction", "Full · actual fraction"),
                ("full", "inner_overlap_fraction", "Full · inner fraction"),
            ]
        ):
            image = heat(
                axes[i, j],
                [[comparison[rep, c, v, rank][field] for v in horizons] for c in calls],
                list(horizons.values()),
                calls,
                title,
            )
            if j == 0:
                axes[i, j].set_ylabel(rep.replace("_", " ") + "\nOuter call")
    fig.colorbar(
        image, ax=axes.ravel().tolist(), label="Subspace overlap fraction", shrink=0.8
    )
    save(fig, "subspace-overlap")
    energy = {
        (r["representation"], r["outer_call"], r["basis_side"], r["family"]): r[
            "equal_axis_energy_share"
        ]
        for r in tables["family_basis_energy"]
        if selected(r) and r["actual_view"] == "matched" and r["requested_rank"] == "2"
    }
    if energy:
        families = sorted({key[-1] for key in energy})
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(max(6, 2.2 * len(families)), max(4.5, 0.95 * len(calls))),
            layout="constrained",
        )
        for i, (rep, label) in enumerate(
            (("raw_saved", "Raw"), ("bf16_evaluated", "BF16"))
        ):
            for j, (side, title) in enumerate(
                (("inner", "Inner"), ("actual", "Matched actual"))
            ):
                ax = axes[i, j]
                image = heat(
                    ax,
                    [[energy.get((rep, c, side, f)) for f in families] for c in calls],
                    [f.replace("_", " ") for f in families],
                    calls,
                    f"{label} · {title}",
                )
                if j == 0:
                    ax.set_ylabel("Outer call")
                ax.tick_params(axis="x", labelrotation=25)
        fig.suptitle(
            "Module-family shares of whole-model axes\nRequested rank 2 · norm-weighted displacements"
        )
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Equal-axis average energy share",
            shrink=0.8,
        )
        save(fig, "family-basis-energy")

    capture = [
        r for r in tables["capture"] if selected(r) and r["actual_view"] == "all_states"
    ]
    peak = max(
        (r["capture_fraction"] for r in capture if r["capture_fraction"] is not None),
        default=1.0,
    )
    upper = 1.02 * peak if peak > 0 else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained", sharey=True)
    for i, rep in enumerate(REPRESENTATIONS):
        for j, rank in enumerate(("2", "full")):
            ax = axes[i, j]
            for k, c in enumerate(calls):
                line = [
                    r
                    for r in capture
                    if r["representation"] == rep
                    and r["requested_rank"] == rank
                    and r["outer_call"] == c
                ]
                ax.plot(
                    [r["end_step"] for r in line],
                    [r["capture_fraction"] for r in line],
                    marker=".",
                    lw=1.5,
                    color=plt.get_cmap("tab10")(k % 10),
                    label=f"Call {c}",
                )
            ax.set(
                xscale="log",
                ylim=(-0.02 * upper, upper),
                title=f"{rep.replace('_',' ')} · inner rank {rank}",
            )
            ax.grid(alpha=0.2)
    fig.supxlabel("Archived checkpoint step (log scale)")
    fig.supylabel("Fraction of displacement energy in the inner span")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside upper center",
        ncol=min(5, len(calls)),
        frameon=False,
    )
    save(fig, "capture-by-step")

    similarity = {
        (r["representation"], r["outer_call"], r["left_end"], r["right_end"]): r[
            "cosine"
        ]
        for r in tables["inner_similarity"]
        if r["motion"] == "interval_update"
    }
    fig, axes = plt.subplots(
        2,
        len(calls),
        squeeze=False,
        figsize=(max(4, 2.65 * len(calls)), 5.9),
        layout="constrained",
    )
    labels = [f"{a}–{b}" for a, b in zip([0, *steps[:-1]], steps)]
    for i, rep in enumerate(REPRESENTATIONS):
        for j, c in enumerate(calls):
            ax = axes[i, j]
            matrix = [
                [similarity[rep, c, min(a, b), max(a, b)] for b in steps] for a in steps
            ]
            image = heat(ax, matrix, labels, labels, f"Outer call {c}", signed=True)
            ax.tick_params(axis="x", labelrotation=70)
            if j == 0:
                ax.set_ylabel(rep.replace("_", " ") + "\nInner step interval")
    fig.supxlabel("Inner step interval")
    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Within-call net-update cosine",
        shrink=0.8,
    )
    save(fig, "within-call-update-similarity")
    return files


def run(stage, spec, folder):
    folder = Path(folder)
    states = folder / "status"
    states.mkdir(parents=True, exist_ok=True)
    with (folder / ".geometry.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if spec["version"] != 1 or not spec["aggregates"]:
            raise ValueError("Expected a version-1 experiment with per-call aggregates")
        records = []
        files = []
        for path in map(Path, spec["aggregates"]):
            data = path.read_bytes()
            record = json.loads(data)
            validate_record(record, spec["actual_views"], spec["actual_states"])
            records.append(record)
            files.append(
                {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
        records.sort(key=lambda r: r["outer_call"])
        calls = [r["outer_call"] for r in records]
        if len(set(calls)) != len(calls):
            raise ValueError("Duplicate outer calls")
        first = records[0]
        for record in records[1:]:
            for field in (
                "references",
                "inner_steps",
                "coordinates",
                "layout",
                "source_sha256",
                "reference_files",
                "comparison_anchor",
            ):
                if record.get(field) != first.get(field):
                    raise ValueError(f"Call {record['outer_call']} differs in {field}")
            for rep in REPRESENTATIONS.values():
                np.testing.assert_allclose(
                    record["representations"][rep]["reference_gram"],
                    first["representations"][rep]["reference_gram"],
                    rtol=1e-10,
                    atol=0,
                    err_msg=f"Archived reference geometry differs in call {record['outer_call']}",
                )
        identity = {
            "spec": spec,
            "inputs": files,
            "source_sha256": [
                hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, basis.__file__)
            ],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        provenance = folder / "provenance.json"
        if provenance.exists() and basis.read(provenance)["signature"] != key:
            raise ValueError(
                "Output directory belongs to different inputs or source code"
            )
        basis.publish(provenance, {"signature": key, **identity})
        summary_status = states / "geometry.json"
        if summary_status.exists():
            previous = basis.read(summary_status)
            if previous["state"] == "complete" and all(
                (folder / p).is_file() for p in previous["files"]
            ):
                return
        try:
            tables = {k: [] for k in TABLES}
            basis.publish(summary_status, {"state": "running", "stage": "analysis"})
            for record in records:
                call = record["outer_call"]
                path = folder / "per-call" / f"outer-{call:06d}.json"
                if not path.exists():
                    per_call = analyze_call(
                        record, spec["actual_views"], spec["actual_states"]
                    )
                    basis.publish(
                        path, {"signature": key, "outer_call": call, "tables": per_call}
                    )
                else:
                    saved = basis.read(path)
                    if saved["signature"] != key or saved["outer_call"] != call:
                        raise ValueError(f"Wrong saved result for outer call {call}")
                    per_call = saved["tables"]
                for name, rows in per_call.items():
                    tables[name].extend(rows)
                basis.publish(
                    states / "analysis.json",
                    {"state": "running", "last_outer_call": call},
                )
            basis.publish(
                states / "analysis.json",
                {"state": "complete", "calls": calls, "signature": key},
            )
            if stage == "analyze":
                basis.publish(
                    summary_status, {"state": "analysis_complete", "signature": key}
                )
                return
            basis.publish(summary_status, {"state": "running", "stage": "report"})
            artifacts = []
            for name, rows in tables.items():
                path = folder / f"{name}.csv"
                tmp = path.with_suffix(".csv.tmp")
                with tmp.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=TABLES[name])
                    writer.writeheader()
                    writer.writerows(rows)
                tmp.replace(path)
                artifacts.append(path.name)
            artifacts.extend(
                figures(
                    folder, tables, first["inner_steps"], spec["actual_views"], calls
                )
            )
            basis.publish(
                folder / "summary.json",
                {
                    "signature": key,
                    "outer_calls": calls,
                    "inner_steps": first["inner_steps"],
                    "actual_views": spec["actual_views"],
                    "actual_states": spec["actual_states"],
                    "coordinates": first["coordinates"],
                    "anchor_caveat": CAVEAT,
                    "cross_call_scope": "Each aggregate contains one inner trajectory. Between-call inner angles and pooled multi-call spans require unrecorded cross-call inner products.",
                    "representations": {
                        "raw_saved": "The recorder converts each saved or transient endpoint and its anchor to FP32 before subtracting. It accumulates the resulting inner products in FP64.",
                        "bf16_evaluated": "The recorder casts each endpoint and its anchor to BF16, converts both to FP32 for subtraction, and accumulates inner products in FP64.",
                    },
                    "motion": "Interval updates are net changes over (start_step, end_step]. Each horizon differences successive saved points in that horizon. Only adjacent step numbers measure a single optimizer update.",
                    "normalization": "Uncentered PCA uses displacement magnitudes for norm_weighted and gives each nonzero vector unit length for equal_unit. actual_energy_captured always weights targets by squared norm; mean_actual_vector_capture weights each nonzero target equally.",
                    "rank": "At requested rank 1 or 2, both spans use the smaller available rank capped at the request. Full uses both numerical ranks independently. Overlap divides the sum of squared principal cosines by each side's rank.",
                    "all_states": "Projection-only rows in capture.csv fit the requested inner rank independently. No PCA across all archived reference states is fitted.",
                    "families": "family_comparisons.csv fits both spans anew within each parameter family. family_basis_energy.csv retains the globally fitted axes and partitions their squared norms across module families. Its share is the average family-restricted squared norm across the selected orthonormal axes, weighting each axis equally. At positive rank the shares sum to one.",
                    "rank_tolerance": "Eigenvalues below or equal to 1e-11 times the largest eigenvalue are discarded. Each local Gram is scaled by its largest diagonal before fitting.",
                    "zero_motion": "Zero denominators have missing fractions or cosines (blank CSV cells, gray heatmap cells).",
                    "figures": "Overlap and capture panels use cumulative displacements with norm-weighted PCA. The similarity panels show interval updates within each call.",
                    "tables": {k: len(v) for k, v in tables.items()},
                },
            )
            artifacts.append("summary.json")
            artifacts += [f"per-call/outer-{c:06d}.json" for c in calls]
            basis.publish(
                states / "report.json",
                {"state": "complete", "signature": key, "files": artifacts},
            )
            basis.publish(
                summary_status,
                {"state": "complete", "signature": key, "files": artifacts},
            )
        except Exception as exc:
            basis.publish(summary_status, {"state": "failed", "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("analyze", "report", "all"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.stage, basis.read(args.spec), args.output)


if __name__ == "__main__":
    main()
