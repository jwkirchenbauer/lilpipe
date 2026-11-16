"""Bounded embedding/head audit; numerical work requires an allocated GPU.

Only the two named matrix slices are read from weights and native basis files.
No language model is instantiated. BF16(anchor) is a candidate loading map whose
agreement with saved motion is measured, not assumed to prove a training cause.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PARAMETERS = {"embedding": "model.embed_tokens.weight", "head": "lm_head.weight"}
METHODS = ("pc1", "pc2", "uniform_plane", "variance_plane", "variance_full")


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def csv_rows(path, rows, fields):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_helpers(run):
    source = run / "source/v1/tar_basis_structure.py"
    spec = importlib.util.spec_from_file_location(
        "frozen_structure_token_audit", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_identity(path):
    path = Path(path)
    info = path.stat()
    return {
        "path": str(path),
        "resolved": str(path.resolve()),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
    }


def batch_frequencies(batch, vocabulary, roles=None):
    """All IDs include padding; targets follow the causal one-position shift."""
    result = {
        key: np.zeros(vocabulary, np.int64)
        for key in ("all_input", "attended_input", "supervised_target")
    }
    kept = []
    for index, ids in enumerate(batch["input_ids"]):
        if roles is not None and batch["metadata_ids"][index] not in roles:
            continue
        kept.append(index)
        ids = np.asarray(ids, dtype=np.int64)
        mask = np.asarray(batch["attention_mask"][index], dtype=bool)
        labels = np.asarray(batch["labels"][index], dtype=np.int64)[1:]
        if np.any(ids < 0) or np.any(ids >= vocabulary):
            raise ValueError("Input IDs outside matrix vocabulary")
        targets = labels[labels != -100]
        if np.any(targets < 0) or np.any(targets >= vocabulary):
            raise ValueError("Target IDs outside matrix vocabulary")
        result["all_input"] += np.bincount(ids, minlength=vocabulary)
        result["attended_input"] += np.bincount(ids[mask], minlength=vocabulary)
        result["supervised_target"] += np.bincount(targets, minlength=vocabulary)
    if not kept:
        raise ValueError("No selected outer/attacker rows")
    return result, kept


def motion_block(before, after):
    """FP64 differences of exact source values, with no in-place source edits."""
    import torch

    before, after = before.double(), after.double()
    loaded = before.to(torch.bfloat16).double()
    values = {
        "raw": after - before,
        "load_offset": loaded - before,
        "adjusted": after - loaded,
        "evaluated": after.to(torch.bfloat16).double() - loaded,
    }
    result = {}
    for label, value in values.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite checkpoint displacement")
        result[f"{label}_squared_norm"] = value.square().sum(1).cpu().numpy()
        result[f"{label}_changed_scalars"] = (value != 0).sum(1).cpu().numpy()
    residual = values["raw"] - values["load_offset"]
    result["raw_minus_offset_squared_norm"] = residual.square().sum(1).cpu().numpy()
    result["raw_minus_offset_max_abs"] = residual.abs().amax(1).cpu().numpy()
    result["raw_equals_offset"] = (residual == 0).all(1).cpu().numpy()
    return result


def matrix_slice(reader, name, start, stop):
    """Clone only authorized rows before any HIP transfer."""
    if name not in PARAMETERS.values():
        raise ValueError("Only input embedding and output head are authorized")
    filename = reader.mapping[name] if reader.mapping else "model.safetensors"
    return reader.handles[filename].get_slice(name)[start:stop].clone()


def matrix_info(reader, name):
    filename = reader.mapping[name] if reader.mapping else "model.safetensors"
    sliced = reader.handles[filename].get_slice(name)
    return {"shape": list(sliced.get_shape()), "saved_dtype": sliced.get_dtype()}


def concentration(energy):
    energy = np.asarray(energy, dtype=np.float64)
    if np.any(energy < 0) or not np.isfinite(energy).all():
        raise ValueError("Invalid energy array")
    order = np.argsort(-energy, kind="stable")
    total = float(energy.sum())
    return {
        "energy": total,
        "coordinates": len(energy),
        "nonzero": int(np.count_nonzero(energy)),
        "count_for_90_percent": (
            int(np.searchsorted(np.cumsum(energy[order]), 0.9 * total) + 1)
            if total
            else 0
        ),
        "top10_fraction": float(energy[order[:10]].sum() / total) if total else 0.0,
        "top100_fraction": float(energy[order[:100]].sum() / total) if total else 0.0,
        "top20_ids": order[:20].tolist(),
    }


def combine_energy(axis_energy, eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if (
        len(eigenvalues) < 2
        or np.any(eigenvalues[:2] <= 0)
        or not np.isfinite(eigenvalues).all()
    ):
        raise ValueError("Two positive native attack eigenvalues required")
    if np.any(eigenvalues < 0) or eigenvalues[0] < eigenvalues[1]:
        raise ValueError("Invalid native attack spectrum")
    weights = np.asarray(
        [
            [1, 0],
            [0, 1],
            [0.5, 0.5],
            eigenvalues[:2] / eigenvalues[:2].sum(),
            eigenvalues[:2] / eigenvalues.sum(),
        ]
    )
    return weights @ np.asarray(axis_energy, dtype=np.float64), weights


def loading_check(
    module, snapshot, observations, defense, output, arrays, row_records, chunk_rows
):
    import torch

    with Path(observations).open() as stream:
        record = json.loads(next(stream))
    if record["step"] != 1:
        raise ValueError("First recorded batch must be update 1")
    signature = hashlib.sha256(
        json.dumps(record["batch"], sort_keys=True).encode()
    ).hexdigest()
    if signature != record["batch_sha256"]:
        raise ValueError("First recorded batch hash mismatch")
    result = {
        "snapshot": snapshot,
        "observation_file": str(observations),
        "batch_sha256": signature,
        "roles": [0, 1] if defense else "all attacker rows",
        "matrices": {},
    }
    with contextlib.ExitStack() as stack:
        before = module.WeightReader(snapshot["anchor"], stack)
        after = module.WeightReader(snapshot["path"], stack)
        for label, name in PARAMETERS.items():
            first, second = matrix_info(before, name), matrix_info(after, name)
            if first["shape"] != second["shape"] or len(first["shape"]) != 2:
                raise ValueError("Embedding/head source shapes differ")
            rows, width = first["shape"]
            frequencies, selected = batch_frequencies(
                record["batch"], rows, {0, 1} if defense else None
            )
            pieces = {}
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                a = matrix_slice(before, name, start, stop).to(
                    "cuda:0", dtype=torch.float64
                )
                b = matrix_slice(after, name, start, stop).to(
                    "cuda:0", dtype=torch.float64
                )
                for field, values in motion_block(a, b).items():
                    pieces.setdefault(field, []).append(values)
            data = {field: np.concatenate(values) for field, values in pieces.items()}
            unseen = frequencies["all_input"] == 0
            summary = {
                "before": first,
                "after": second,
                "selected_batch_rows": selected,
                "unique_all_input_ids": int((~unseen).sum()),
                "scopes": {},
            }
            for scope, mask in (
                ("all", np.ones(rows, bool)),
                ("seen_all_input", ~unseen),
                ("unseen_all_input", unseen),
            ):
                stats = {"rows": int(mask.sum()), "scalars": int(mask.sum()) * width}
                for field in ("raw", "load_offset", "adjusted", "evaluated"):
                    stats[f"{field}_l2"] = float(
                        np.sqrt(data[f"{field}_squared_norm"][mask].sum())
                    )
                    stats[f"{field}_changed_rows"] = int(
                        np.count_nonzero(data[f"{field}_changed_scalars"][mask])
                    )
                    stats[f"{field}_changed_scalars"] = int(
                        data[f"{field}_changed_scalars"][mask].sum()
                    )
                    stats[f"{field}_changed_fraction"] = (
                        stats[f"{field}_changed_scalars"] / stats["scalars"]
                        if stats["scalars"]
                        else 0.0
                    )
                stats["raw_minus_offset_residual_l2"] = float(
                    np.sqrt(data["raw_minus_offset_squared_norm"][mask].sum())
                )
                stats["raw_equals_offset_exactly"] = bool(
                    data["raw_equals_offset"][mask].all()
                )
                stats["maximum_raw_minus_offset_residual"] = float(
                    data["raw_minus_offset_max_abs"][mask].max(initial=0)
                )
                summary["scopes"][scope] = stats
            result["matrices"][label] = summary
            prefix = f"{output}__{label}"
            arrays.update(
                {
                    f"{prefix}__{field}": values
                    for field, values in {**data, **frequencies}.items()
                }
            )
            for row in range(rows):
                row_records.append(
                    {
                        "check": output,
                        "matrix": label,
                        "row_id": row,
                        "unseen_all_input": bool(unseen[row]),
                        **{
                            field: int(values[row])
                            for field, values in frequencies.items()
                        },
                        **{field: value[row].item() for field, value in data.items()},
                    }
                )
            print(
                f"Loading check {output}/{label} finished; unseen adjusted L2={summary['scopes']['unseen_all_input']['adjusted_l2']:.8g}",
                flush=True,
            )
    return result


def basis_check(
    module, job, name, arrays, row_records, column_records, token_records, chunk_rows
):
    import torch
    from transformers import AutoTokenizer

    geometry = Path(job["layout"])
    layout = read(geometry)
    path = geometry.parent / "attack-basis.npy"
    basis = np.load(path, mmap_mode="r", allow_pickle=False)
    if basis.shape != (2, layout["parameters"]) or basis.dtype != np.float32:
        raise ValueError("Native two-axis FP32 basis layout mismatch")
    eigenvalues = layout["attack_eigenvalues"]
    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"], local_files_only=True)
    result = {
        "basis": str(path),
        "geometry": str(geometry),
        "attack_eigenvalues": eigenvalues,
        "matrices": {},
    }
    for label, parameter in PARAMETERS.items():
        rows, width = layout["shapes"][parameter]
        offset = layout["offsets"][parameter]
        row_energy = np.zeros((2, rows), np.float64)
        column_energy = np.zeros((2, width), np.float64)
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            owned = np.array(
                basis[:, offset + start * width : offset + stop * width], copy=True
            ).reshape(2, stop - start, width)
            value = torch.from_numpy(owned).to("cuda:0", dtype=torch.float64)
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite basis slice")
            squared = value.square()
            row_energy[:, start:stop] = squared.sum(2).cpu().numpy()
            column_energy += squared.sum(1).cpu().numpy()
        np.testing.assert_allclose(
            row_energy.sum(1), column_energy.sum(1), atol=1e-12, rtol=1e-12
        )
        row_values, weights = combine_energy(row_energy, eigenvalues)
        column_values, _ = combine_energy(column_energy, eigenvalues)
        frequencies, batches = module.token_frequencies(job["observations"], rows)
        summaries = {}
        for index, method in enumerate(METHODS):
            summaries[method] = {
                "axis_weights": weights[index].tolist(),
                "rows": concentration(row_values[index]),
                "columns": concentration(column_values[index]),
            }
            total = float(row_values[index].sum())
            summaries[method]["rows"]["fraction_in_unseen_all_input"] = (
                float(row_values[index, frequencies["all_input"] == 0].sum() / total)
                if total
                else 0.0
            )
        prefix = f"{name}__{label}"
        arrays[f"{prefix}__row_energy"] = row_values
        arrays[f"{prefix}__column_energy"] = column_values
        arrays.update(
            {f"{prefix}__{key}": values for key, values in frequencies.items()}
        )
        for coordinate, count, values, records in (
            ("row_id", rows, row_values, row_records),
            ("column_id", width, column_values, column_records),
        ):
            for coordinate_id in range(count):
                records.append(
                    {
                        "case": name,
                        "matrix": label,
                        coordinate: coordinate_id,
                        **{
                            f"{method}_energy": float(values[index, coordinate_id])
                            for index, method in enumerate(METHODS)
                        },
                        **(
                            {
                                field: int(data[coordinate_id])
                                for field, data in frequencies.items()
                            }
                            if coordinate == "row_id"
                            else {}
                        ),
                    }
                )
        # Decode only this selection of 20 rows.
        for rank, row in enumerate(summaries["uniform_plane"]["rows"]["top20_ids"], 1):
            token_records.append(
                {
                    "case": name,
                    "matrix": label,
                    "uniform_rank": rank,
                    "row_id": row,
                    "token": tokenizer.decode(
                        [row], clean_up_tokenization_spaces=False
                    ),
                    **{
                        f"{method}_energy": float(row_values[index, row])
                        for index, method in enumerate(METHODS)
                    },
                    **{
                        field: int(values[row]) for field, values in frequencies.items()
                    },
                }
            )
        result["matrices"][label] = {
            "shape": [rows, width],
            "observed_batches": batches,
            "methods": summaries,
        }
        print(f"Native basis {name}/{label} finished", flush=True)
    return result


def inputs(run, jobs):
    paths = {Path(__file__), run / "source/v1/tar_basis_structure.py"}
    for case, job in jobs.items():
        paths.update(
            (
                run / "cases" / case / "job.json",
                Path(job["layout"]),
                Path(job["layout"]).parent / "attack-basis.npy",
                Path(job["observations"]),
            )
        )
        if job.get("defense_observations"):
            paths.add(Path(job["defense_observations"]))
        for snapshot in job["snapshots"]:
            if snapshot["name"] not in ("attack-1", "defense-1"):
                continue
            for folder in (snapshot["anchor"], snapshot["path"]):
                paths.update(Path(folder).glob("*.safetensors"))
                paths.update(Path(folder).glob("*.safetensors.index.json"))
        tokenizer = Path(job["tokenizer"])
        paths.update(tokenizer.glob("*.json"))
        paths.update(tokenizer.glob("*.model"))
    protocol = {
        "version": 1,
        "jobs": jobs,
        "files": [file_identity(path) for path in sorted(paths)],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(
            (run / "source/v1/tar_basis_structure.py").read_bytes()
        ).hexdigest(),
    }
    return {
        "signature": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "protocol": protocol,
    }


def values(run, chunk_rows):
    import torch

    if not os.environ.get("FLUX_JOB_ID") or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Numerical audit requires an allocated single-visible-GPU job"
        )
    module = load_helpers(run)
    jobs = {
        name: read(run / "cases" / name / "job.json")
        for name in ("local-defense-attack", "collaborator-attack")
    }
    identity = inputs(run, jobs)
    output = run / "token-audit"
    output.mkdir(exist_ok=True)
    if (output / "input-identity.json").exists() and read(
        output / "input-identity.json"
    ) != identity:
        raise ValueError(
            "Token audit input/source identity changed; use a new audit version/output"
        )
    publish(output / "input-identity.json", identity)
    arrays, loading_rows, basis_rows, column_rows, top_tokens = {}, [], [], [], []
    result = {
        "state": "running",
        "input_signature": identity["signature"],
        "source_sha256": identity["protocol"]["source_sha256"],
        "helper_sha256": identity["protocol"]["helper_sha256"],
        "chunk_rows": chunk_rows,
        "gpu": torch.cuda.get_device_name(0),
        "flux_job_id": os.environ["FLUX_JOB_ID"],
        "loading": {},
        "basis": {},
        "methods": list(METHODS),
    }
    publish(output / "results.json", result)
    for case, job in jobs.items():
        selected = [
            snapshot
            for snapshot in job["snapshots"]
            if snapshot["name"] in ("attack-1", "defense-1")
        ]
        for snapshot in selected:
            defense = snapshot["name"] == "defense-1"
            observations = (
                job["defense_observations"] if defense else job["observations"]
            )
            label = f"{case}-{snapshot['name']}"
            result["loading"][label] = loading_check(
                module,
                snapshot,
                observations,
                defense,
                label,
                arrays,
                loading_rows,
                chunk_rows,
            )
        result["basis"][case] = basis_check(
            module, job, case, arrays, basis_rows, column_rows, top_tokens, chunk_rows
        )
    for filename, rows in (
        ("loading-offset-rows.csv", loading_rows),
        ("basis-rows.csv", basis_rows),
        ("basis-columns.csv", column_rows),
        ("top-tokens.csv", top_tokens),
    ):
        csv_rows(output / filename, rows, list(rows[0]))
    temporary = output / "arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(output / "arrays.npz")
    result["definitions"] = {
        "raw": "saved_after - raw_source_anchor, including any load conversion",
        "load_offset": "BF16(raw_source_anchor) - raw_source_anchor, candidate loading map",
        "adjusted": "saved_after - BF16(raw_source_anchor); not a direct capture of training memory",
        "evaluated": "BF16(saved_after) - BF16(raw_source_anchor)",
        "unseen": "Absent from every input_id in the first actual batch, including padding IDs; defense includes only outer roles 0/1 and excludes stopped inner role 2",
        "basis": "Saved native raw-displacement PC1/PC2. Uniform plane uses equal 0.5 weights; variance_plane weights by the first two eigenvalues normalized by their sum; variance_full divides by the full native spectrum sum. The latter describes only the contribution of the first two axes.",
        "eigenvalues": "Native eigenvalues sum squared displacements over selected snapshots without centering. Variance weighting here denotes this uncentered trajectory-energy weighting, not centered statistical variance.",
        "concentration": "90% counts and top 10/100 fractions normalize within the named tensor and method; hidden columns are parameter coordinates",
        "decoded_tokens": "Only the top 20 rows per tensor/case by uniform-plane energy are decoded. Frequencies describe all recorded attack batches; target frequencies use shifted nonmasked labels.",
        "interpretation": "Algebraic agreement with a candidate BF16 loading map is measured, not assumed causal. Input embeddings and untied output head have different gradient support; output softmax can update classes absent from input IDs. No semantic biology-token inference is made.",
    }
    result["state"] = "complete"
    publish(output / "results.json", result)
    print(f"Complete numeric token audit: {output}", flush=True)


def report(run):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = run / "token-audit"
    result = read(output / "results.json")
    if result["state"] != "complete":
        raise ValueError("Complete numerical token audit required before rendering")
    cases = list(result["basis"])
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True
    )
    with np.load(output / "arrays.npz", allow_pickle=False) as arrays:
        for row, case in enumerate(cases):
            for col, label in enumerate(PARAMETERS):
                axis = axes[row, col]
                energy = arrays[f"{case}__{label}__column_energy"]
                for index, method in enumerate(METHODS[:4]):
                    curve = np.cumsum(np.sort(energy[index])[::-1])
                    if curve[-1]:
                        axis.plot(
                            np.arange(1, len(curve) + 1),
                            curve / curve[-1],
                            label=method.replace("_", " "),
                        )
                axis.axhline(0.9, color="gray", linestyle=":", linewidth=1)
                axis.set(
                    xscale="log",
                    ylim=(0, 1.02),
                    title=f"{case.replace('-defense-attack', '').replace('-attack', '')}: {label}",
                    xlabel="Highest-energy hidden columns (count)",
                    ylabel="Fraction of this matrix's energy",
                )
                axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Native attack basis: hidden-column concentration\nColumns ranked separately per method; common axes"
    )
    figure.savefig(output / "column-concentration.png", dpi=170)
    figure.savefig(output / "column-concentration.pdf")
    plt.close(figure)
    lines = [
        "# Embedding/head loading and native-basis token audit",
        "",
        "Only embedding and output-head matrix slices were read. No model was instantiated.",
        "",
        "Candidate loading map: BF16(source anchor). Exact equality is an algebraic measurement, not proof of a training cause. 'Unseen' includes all padded input IDs; defense uses only outer roles 0/1.",
        "",
        "| First saved update / matrix | Source → saved dtype | Unseen rows | Raw L2 | Load-offset L2 | Adjusted L2 | Max residual | Exact raw=offset |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, check in result["loading"].items():
        for label, matrix in check["matrices"].items():
            stats = matrix["scopes"]["unseen_all_input"]
            lines.append(
                f"| {name}/{label} | {matrix['before']['saved_dtype']} → {matrix['after']['saved_dtype']} | {stats['rows']} | {stats['raw_l2']:.8g} | {stats['load_offset_l2']:.8g} | {stats['adjusted_l2']:.8g} | {stats['maximum_raw_minus_offset_residual']:.8g} | {stats['raw_equals_offset_exactly']} |"
            )
    lines += [
        "",
        "Counts below cover 90% of the named matrix's energy. Uniform-plane energy averages unit PC axes; variance-plane energy weights them by native uncentered displacement eigenvalues. The contribution of the first two axes normalized by the full spectrum is also preserved in JSON/CSV/NPZ. This does not estimate centered statistical variance.",
        "",
        "| Case / matrix / method | Matrix energy | Rows for 90% | Columns for 90% | Top 10 column fraction | Top 100 column fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, value in result["basis"].items():
        for label, matrix in value["matrices"].items():
            for method, stats in matrix["methods"].items():
                lines.append(
                    f"| {case}/{label}/{method} | {stats['rows']['energy']:.8g} | {stats['rows']['count_for_90_percent']} | {stats['columns']['count_for_90_percent']} | {stats['columns']['top10_fraction']:.4f} | {stats['columns']['top100_fraction']:.4f} |"
                )
    lines += [
        "",
        "[Column concentration](column-concentration.png) · [All numeric results](results.json) · [Loading rows](loading-offset-rows.csv) · [Basis rows](basis-rows.csv) · [Basis columns](basis-columns.csv) · [Top20 decoded rows per tensor/case](top-tokens.csv)",
        "",
        result["definitions"]["interpretation"],
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def self_test():
    import torch
    from safetensors.torch import save_file

    anchor = torch.tensor(
        [[0.01001, 0.1234], [0.5678, -0.3333], [0.00123, -0.07001]], dtype=torch.float32
    )
    frozen = anchor.clone()
    target = anchor.bfloat16().float()
    target[0, 0] += 0.001
    stats = motion_block(anchor, target)
    assert stats["raw_equals_offset"][1:].all()
    assert not stats["raw_equals_offset"][0]
    np.testing.assert_array_equal(stats["adjusted_squared_norm"][1:], 0)
    torch.testing.assert_close(anchor, frozen, rtol=0, atol=0)
    half = anchor.half()
    assert motion_block(half, half.bfloat16().float())["raw_equals_offset"].all()
    batch = {
        "input_ids": [[1, 3, 9], [2, 4, 9], [7, 5, 9]],
        "attention_mask": [[1, 1, 0]] * 3,
        "labels": [[1, 3, -100], [2, 4, -100], [7, 5, -100]],
        "metadata_ids": [0, 1, 2],
    }
    counts, selected = batch_frequencies(batch, 10, {0, 1})
    assert selected == [0, 1] and counts["all_input"][9] == 2
    assert counts["all_input"][7] == counts["attended_input"][9] == 0
    assert counts["supervised_target"][3] == counts["supervised_target"][4] == 1
    assert counts["supervised_target"][1] == counts["supervised_target"][5] == 0
    data = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    combined, weights = combine_energy(data, [9, 1])
    np.testing.assert_allclose(combined[2], [0.5, 0.5, 0])
    np.testing.assert_allclose(combined[3], [0.9, 0.1, 0])
    assert concentration(combined[2])["count_for_90_percent"] == 2
    assert concentration(combined[3])["count_for_90_percent"] == 1
    assert concentration(np.zeros(3))["count_for_90_percent"] == 0
    try:
        matrix_slice(None, "model.layers.0.mlp.up_proj.weight", 0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Unauthorized parameter read was accepted")
    module = load_helpers(Path(__file__).resolve().parents[2])
    with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
        save_file(
            {PARAMETERS["embedding"]: anchor},
            str(Path(directory) / "model.safetensors"),
        )
        reader = module.WeightReader(directory, stack)
        assert matrix_info(reader, PARAMETERS["embedding"]) == {
            "shape": [3, 2],
            "saved_dtype": "F32",
        }
        owned = matrix_slice(reader, PARAMETERS["embedding"], 1, 2)
        torch.testing.assert_close(owned, frozen[1:2], rtol=0, atol=0)
        owned.add_(1)
        torch.testing.assert_close(
            matrix_slice(reader, PARAMETERS["embedding"], 1, 2),
            frozen[1:2],
            rtol=0,
            atol=0,
        )
    print(
        "PASS: mixed source dtype loading, exact unseen-row residual, immutable anchors, outer-only all-ID support, variance weights, concentration, authorized reads"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.run is None or not 1 <= args.chunk_rows <= 512:
        parser.error("run and chunk-rows between 1 and 512 required")
    elif args.report_only:
        report(args.run.resolve())
    else:
        values(args.run.resolve(), args.chunk_rows)


if __name__ == "__main__":
    main()
