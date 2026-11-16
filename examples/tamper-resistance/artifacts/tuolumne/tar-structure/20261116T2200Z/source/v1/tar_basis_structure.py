"""Stream checkpoint trajectories to measure coordinate and module concentration.

All large scans run through launch_tuo.py. Trajectory PCA is uncentered, matching
tar_dynamics: snapshots are displacements from each phase's starting model.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path, **arrays):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def classify_parameter(name):
    if name == "model.embed_tokens.weight":
        return "embedding", -1, "input_embedding"
    if name == "lm_head.weight":
        return "output_head", -1, "output_head"
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)\.weight", name)
    if match:
        layer, component = int(match[1]), match[2]
        if component.startswith("self_attn."):
            return "attention", layer, component.split(".")[-1]
        if component.startswith("mlp."):
            return "mlp", layer, component.split(".")[-1]
        if "norm" in component:
            return "norm", layer, component
    if "norm" in name:
        return "norm", -1, name
    return "other", -1, name


def fit_subspace(gram, indices, rank=None):
    """Return orthonormal coefficients and the uncentered snapshot spectrum."""
    gram = np.asarray(gram, dtype=np.float64)
    indices = np.asarray(indices, dtype=int)
    small = gram[np.ix_(indices, indices)]
    values, vectors = np.linalg.eigh((small + small.T) / 2)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = max(float(values[0]) * 1e-11, 1e-14)
    if values.min() < -tolerance * 10:
        raise ValueError("Trajectory Gram matrix is not positive semidefinite")
    valid = values > tolerance
    actual_rank = int(valid.sum())
    if not actual_rank:
        raise ValueError("Zero trajectory has no preferential directions")
    count = actual_rank if rank is None else min(rank, actual_rank)
    coefficients = np.zeros((len(gram), count), dtype=np.float64)
    coefficients[indices] = vectors[:, :count] / np.sqrt(values[:count])
    np.testing.assert_allclose(
        coefficients.T @ gram @ coefficients, np.eye(count), atol=1e-6
    )
    return coefficients, {
        "rank": actual_rank,
        "used_rank": count,
        "eigenvalues": np.maximum(values, 0).tolist(),
        "captured_snapshot_energy": float(values[:count].sum() / values[valid].sum()),
        "snapshot_count": len(indices),
    }


def overlap(gram, left, right):
    """Principal angles after whitening each restricted subspace, including blocks."""
    gram = np.asarray(gram, dtype=np.float64)

    def whiten(coefficients):
        small = coefficients.T @ gram @ coefficients
        values, vectors = np.linalg.eigh((small + small.T) / 2)
        cutoff = max(float(values.max()) * 1e-10, 1e-24)
        valid = values > cutoff
        return coefficients @ (vectors[:, valid] / np.sqrt(values[valid]))

    first, second = whiten(left), whiten(right)
    if not first.shape[1] or not second.shape[1]:
        return {
            "left_rank": first.shape[1],
            "right_rank": second.shape[1],
            "angles_degrees": [],
            "mean_cosine_squared": None,
        }
    singular = np.linalg.svd(first.T @ gram @ second, compute_uv=False)
    if singular.max() > 1 + 1e-5:
        raise ValueError("Subspace overlap exceeds one")
    singular = np.clip(singular, 0, 1)
    return {
        "left_rank": first.shape[1],
        "right_rank": second.shape[1],
        "angles_degrees": np.degrees(np.arccos(singular)).tolist(),
        "mean_cosine_squared": float(np.mean(singular**2)),
    }


def concentration(counts, masses, edges, total_count, squared_energy_sum):
    """Bound support needed for target energy using exact bin totals and edges."""
    counts = np.asarray(counts, dtype=np.int64)
    masses = np.asarray(masses, dtype=np.float64)
    total = float(masses.sum())
    positive = int(counts.sum())
    result = {
        "energy": total,
        "coordinates": int(total_count),
        "nonzero_coordinates": positive,
        "nonzero_fraction": positive / total_count,
        "effective_support_fraction": (
            total * total / squared_energy_sum / total_count
            if squared_energy_sum
            else 0
        ),
        "energy_targets": {},
        "top_fractions": {},
    }
    reverse_mass = np.cumsum(masses[::-1])
    reverse_count = np.cumsum(counts[::-1])
    for target in (0.5, 0.9, 0.99, 0.999):
        if not total:
            lower = upper = 0
        else:
            index = min(
                int(np.searchsorted(reverse_mass, target * total)), len(counts) - 1
            )
            bucket = len(counts) - 1 - index
            before_mass = reverse_mass[index - 1] if index else 0.0
            before_count = int(reverse_count[index - 1]) if index else 0
            remainder = max(0.0, target * total - before_mass)
            lower = before_count + min(
                int(counts[bucket]), math.ceil(remainder / edges[bucket + 1])
            )
            upper = before_count + min(
                int(counts[bucket]), math.ceil(remainder / edges[bucket])
            )
        result["energy_targets"][str(target)] = {
            "coordinate_fraction_lower": lower / total_count,
            "coordinate_fraction_upper": upper / total_count,
        }
    for fraction in (1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0):
        take = min(int(math.ceil(fraction * total_count)), positive)
        if not total or not take:
            low = high = 0.0
        elif take == positive:
            low = high = 1.0
        else:
            index = int(np.searchsorted(reverse_count, take))
            bucket = len(counts) - 1 - index
            before_count = int(reverse_count[index - 1]) if index else 0
            before_mass = reverse_mass[index - 1] if index else 0.0
            partial, size, mass = (
                take - before_count,
                int(counts[bucket]),
                masses[bucket],
            )
            low_mass = max(
                partial * edges[bucket], mass - (size - partial) * edges[bucket + 1]
            )
            high_mass = min(
                partial * edges[bucket + 1], mass - (size - partial) * edges[bucket]
            )
            low, high = (before_mass + low_mass) / total, (
                before_mass + high_mass
            ) / total
        result["top_fractions"][str(fraction)] = {
            "energy_fraction_lower": float(np.clip(low, 0, 1)),
            "energy_fraction_upper": float(np.clip(high, 0, 1)),
        }
    return result


class WeightReader:
    def __init__(self, folder, stack):
        from safetensors import safe_open

        self.folder = Path(folder)
        index = self.folder / "model.safetensors.index.json"
        if index.exists():
            self.mapping = read(index)["weight_map"]
        else:
            self.mapping = None
        files = (
            sorted(set(self.mapping.values()))
            if self.mapping
            else ["model.safetensors"]
        )
        self.handles = {
            name: stack.enter_context(
                safe_open(self.folder / name, framework="pt", device="cpu")
            )
            for name in files
        }

    def tensor(self, name):
        filename = self.mapping[name] if self.mapping else "model.safetensors"
        return self.handles[filename].get_tensor(name)


def identity(job):
    paths = [Path(job["layout"])]
    for snapshot in job["snapshots"]:
        for folder in (snapshot["anchor"], snapshot["path"]):
            paths.extend(sorted(Path(folder).glob("*.safetensors")))
            paths.extend(sorted(Path(folder).glob("*.safetensors.index.json")))
    paths += [Path(item["path"]) for item in job.get("external_bases", [])]
    paths += [
        Path(item["native_geometry"])
        for item in job.get("groups", [])
        if item.get("native_geometry")
    ]
    paths += [Path(job["observations"])]
    if job.get("defense_observations"):
        paths.append(Path(job["defense_observations"]))
    files = []
    for path in sorted(set(paths)):
        info = path.stat()
        files.append(
            {
                "path": str(path),
                "resolved": str(path.resolve()),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        )
    declaration = {
        "job": job,
        "files": files,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    signature = hashlib.sha256(
        json.dumps(declaration, sort_keys=True).encode()
    ).hexdigest()
    return signature, declaration


@contextlib.contextmanager
def blocks(job, layout, start_parameter=0, include_external=True):
    """Yield raw and BF16-evaluated displacements with bounded device memory."""
    import torch

    with contextlib.ExitStack() as stack:
        folders = {item[key] for item in job["snapshots"] for key in ("anchor", "path")}
        readers = {folder: WeightReader(folder, stack) for folder in sorted(folders)}
        external = (
            [
                np.load(item["path"], mmap_mode="r", allow_pickle=False)
                for item in job.get("external_bases", [])
            ]
            if include_external
            else []
        )
        if include_external:
            for item in job.get("external_bases", []):
                metadata = read(Path(item["path"]).parent / "geometry.json")
                for key in ("names", "shapes", "offsets", "parameters"):
                    if metadata[key] != layout[key]:
                        raise ValueError("External basis parameter ordering differs")
        for array in external:
            if array.shape != (2, layout["parameters"]) or array.dtype != np.float32:
                raise ValueError("External basis layout differs")

        def iterator():
            for parameter_index, name in enumerate(layout["names"]):
                if parameter_index < start_parameter:
                    continue
                count = math.prod(layout["shapes"][name])
                tensors = {
                    folder: reader.tensor(name).reshape(-1)
                    for folder, reader in readers.items()
                }
                if any(tensor.numel() != count for tensor in tensors.values()):
                    raise ValueError("Parameter shape differs")
                dtypes = {
                    folder: str(tensor.dtype) for folder, tensor in tensors.items()
                }
                for start in range(0, count, job.get("chunk_elements", 1048576)):
                    stop = min(start + job.get("chunk_elements", 1048576), count)
                    values = {
                        folder: tensor[start:stop].to(
                            device="cuda:0", dtype=torch.float64
                        )
                        for folder, tensor in tensors.items()
                    }
                    raw = torch.stack(
                        [
                            values[item["path"]] - values[item["anchor"]]
                            for item in job["snapshots"]
                        ]
                    )
                    cast = {
                        folder: value.to(torch.bfloat16).double()
                        for folder, value in values.items()
                    }
                    evaluated = torch.stack(
                        [
                            cast[item["path"]] - cast[item["anchor"]]
                            for item in job["snapshots"]
                        ]
                    )
                    if external:
                        offset = layout["offsets"][name] + start
                        basis = [
                            torch.from_numpy(
                                np.array(
                                    array[:, offset : offset + stop - start], copy=True
                                )
                            ).to(device="cuda:0", dtype=torch.float64)
                            for array in external
                        ]
                        raw = torch.cat([raw, *basis])
                    yield parameter_index, name, start, stop, count, raw, evaluated, dtypes

        yield iterator()


def scan(folder, job):
    import torch

    layout = read(job["layout"])
    signature, declaration = identity(job)
    progress = folder / "cache/scan"
    progress.mkdir(parents=True, exist_ok=True)
    registry = folder / "input-identity.json"
    if registry.exists() and read(registry)["signature"] != signature:
        raise ValueError("Analysis inputs changed; use a fresh output directory")
    publish(registry, {"signature": signature, "declaration": declaration})
    completed = 0
    while (progress / f"{completed:03}.npz").exists() and (
        progress / f"{completed:03}.json"
    ).exists():
        with np.load(progress / f"{completed:03}.npz") as cached:
            if str(cached["signature"]) != signature:
                raise ValueError("Invalid scan cache identity")
        completed += 1
    print(
        f"Reusing {completed}/{len(layout['names'])} completed parameter scans",
        flush=True,
    )
    raw_sum = eval_sum = None
    current = None
    with blocks(job, layout, completed) as iterator:
        for index, name, start, stop, count, raw, evaluated, dtypes in iterator:
            if name != current:
                current = name
                raw_sum = torch.zeros(
                    (len(raw), len(raw)), dtype=torch.float64, device="cuda:0"
                )
                eval_sum = torch.zeros(
                    (len(evaluated), len(evaluated)),
                    dtype=torch.float64,
                    device="cuda:0",
                )
            raw_sum.add_(raw @ raw.T)
            eval_sum.add_(evaluated @ evaluated.T)
            if stop == count:
                save_npz(
                    progress / f"{index:03}.npz",
                    raw=raw_sum.cpu().numpy(),
                    evaluated=eval_sum.cpu().numpy(),
                    signature=np.array(signature),
                )
                publish(
                    progress / f"{index:03}.json",
                    {"name": name, "shape": layout["shapes"][name], "dtypes": dtypes},
                )
                publish(
                    folder / "status/scan.json",
                    {"state": "running", "completed_parameters": index + 1},
                )
                print("Scanned", index + 1, name, flush=True)
    gram_raw = gram_eval = None
    for index in range(len(layout["names"])):
        with np.load(progress / f"{index:03}.npz") as data:
            gram_raw = (
                data["raw"].copy() if gram_raw is None else gram_raw + data["raw"]
            )
            gram_eval = (
                data["evaluated"].copy()
                if gram_eval is None
                else gram_eval + data["evaluated"]
            )
    save_npz(folder / "gram.npz", raw=gram_raw, evaluated=gram_eval)
    for index in job.get("require_zero_snapshots", []):
        if gram_raw[index, index] != 0 or gram_eval[index, index] != 0:
            raise ValueError("Defense and attack anchor checkpoints differ")
    fitted, spectra = {}, {}
    for precision, gram in (("raw", gram_raw), ("evaluated", gram_eval)):
        for group in job.get("groups", []):
            indices = group["indices"]
            for rank, suffix in ((2, "plane"), (None, "span")):
                coeff, info = fit_subspace(gram, indices, rank)
                key = f"{group['name']}-{precision}-{suffix}"
                fitted[key] = coeff.tolist()
                spectra[key] = info
    n = len(job["snapshots"])
    for index, item in enumerate(job.get("external_bases", [])):
        coeff = np.zeros((len(gram_raw), 2))
        coeff[n + 2 * index : n + 2 * index + 2] = np.eye(2)
        np.testing.assert_allclose(coeff.T @ gram_raw @ coeff, np.eye(2), atol=1e-5)
        fitted[f"{item['name']}-raw-plane"] = coeff.tolist()
    validations = []
    for group in job.get("groups", []):
        if group.get("native_geometry"):
            native = read(group["native_geometry"])
            indices = group["indices"]
            np.testing.assert_allclose(
                gram_raw[np.ix_(indices, indices)],
                np.asarray(native["gram"])[: len(indices), : len(indices)],
                rtol=1e-7,
                atol=1e-10,
            )
            validations.append({"group": group["name"], "native_gram_matches": True})
    overlaps, captures = [], []
    for precision, gram in (("raw", gram_raw), ("evaluated", gram_eval)):
        keys = [key for key in fitted if f"-{precision}-" in key]
        for i, left in enumerate(keys):
            for right in keys[i + 1 :]:
                if (
                    left.rsplit(f"-{precision}-", 1)[0]
                    == right.rsplit(f"-{precision}-", 1)[0]
                ):
                    continue
                overlaps.append(
                    {
                        "left": left,
                        "right": right,
                        "precision": precision,
                        **overlap(
                            gram, np.asarray(fitted[left]), np.asarray(fitted[right])
                        ),
                    }
                )
        for key in keys:
            coeff = np.asarray(fitted[key])
            cross = gram[:n] @ coeff
            inverse = np.linalg.pinv(coeff.T @ gram @ coeff)
            for index, item in enumerate(job["snapshots"]):
                if item.get("include_capture") is False:
                    continue
                norm = gram[index, index]
                power = float(cross[index] @ inverse @ cross[index])
                captures.append(
                    {
                        "basis": key,
                        "snapshot": item["name"],
                        "precision": precision,
                        "squared_displacement": float(norm),
                        "captured_fraction": (
                            float(np.clip(power / norm, 0, 1)) if norm else 1.0
                        ),
                    }
                )
    reversals = []
    for precision, gram in (("raw", gram_raw), ("evaluated", gram_eval)):
        for pair in job.get("phase_pairs", []):
            defense, attack = pair["defense_endpoint"], pair["attack_endpoint"]
            squared_defense, squared_attack = (
                gram[defense, defense],
                gram[attack, attack],
            )
            dot = gram[defense, attack]
            reversals.append(
                {
                    "pair": pair["name"],
                    "precision": precision,
                    "endpoint_cosine": float(
                        dot / math.sqrt(squared_defense * squared_attack)
                    ),
                    "fraction_of_defense_endpoint_reversed_along_its_direction": float(
                        -dot / squared_defense
                    ),
                    "distance_from_base_after_attack_over_defense_distance": float(
                        math.sqrt(
                            max(0, squared_defense + squared_attack + 2 * dot)
                            / squared_defense
                        )
                    ),
                }
            )
    publish(
        folder / "subspaces.json",
        {
            "state": "complete",
            "spectra": spectra,
            "coefficients": fitted,
            "overlaps": overlaps,
            "native_validations": validations,
            "endpoint_reversal": reversals,
            "anchor_continuity_verified": bool(job.get("require_zero_snapshots")),
        },
    )
    write_csv(folder / "snapshot-capture.csv", captures)
    publish(
        folder / "status/scan.json",
        {"state": "complete", "parameters": len(layout["names"])},
    )


def build_views(job, subspaces, raw_size, eval_size):
    views = []
    for key, coefficients in subspaces["coefficients"].items():
        if key.endswith("-plane") and not any(
            key.startswith(name + "-") for name in job.get("overlap_only", [])
        ):
            views.append(
                {
                    "name": key,
                    "precision": "evaluated" if "-evaluated-" in key else "raw",
                    "coefficients": np.asarray(coefficients),
                }
            )
    for index, item in enumerate(job["snapshots"]):
        if item.get("inspect"):
            for precision, size in (("raw", raw_size), ("evaluated", eval_size)):
                coefficients = np.zeros((size, 1))
                coefficients[index, 0] = 1
                views.append(
                    {
                        "name": item["name"] + "-" + precision + "-displacement",
                        "precision": precision,
                        "coefficients": coefficients,
                    }
                )
    for item in job.get("intervals", []):
        for precision, size in (("raw", raw_size), ("evaluated", eval_size)):
            coefficients = np.zeros((size, 1))
            coefficients[item["end"], 0] = 1
            coefficients[item["start"], 0] = -1
            views.append(
                {
                    "name": item["name"] + "-" + precision + "-displacement",
                    "precision": precision,
                    "coefficients": coefficients,
                }
            )
    return views


def structure(folder, job):
    import torch

    layout = read(job["layout"])
    signature, _ = identity(job)
    if read(folder / "input-identity.json")["signature"] != signature:
        raise ValueError("Scan inputs changed")
    subspaces = read(folder / "subspaces.json")
    with np.load(folder / "gram.npz") as data:
        grams = {key: data[key] for key in ("raw", "evaluated")}
    views = build_views(job, subspaces, len(grams["raw"]), len(grams["evaluated"]))
    cache = folder / "cache/structure"
    cache.mkdir(parents=True, exist_ok=True)
    completed = 0
    while (cache / f"{completed:03}.npz").exists():
        with np.load(cache / f"{completed:03}.npz") as data:
            if str(data["signature"]) != signature:
                raise ValueError("Invalid structure cache identity")
        completed += 1
    edges = np.logspace(-100, 100, 8193)
    coefficients_gpu = [
        torch.tensor(view["coefficients"].T, dtype=torch.float64, device="cuda:0")
        for view in views
    ]
    include_external = any(
        np.any(view["coefficients"][len(job["snapshots"]) :])
        for view in views
        if view["precision"] == "raw"
    )
    n_views = len(views)
    current = None
    with blocks(job, layout, completed, include_external) as iterator:
        for index, name, start, stop, count, raw, evaluated, _ in iterator:
            if name != current:
                current = name
                counts = np.zeros((n_views, 8192), dtype=np.int64)
                masses = np.zeros((n_views, 8192), dtype=np.float64)
                fourth = np.zeros(n_views)
                token = (
                    np.zeros((n_views, layout["shapes"][name][0]))
                    if name in ("model.embed_tokens.weight", "lm_head.weight")
                    else np.zeros((n_views, 0))
                )
            for j, (view, coefficients) in enumerate(zip(views, coefficients_gpu)):
                vectors = raw if view["precision"] == "raw" else evaluated
                projected = coefficients[:, : len(vectors)] @ vectors
                if view["name"].endswith("-plane"):
                    projected = projected.float().double()
                energy = projected.square().sum(dim=0)
                positive = energy[energy > 0]
                if not torch.isfinite(energy).all():
                    raise ValueError("Nonfinite displacement energy")
                if len(positive):
                    bins = torch.floor(
                        (torch.log10(positive) + 100) * (8192 / 200)
                    ).long()
                    if bins.min() < 0 or bins.max() >= 8192:
                        raise ValueError("Energy histogram range exceeded")
                    counts[j] += torch.bincount(bins, minlength=8192).cpu().numpy()
                    masses[j] += (
                        torch.bincount(bins, weights=positive, minlength=8192)
                        .cpu()
                        .numpy()
                    )
                fourth[j] += float(energy.square().sum())
                if token.shape[1]:
                    width = layout["shapes"][name][1]
                    if start % width or (stop - start) % width:
                        raise ValueError(
                            "Token matrix chunks must align with token rows"
                        )
                    token[j, start // width : stop // width] = (
                        energy.reshape(-1, width).sum(dim=1).cpu().numpy()
                    )
            if stop == count:
                save_npz(
                    cache / f"{index:03}.npz",
                    counts=counts,
                    masses=masses,
                    fourth=fourth,
                    tokens=token,
                    signature=np.array(signature),
                )
                publish(
                    folder / "status/structure.json",
                    {"state": "running", "completed_parameters": index + 1},
                )
                print("Structured", index + 1, name, flush=True)
    all_counts = np.zeros((n_views, 8192), dtype=np.int64)
    all_masses = np.zeros((n_views, 8192))
    all_fourth = np.zeros(n_views)
    rows, token_arrays, trajectory_rows, block_overlaps = [], {}, [], []
    for index, name in enumerate(layout["names"]):
        category, layer, component = classify_parameter(name)
        count = math.prod(layout["shapes"][name])
        storage = read(folder / "cache/scan" / f"{index:03}.json")["dtypes"]
        with np.load(folder / "cache/scan" / f"{index:03}.npz") as block:
            for precision in ("raw", "evaluated"):
                for group in job.get("groups", []):
                    indices = group["indices"]
                    energy = float(np.trace(block[precision][np.ix_(indices, indices)]))
                    total = float(np.trace(grams[precision][np.ix_(indices, indices)]))
                    trajectory_rows.append(
                        {
                            "trajectory": group["name"],
                            "precision": precision,
                            "parameter": name,
                            "category": category,
                            "layer": layer,
                            "component": component,
                            "parameters": count,
                            "energy_fraction": energy / total,
                            "enrichment": energy
                            / total
                            / (count / layout["parameters"]),
                        }
                    )
            for pair in job.get("phase_pairs", []):
                defense, attack = pair["defense_endpoint"], pair["attack_endpoint"]
                for precision in ("raw", "evaluated"):
                    matrix = block[precision]
                    denominator = math.sqrt(
                        max(0, matrix[defense, defense] * matrix[attack, attack])
                    )
                    left = np.asarray(
                        subspaces["coefficients"][
                            f"{pair['defense_group']}-{precision}-plane"
                        ]
                    )
                    right = np.asarray(
                        subspaces["coefficients"][
                            f"{pair['attack_group']}-{precision}-plane"
                        ]
                    )
                    overlap_data = overlap(matrix, left, right)
                    block_overlaps.append(
                        {
                            "parameter": name,
                            "category": category,
                            "layer": layer,
                            "precision": precision,
                            "endpoint_cosine": (
                                float(matrix[defense, attack] / denominator)
                                if denominator
                                else None
                            ),
                            "block_plane_overlap": overlap_data["mean_cosine_squared"],
                            "left_block_rank": overlap_data["left_rank"],
                            "right_block_rank": overlap_data["right_rank"],
                        }
                    )
        with np.load(cache / f"{index:03}.npz") as data:
            all_counts += data["counts"]
            all_masses += data["masses"]
            all_fourth += data["fourth"]
            if data["tokens"].shape[1]:
                token_arrays[category] = data["tokens"].copy()
            for j, view in enumerate(views):
                energy = float(data["masses"][j].sum())
                predicted = float(
                    np.trace(
                        view["coefficients"].T
                        @ grams[view["precision"]]
                        @ view["coefficients"]
                    )
                )
                rows.append(
                    {
                        "view": view["name"],
                        "precision": view["precision"],
                        "parameter": name,
                        "category": category,
                        "layer": layer,
                        "component": component,
                        "storage_dtypes": "|".join(sorted(set(storage.values()))),
                        "parameters": count,
                        "energy": energy,
                        "total_expected_energy": predicted,
                        "energy_fraction": energy / predicted if predicted else 0,
                        "parameter_fraction": count / layout["parameters"],
                        "enrichment": (
                            energy / predicted / (count / layout["parameters"])
                            if predicted
                            else 0
                        ),
                        "rms_displacement": math.sqrt(energy / count),
                        "nonzero_fraction": float(data["counts"][j].sum()) / count,
                    }
                )
    summaries = {}
    for j, view in enumerate(views):
        summary = concentration(
            all_counts[j], all_masses[j], edges, layout["parameters"], all_fourth[j]
        )
        expected = float(
            np.trace(
                view["coefficients"].T @ grams[view["precision"]] @ view["coefficients"]
            )
        )
        if not math.isclose(summary["energy"], expected, rel_tol=2e-6, abs_tol=1e-12):
            raise ValueError(f"Streamed and Gram energies disagree: {view['name']}")
        summaries[view["name"]] = {
            "precision": view["precision"],
            "dimensions": view["coefficients"].shape[1],
            **summary,
        }
    save_npz(
        folder / "token-row-energy.npz",
        names=np.array([view["name"] for view in views]),
        **token_arrays,
    )
    save_npz(
        folder / "coordinate-histograms.npz",
        names=np.array([view["name"] for view in views]),
        edges=edges,
        counts=all_counts,
        masses=all_masses,
        fourth=all_fourth,
    )
    write_csv(folder / "parameter-structure.csv", rows)
    write_csv(folder / "trajectory-energy.csv", trajectory_rows)
    write_csv(folder / "block-defense-attack-alignment.csv", block_overlaps)
    publish(
        folder / "concentration.json",
        {
            "state": "complete",
            "views": summaries,
            "histogram_note": "Exact nonzero counts and bin masses; energy-covering coordinate fractions are bounded by logarithmic bin edges, not exact global sorts. Basis energies sum squared entries across the whole fitted plane and are invariant to orthogonal changes of basis within that plane.",
        },
    )
    publish(folder / "status/structure.json", {"state": "complete"})


def token_frequencies(path, vocabulary, role=None):
    counts = {
        key: np.zeros(vocabulary, dtype=np.int64)
        for key in (
            "attended_input",
            "all_input",
            "supervised_target",
            "attended_target",
        )
    }
    batches = 0
    for line in Path(path).read_text().splitlines():
        batch = json.loads(line)["batch"]
        for row_index, (ids, mask, labels) in enumerate(
            zip(batch["input_ids"], batch["attention_mask"], batch["labels"])
        ):
            if role is not None and batch["metadata_ids"][row_index] != role:
                continue
            ids, mask, labels = (
                np.asarray(ids),
                np.asarray(mask).astype(bool),
                np.asarray(labels),
            )
            counts["all_input"] += np.bincount(ids, minlength=vocabulary)
            counts["attended_input"] += np.bincount(ids[mask], minlength=vocabulary)
            valid = labels[1:] != -100
            counts["supervised_target"] += np.bincount(
                labels[1:][valid], minlength=vocabulary
            )
            counts["attended_target"] += np.bincount(
                labels[1:][valid & mask[1:]], minlength=vocabulary
            )
        batches += 1
    return counts, batches


def report(folder, job):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    layout = read(job["layout"])
    data = read(folder / "concentration.json")
    with (folder / "parameter-structure.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = list(data["views"])
    planes = [name for name in names if name.endswith("-plane")]
    categories = ["embedding", "output_head", "norm", "attention", "mlp", "other"]
    aggregates = []
    for name in names:
        for category in categories:
            selected = [
                row
                for row in rows
                if row["view"] == name and row["category"] == category
            ]
            size = sum(int(row["parameters"]) for row in selected)
            energy = sum(float(row["energy"]) for row in selected)
            total = data["views"][name]["energy"]
            aggregates.append(
                {
                    "view": name,
                    "category": category,
                    "parameters": size,
                    "parameter_fraction": size / layout["parameters"],
                    "energy_fraction": energy / total if total else 0,
                    "enrichment": (
                        energy / total / (size / layout["parameters"])
                        if total and size
                        else 0
                    ),
                }
            )
    write_csv(folder / "module-structure.csv", aggregates)
    subspaces = read(folder / "subspaces.json")
    if job.get("groups"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for precision, axis in zip(("raw", "evaluated"), axes):
            selected = [
                key for key in subspaces["coefficients"] if f"-{precision}-plane" in key
            ]
            matrix = np.eye(len(selected))
            for item in subspaces["overlaps"]:
                if item["left"] in selected and item["right"] in selected:
                    i, j = selected.index(item["left"]), selected.index(item["right"])
                    matrix[i, j] = matrix[j, i] = item["mean_cosine_squared"]
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            axis.set_xticks(
                range(len(selected)),
                [key.replace(f"-{precision}-plane", "") for key in selected],
                rotation=35,
                ha="right",
                fontsize=8,
            )
            axis.set_yticks(
                range(len(selected)),
                [key.replace(f"-{precision}-plane", "") for key in selected],
                fontsize=8,
            )
            axis.set_title(precision + " phase-plane overlap")
            for i in range(len(selected)):
                for j in range(len(selected)):
                    axis.text(
                        j,
                        i,
                        f"{matrix[i,j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if matrix[i, j] < 0.5 else "black",
                    )
        fig.colorbar(
            image, ax=list(axes), label="Mean squared cosine of principal angles"
        )
        fig.savefig(folder / "defense-attack-plane-overlap.png", dpi=160)
        fig.savefig(folder / "defense-attack-plane-overlap.pdf")
        plt.close(fig)
        with (folder / "snapshot-capture.csv").open(newline="") as stream:
            captures = list(csv.DictReader(stream))
        attack_names = {
            item["name"]: item["step"]
            for item in job["snapshots"]
            if item["name"].startswith("attack-")
        }
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for precision, axis in zip(("raw", "evaluated"), axes):
            for key in subspaces["coefficients"]:
                if f"-{precision}-" not in key or key.startswith("collaborator"):
                    continue
                points = [
                    row
                    for row in captures
                    if row["basis"] == key and row["snapshot"] in attack_names
                ]
                if points:
                    axis.plot(
                        [attack_names[row["snapshot"]] for row in points],
                        [float(row["captured_fraction"]) for row in points],
                        "o-",
                        label=key,
                    )
            axis.set(
                xscale="log",
                xlabel="Saved attack update",
                ylabel="Squared displacement captured",
                ylim=(0, 1.02),
                title=precision,
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=6)
        fig.savefig(folder / "attack-capture-by-defense-subspace.png", dpi=160)
        fig.savefig(folder / "attack-capture-by-defense-subspace.pdf")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(planes))
    for i, name in enumerate(planes):
        selected = [row for row in aggregates if row["view"] == name]
        axes[0].bar(
            x + (i - (len(planes) - 1) / 2) * width,
            [100 * row["energy_fraction"] for row in selected],
            width,
            label=name,
        )
        axes[1].plot(
            x, [max(row["enrichment"], 1e-12) for row in selected], "o-", label=name
        )
    for axis in axes:
        axis.set_xticks(x, categories, rotation=30)
        axis.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("Share of plane squared-coordinate energy (%)")
    axes[1].set_ylabel("Energy share / parameter share")
    axes[1].set_yscale("log")
    axes[1].axhline(1, color="black", linestyle=":", label="isotropic size baseline")
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "module-concentration.png", dpi=160)
    fig.savefig(folder / "module-concentration.pdf")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name in planes:
        summary = data["views"][name]
        xx, low, high = [], [], []
        for fraction, bound in summary["top_fractions"].items():
            xx.append(float(fraction))
            low.append(bound["energy_fraction_lower"])
            high.append(bound["energy_fraction_upper"])
        axis.plot(xx, (np.asarray(low) + high) / 2, marker="o", label=name)
        axis.fill_between(xx, low, high, alpha=0.1)
    fraction = np.logspace(-6, 0, 200)
    axis.plot(
        fraction,
        fraction * (1 - np.log(fraction)),
        "k:",
        label="ideal isotropic rank-two Gaussian plane",
    )
    axis.set(
        xscale="log",
        xlabel="Fraction of largest-energy parameter coordinates",
        ylabel="Fraction of squared plane energy",
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    fig.savefig(folder / "coordinate-concentration.png", dpi=160)
    fig.savefig(folder / "coordinate-concentration.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        len(planes),
        1,
        figsize=(11, max(3, 2.8 * len(planes))),
        constrained_layout=True,
        squeeze=False,
    )
    components = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "input_layernorm",
        "post_attention_layernorm",
    ]
    for name, axis in zip(planes, axes[:, 0]):
        matrix = np.full((len(components), 32), np.nan)
        for row in rows:
            if (
                row["view"] == name
                and int(row["layer"]) >= 0
                and row["component"] in components
            ):
                matrix[components.index(row["component"]), int(row["layer"])] = (
                    math.log10(max(float(row["enrichment"]), 1e-12))
                )
        artist = axis.imshow(matrix, aspect="auto", vmin=-4, vmax=4, cmap="coolwarm")
        axis.set_yticks(range(len(components)), components, fontsize=7)
        axis.set_title(name)
        axis.set_xlabel("Transformer layer")
    fig.colorbar(
        artist, ax=list(axes[:, 0]), label="log10 energy enrichment per parameter"
    )
    fig.savefig(folder / "layer-module-enrichment.png", dpi=160)
    fig.savefig(folder / "layer-module-enrichment.pdf")
    plt.close(fig)
    vocabulary = layout["shapes"]["model.embed_tokens.weight"][0]
    frequencies, batch_count = token_frequencies(job["observations"], vocabulary)
    defense_frequencies = {}
    if job.get("defense_observations"):
        for role, label in (
            (0, "defense_forget"),
            (1, "defense_retain"),
            (2, "defense_inner"),
        ):
            role_counts, _ = token_frequencies(
                job["defense_observations"], vocabulary, role
            )
            defense_frequencies[label] = role_counts["attended_input"]
    tokenizer = AutoTokenizer.from_pretrained(job["tokenizer"], local_files_only=True)
    top_rows, token_summary = [], []
    with np.load(folder / "token-row-energy.npz") as token_data:
        np.testing.assert_array_equal(token_data["names"], names)
        for category in ("embedding", "output_head"):
            if category not in token_data:
                continue
            energies = token_data[category]
            for name, energy in zip(names, energies):
                total = float(energy.sum())
                order = np.argsort(energy)[::-1]
                cumulative = np.cumsum(energy[order])
                token_summary.append(
                    {
                        "view": name,
                        "matrix": category,
                        "nonzero_token_rows": int(np.count_nonzero(energy)),
                        "rows_for_90_percent_energy": (
                            int(np.searchsorted(cumulative, 0.9 * total) + 1)
                            if total
                            else 0
                        ),
                        "top_10_energy_fraction": (
                            float(energy[order[:10]].sum() / total) if total else 0
                        ),
                        "top_100_energy_fraction": (
                            float(energy[order[:100]].sum() / total) if total else 0
                        ),
                        "energy_in_unseen_attended_input_rows": (
                            float(
                                energy[frequencies["attended_input"] == 0].sum() / total
                            )
                            if total
                            else 0
                        ),
                        "energy_in_unseen_all_input_rows": (
                            float(energy[frequencies["all_input"] == 0].sum() / total)
                            if total
                            else 0
                        ),
                    }
                )
                for rank, token_id in enumerate(order[:100], 1):
                    top_rows.append(
                        {
                            "view": name,
                            "matrix": category,
                            "rank": rank,
                            "token_id": int(token_id),
                            "token": repr(
                                tokenizer.decode(
                                    [int(token_id)], skip_special_tokens=False
                                )
                            ),
                            "energy_fraction_within_matrix": (
                                float(energy[token_id] / total) if total else 0
                            ),
                            **{
                                key: int(value[token_id])
                                for key, value in {
                                    **frequencies,
                                    **defense_frequencies,
                                }.items()
                            },
                        }
                    )
        for category in ("embedding", "output_head"):
            if category not in token_data:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
            for name in planes:
                energy = token_data[category][names.index(name)]
                total = energy.sum()
                normalized = energy / total if total else energy
                order = np.argsort(normalized)[::-1]
                axes[0].plot(
                    np.arange(1, vocabulary + 1),
                    np.cumsum(normalized[order]),
                    label=name,
                )
                axes[1].scatter(
                    frequencies["attended_input"] + 1,
                    normalized + 1e-30,
                    s=3,
                    alpha=0.3,
                    label=name,
                )
            axes[0].set(
                xscale="log",
                xlabel="Highest-energy token rows",
                ylabel="Cumulative energy within matrix",
                ylim=(0, 1.02),
            )
            axes[1].set(
                xscale="log",
                yscale="log",
                xlabel="Attended attack input occurrences + 1",
                ylabel="Token-row energy share within matrix",
            )
            axes[0].legend(fontsize=7)
            fig.savefig(folder / f"{category}-token-concentration.png", dpi=160)
            fig.savefig(folder / f"{category}-token-concentration.pdf")
            plt.close(fig)
    write_csv(folder / "top-token-rows.csv", top_rows)
    write_csv(folder / "token-summary.csv", token_summary)
    write_csv(
        folder / "token-frequencies.csv",
        [
            {
                "token_id": index,
                **{
                    key: int(value[index])
                    for key, value in {**frequencies, **defense_frequencies}.items()
                },
            }
            for index in range(vocabulary)
        ],
    )
    publish(
        folder / "report-validation.json",
        {
            "state": "passed",
            "observed_batches": batch_count,
            "views": len(names),
            "planes": len(planes),
            "independent_input_output_matrices": True,
            "parameters": layout["parameters"],
            "storage_precision_note": "Raw means dequantized saved mixed BF16/FP32 tensors; evaluated means individually BF16-cast saved weights before subtraction. Zero saved motion does not identify zero optimizer or gradient motion.",
        },
    )
    publish(folder / "status/report.json", {"state": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("scan", "structure", "report", "run"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if not os.environ.get("FLUX_JOB_ID"):
        raise ValueError("Run structure analysis through launch_tuo.py")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(8)
    if torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one analysis GPU")
    folder = args.folder.resolve()
    job = read(folder / "job.json")
    (folder / "status").mkdir(exist_ok=True)
    publish(folder / "status/analysis.json", {"state": "running"})
    smoke_values = torch.tensor(
        [0.25, 1.0, 4.0, 1.0], dtype=torch.float64, device="cuda:0"
    )
    smoke_bins = torch.tensor([0, 1, 2, 1], device="cuda:0")
    np.testing.assert_allclose(
        torch.bincount(smoke_bins, weights=smoke_values, minlength=3).cpu().numpy(),
        [0.25, 2, 4],
    )
    smoke_gram = torch.stack([smoke_values, -smoke_values])
    np.testing.assert_allclose(
        (smoke_gram @ smoke_gram.T).cpu().numpy(),
        np.array([[18.0625, -18.0625], [-18.0625, 18.0625]]),
    )
    publish(
        folder / "gpu-preflight.json",
        {
            "state": "passed",
            "fp64_gram": True,
            "fp64_weighted_histogram": True,
            "visible_devices": 1,
        },
    )
    publish(
        folder / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "hip": torch.version.hip,
            "device": str(torch.cuda.get_device_properties(0)),
            "flux_job_id": os.environ["FLUX_JOB_ID"],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    print(
        "Structure device:",
        torch.cuda.get_device_name(0),
        "job",
        os.environ["FLUX_JOB_ID"],
        flush=True,
    )
    stages = ("scan", "structure", "report") if args.stage == "run" else (args.stage,)
    for stage in stages:
        marker = folder / "status" / (stage + ".json")
        if (
            marker.exists()
            and read(marker).get("state") == "complete"
            and not (stage == "report" and args.stage == "report")
        ):
            print("Reusing", stage, flush=True)
            continue
        publish(marker, {"state": "running"})
        try:
            globals()[stage](folder, job)
        except Exception as error:
            publish(marker, {"state": "failed", "error": repr(error)})
            raise
    publish(folder / "status/analysis.json", {"state": "complete"})


if __name__ == "__main__":
    main()
