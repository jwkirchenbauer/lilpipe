"""Measure baseline attack paths and fixed/moving loss slices on Tuolumne."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import runpy
from pathlib import Path

import numpy as np


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plane_coefficients(
    gram,
    n_attack,
    reference_names=(),
    *,
    planes=None,
    allow_degenerate=False,
    preserve_shared=False,
):
    """Fit planes in a Gram matrix of attack, random, and reference vectors.

    The two random vectors follow the attack vectors. Each named reference
    contributes two more columns, in ``reference_names`` order. Shared FP32
    directions are selected exactly, after checking their orthonormality.
    """
    gram = np.asarray(gram, dtype=np.float64)
    reference_names = list(reference_names)
    expected = n_attack + 2 + 2 * len(reference_names)
    if n_attack < 1 or gram.shape != (expected, expected):
        raise ValueError("Gram matrix does not match attack and reference vectors")
    reserved = {"random", "attack", "transverse"}
    if len(set(reference_names)) != len(reference_names) or reserved.intersection(
        reference_names
    ):
        raise ValueError("Reference plane names must be unique and nonreserved")
    plane_order = ["random", "attack", "transverse"] if planes is None else list(planes)
    if (
        not plane_order
        or len(set(plane_order)) != len(plane_order)
        or set(plane_order) - reserved - set(reference_names)
    ):
        raise ValueError("Visualization planes must be known, unique, and nonempty")
    attack = gram[:n_attack, :n_attack]
    values, vectors = np.linalg.eigh(attack)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    valid = values > max(values[0] * 1e-11, 1e-14)
    rank = int(np.sum(valid))
    if rank < 2 and not allow_degenerate:
        raise ValueError(
            "The recorded attack has fewer than two independent directions"
        )
    inverse = (vectors[:, valid] / values[valid]) @ vectors[:, valid].T

    def normalize(coefficients):
        small = coefficients.T @ gram @ coefficients
        eigenvalues, eigenvectors = np.linalg.eigh(small)
        if eigenvalues.min() < 1e-12:
            raise ValueError("Degenerate visualization plane")
        result = coefficients @ ((eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T)
        np.testing.assert_allclose(result.T @ gram @ result, np.eye(2), atol=1e-7)
        return result

    random = np.zeros((expected, 2))
    random[n_attack : n_attack + 2, :] = np.eye(2)
    coefficients = {}
    for name in plane_order:
        if name == "random":
            if preserve_shared:
                np.testing.assert_allclose(
                    random.T @ gram @ random, np.eye(2), atol=1e-5
                )
                coefficients[name] = random.copy()
            else:
                coefficients[name] = normalize(random)
        elif name == "attack":
            if rank == 0:
                coefficients[name] = normalize(random)
                continue
            aligned = np.zeros_like(random)
            aligned[:n_attack, : min(rank, 2)] = vectors[:, : min(rank, 2)] / np.sqrt(
                values[: min(rank, 2)]
            )
            if rank == 1:
                index = int(
                    np.flatnonzero(np.diag(attack) > max(values[0] * 1e-11, 1e-14))[0]
                )
                aligned[:, 0] = 0
                aligned[index, 0] = 1 / np.sqrt(attack[index, index])
                first = aligned[:, :1]
                completion = random - first @ (first.T @ gram @ random)
                energies = np.diag(completion.T @ gram @ completion)
                index = int(np.argmax(energies))
                if energies[index] < 1e-12:
                    raise ValueError(
                        "Random directions cannot complete the attack plane"
                    )
                aligned[:, 1] = completion[:, index] / np.sqrt(energies[index])
            coefficients[name] = normalize(aligned)
        elif name == "transverse":
            transverse = random.copy()
            transverse[:n_attack] = -inverse @ gram[:n_attack, n_attack : n_attack + 2]
            coefficients[name] = normalize(transverse)
            np.testing.assert_allclose(
                coefficients[name].T @ gram[:, :n_attack], 0, atol=1e-7
            )
        else:
            reference = np.zeros_like(random)
            start = n_attack + 2 + 2 * reference_names.index(name)
            reference[start : start + 2] = np.eye(2)
            np.testing.assert_allclose(
                reference.T @ gram @ reference, np.eye(2), atol=1e-5
            )
            coefficients[name] = reference
    return coefficients, {
        "attack_rank": rank,
        "attack_eigenvalues": values.tolist(),
        "attack_plane_energy_fraction": (
            float(values[: min(rank, 2)].sum() / values.sum()) if rank else 0.0
        ),
    }


def paths(study, settings):
    return [
        study / "models/attack" / f"checkpoint-{step}"
        for step in settings["attack_steps"]
    ]


def _geometry_progress_save(path, value):
    """Publish small progress only after its associated basis data are flushed."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
    temporary.replace(path)


def _geometry_progress_load(path, identity, parameter_count, arrays):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        completed = value["completed_parameters"]
        if (
            value["identity"] != identity
            or type(completed) is not int
            or not 0 <= completed <= parameter_count
        ):
            raise ValueError("identity or parameter prefix differs")
        for name, shape in arrays.items():
            array = np.asarray(value[name], dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"invalid {name} accumulator")
        return value
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Ignoring invalid geometry progress {path.name}: {error}", flush=True)
        return None


def geometry(study, settings):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = Path(settings.get("attack_anchor", study / "models/defense"))
    shared_paths = {}
    if settings.get("shared_random_basis"):
        shared_paths["random"] = Path(settings["shared_random_basis"])
    reference_bases = settings.get("reference_bases", {})
    if {"random", "attack", "transverse"}.intersection(reference_bases):
        raise ValueError("Reference plane names must not replace built-in planes")
    shared_paths.update({name: Path(path) for name, path in reference_bases.items()})
    if any(not path.is_absolute() for path in shared_paths.values()):
        raise ValueError("Shared basis paths must be absolute")
    model = (
        AutoModelForCausalLM.from_pretrained(
            reference, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(device)
        .eval()
    )
    names = [name for name, _ in model.named_parameters()]
    shapes = {
        name: list(parameter.shape) for name, parameter in model.named_parameters()
    }
    random = []
    if "random" not in shared_paths:
        helper = runpy.run_path(str(study / "source/misc/reference_get_ll.py"))
        directions = helper["get_directions"](model, settings["direction_seed"])
        random = [
            {name: value.cpu() for name, value in direction.items()}
            for direction in directions
        ]
        del directions
    del model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoints = paths(study, settings)
    n = len(checkpoints)
    chunk = 4 * 1024 * 1024
    comparisons = []
    if settings.get("compare_original_weights", True):
        original = Path(settings["original_run"])
        comparisons = [("defense", reference, original / "models/defense")]
        comparisons += [
            (
                f"attack-{step}",
                checkpoint,
                original / "models/attack-lr1e-6" / f"checkpoint-{step}",
            )
            for step, checkpoint in zip(settings["attack_steps"], checkpoints)
            if step % 16 == 0
        ]
    differences = {name: 0.0 for name, _, _ in comparisons}
    vector_count = n + 2 + 2 * len(reference_bases)
    gram_gpu = torch.zeros(
        (vector_count, vector_count), dtype=torch.float64, device=device
    )
    offsets, total = {}, 0
    for name in names:
        offsets[name] = total
        total += math.prod(shapes[name])
    shared = {}
    for plane, path in shared_paths.items():
        if not isinstance(plane, str) or Path(plane).name != plane:
            raise ValueError("Reference plane names must be plain filenames")
        basis = np.load(path, mmap_mode="r", allow_pickle=False)
        if basis.dtype != np.float32 or basis.shape != (2, total):
            raise ValueError(f"Shared basis has the wrong dtype or shape: {path}")
        metadata = path.parent / "geometry.json"
        if metadata.exists():
            source = json.loads(metadata.read_text())
            for key, expected in (
                ("names", names),
                ("shapes", shapes),
                ("offsets", offsets),
            ):
                if source[key] != expected:
                    raise ValueError(f"Shared basis parameter {key} differ: {path}")
        shared[plane] = basis

    cache_enabled = settings.get(
        "geometry_progress_cache", bool(settings.get("shared_random_basis"))
    )
    progress_root = study / "geometry"
    gram_progress_path = progress_root / "gram-progress.json"
    basis_progress_path = progress_root / "basis-progress.json"
    identity, gram_progress = None, None
    if cache_enabled:

        def input_identity(path):
            path = Path(path)
            metadata = path.stat()
            return {
                "path": str(path.absolute()),
                "resolved": str(path.resolve()),
                "size": metadata.st_size,
                "mtime_ns": metadata.st_mtime_ns,
            }

        inputs = [reference / "model.safetensors"]
        inputs += [path / "model.safetensors" for path in checkpoints]
        inputs += [old / "model.safetensors" for _, _, old in comparisons]
        inputs += list(shared_paths.values())
        inputs += [
            path.parent / "geometry.json"
            for path in shared_paths.values()
            if (path.parent / "geometry.json").exists()
            and path.parent.resolve() != progress_root.resolve()
        ]
        if not shared_paths.get("random"):
            inputs.append(study / "source/misc/reference_get_ll.py")
        protocol = {
            "version": 1,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "settings": settings,
            "names": names,
            "shapes": shapes,
            "offsets": offsets,
            "chunk": chunk,
            "accumulator_dtype": "float64",
            "basis_dtype": "float32",
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "inputs": [input_identity(path) for path in inputs],
        }
        serialized = json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), default=str
        )
        identity = hashlib.sha256(serialized.encode()).hexdigest()
        _geometry_progress_save(
            progress_root / "progress-protocol.json",
            {"identity": identity, "protocol": json.loads(serialized)},
        )
        gram_progress = _geometry_progress_load(
            gram_progress_path,
            identity,
            len(names),
            {"gram": (vector_count, vector_count)},
        )
        if gram_progress is not None:
            saved_differences = gram_progress.get("squared_differences", {})
            if set(saved_differences) != set(differences) or any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in saved_differences.values()
            ):
                print("Ignoring invalid geometry comparison progress", flush=True)
                gram_progress = None
            else:
                differences = saved_differences
                gram_gpu.copy_(
                    torch.tensor(
                        gram_progress["gram"], dtype=torch.float64, device=device
                    )
                )
        # A geometry result is published again only once both passes are complete.
        (progress_root / "geometry.json").unlink(missing_ok=True)
    gram_completed = gram_progress["completed_parameters"] if gram_progress else 0
    parameter_indices = {name: index for index, name in enumerate(names)}

    with contextlib.ExitStack() as stack:
        base = stack.enter_context(
            safe_open(reference / "model.safetensors", framework="pt", device="cpu")
        )
        handles = [
            stack.enter_context(
                safe_open(path / "model.safetensors", framework="pt", device="cpu")
            )
            for path in checkpoints
        ]
        old_handles = [
            (
                name,
                stack.enter_context(
                    safe_open(old / "model.safetensors", framework="pt", device="cpu")
                ),
                (
                    None
                    if name == "defense"
                    else settings["attack_steps"].index(int(name.split("-")[1]))
                ),
            )
            for name, _, old in comparisons
        ]

        def blocks(dtype, completed=0):
            for name in names[completed:]:
                center = base.get_tensor(name).reshape(-1)
                tensors = [handle.get_tensor(name).reshape(-1) for handle in handles]
                for start in range(0, center.numel(), chunk):
                    stop = min(start + chunk, center.numel())
                    anchor = center[start:stop].to(device=device, dtype=dtype)
                    deltas = [
                        tensor[start:stop].to(device=device, dtype=dtype) - anchor
                        for tensor in tensors
                    ]
                    # Materialize each mmap slice once per chunk before the HIP
                    # copy. Keep only this chunk's shared vectors on the device.
                    offset = offsets[name] + start
                    cached = {
                        plane: torch.from_numpy(
                            np.array(
                                basis[:, offset : offset + stop - start], copy=True
                            )
                        ).to(device=device, dtype=dtype)
                        for plane, basis in shared.items()
                    }
                    random_vectors = (
                        list(cached["random"])
                        if "random" in cached
                        else [
                            direction[name]
                            .reshape(-1)[start:stop]
                            .to(device=device, dtype=dtype)
                            for direction in random
                        ]
                    )
                    vectors = deltas + random_vectors
                    for plane in reference_bases:
                        vectors.extend(cached[plane])
                    yield name, start, stop, anchor, torch.stack(vectors)

        print("Computing FP64 trajectory Gram matrix", flush=True)
        last = None
        if gram_completed:
            print(
                f"Resuming Gram after {gram_completed}/{len(names)} parameters",
                flush=True,
            )
        for name, start, stop, anchor, vectors in blocks(torch.float64, gram_completed):
            gram_gpu.add_(vectors @ vectors.T)
            for label, old_handle, index in old_handles:
                old = (
                    old_handle.get_tensor(name)
                    .reshape(-1)[start:stop]
                    .to(device=device, dtype=torch.float64)
                )
                current = anchor if index is None else anchor + vectors[index]
                differences[label] += float(torch.sum((current - old).square()))
            if name != last:
                print("Gram:", name, flush=True)
                last = name
            if cache_enabled and stop == math.prod(shapes[name]):
                _geometry_progress_save(
                    gram_progress_path,
                    {
                        "identity": identity,
                        "completed_parameters": parameter_indices[name] + 1,
                        "gram": gram_gpu.cpu().numpy().tolist(),
                        "squared_differences": differences,
                    },
                )
        gram = gram_gpu.cpu().numpy()
        coefficients, details = plane_coefficients(
            gram,
            n,
            reference_bases,
            planes=settings.get("planes"),
            allow_degenerate=settings.get("allow_degenerate", False),
            preserve_shared="random" in shared,
        )
        plane_order = list(coefficients)
        combined = np.concatenate([coefficients[name] for name in plane_order], axis=1)
        combined_gpu = torch.tensor(combined.T, dtype=torch.float64, device=device)
        dimensions = 2 * len(plane_order)
        actual_gram = torch.zeros(
            (dimensions, dimensions), dtype=torch.float64, device=device
        )
        actual_cross = torch.zeros((n, dimensions), dtype=torch.float64, device=device)
        coefficient_identity = hashlib.sha256(
            combined.tobytes() + json.dumps(plane_order).encode()
        ).hexdigest()
        basis_progress = (
            _geometry_progress_load(
                basis_progress_path,
                identity,
                len(names),
                {
                    "actual_gram": (dimensions, dimensions),
                    "actual_cross": (n, dimensions),
                },
            )
            if cache_enabled
            else None
        )
        owned = {
            plane: progress_root / f"{plane}-basis.npy"
            for plane in plane_order
            if plane not in shared
        }

        def owned_identity():
            result = {}
            for plane, target in owned.items():
                if target.is_symlink():
                    raise ValueError("An owned basis is a symlink")
                metadata = target.stat()
                result[plane] = {
                    "path": str(target.absolute()),
                    "size": metadata.st_size,
                    "inode": metadata.st_ino,
                }
            return result

        files = {}
        if basis_progress is not None:
            try:
                if (
                    basis_progress.get("coefficient_identity") != coefficient_identity
                    or basis_progress.get("owned_files") != owned_identity()
                ):
                    raise ValueError("coefficient or owned basis identity differs")
                for plane, target in owned.items():
                    file = np.load(target, mmap_mode="r+", allow_pickle=False)
                    if file.dtype != np.float32 or file.shape != (2, total):
                        raise ValueError("Owned basis dtype or shape differs")
                    files[plane] = file
                actual_gram.copy_(
                    torch.tensor(
                        basis_progress["actual_gram"],
                        dtype=torch.float64,
                        device=device,
                    )
                )
                actual_cross.copy_(
                    torch.tensor(
                        basis_progress["actual_cross"],
                        dtype=torch.float64,
                        device=device,
                    )
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f"Ignoring invalid basis progress: {error}", flush=True)
                basis_progress = None
                files.clear()
        if basis_progress is None:
            for plane, target in owned.items():
                if target.is_symlink():
                    target.unlink()
                files[plane] = np.lib.format.open_memmap(
                    target, mode="w+", dtype=np.float32, shape=(2, total)
                )
        basis_completed = (
            basis_progress["completed_parameters"] if basis_progress else 0
        )

        def save_basis_progress(completed):
            for file in files.values():
                file.flush()
            _geometry_progress_save(
                basis_progress_path,
                {
                    "identity": identity,
                    "completed_parameters": completed,
                    "coefficient_identity": coefficient_identity,
                    "owned_files": owned_identity(),
                    "actual_gram": actual_gram.cpu().numpy().tolist(),
                    "actual_cross": actual_cross.cpu().numpy().tolist(),
                },
            )

        if cache_enabled and basis_progress is None:
            save_basis_progress(0)
        if basis_completed:
            print(
                f"Resuming bases after {basis_completed}/{len(names)} parameters",
                flush=True,
            )
        print(
            f"Constructing and saving {len(plane_order)} visualization planes",
            flush=True,
        )
        for name, start, stop, anchor, vectors in blocks(
            torch.float64, basis_completed
        ):
            basis = (combined_gpu @ vectors).float()
            accurate = basis.double()
            actual_gram.add_(accurate @ accurate.T)
            actual_cross.add_(vectors[:n] @ accurate.T)
            host = basis.cpu().numpy()
            offset = offsets[name] + start
            for i, plane in enumerate(plane_order):
                if plane in files:
                    files[plane][:, offset : offset + stop - start] = host[
                        2 * i : 2 * i + 2
                    ]
            if cache_enabled and stop == math.prod(shapes[name]):
                save_basis_progress(parameter_indices[name] + 1)
        for file in files.values():
            file.flush()
        for plane in plane_order:
            if plane not in shared:
                continue
            source = shared_paths[plane].resolve()
            target = study / "geometry" / f"{plane}-basis.npy"
            if target.resolve() != source:
                target.unlink(missing_ok=True)
                target.symlink_to(source)
        actual_gram = actual_gram.cpu().numpy()
        actual_cross = actual_cross.cpu().numpy()
    norm_sq = np.diag(gram)[:n]
    projections = {}
    for i, plane in enumerate(plane_order):
        small = actual_gram[2 * i : 2 * i + 2, 2 * i : 2 * i + 2]
        np.testing.assert_allclose(small, np.eye(2), atol=1e-5)
        coordinates = actual_cross[:, 2 * i : 2 * i + 2] @ np.linalg.inv(small)
        captured = np.einsum("ni,ij,nj->n", coordinates, small, coordinates)
        projections[plane] = [
            {
                "step": step,
                "c0": float(point[0]),
                "c1": float(point[1]),
                "distance": float(np.sqrt(norm)),
                "captured_fraction": (
                    float(np.clip(power / norm, 0, 1)) if norm else 1.0
                ),
                "off_plane_distance": float(np.sqrt(max(norm - power, 0))),
            }
            for step, point, norm, power in zip(
                settings["attack_steps"], coordinates, norm_sq, captured
            )
        ]
    overlap = None
    if "random" in plane_order and "transverse" in plane_order:
        first = 2 * plane_order.index("random")
        second = 2 * plane_order.index("transverse")
        overlap = float(
            np.sum(actual_gram[first : first + 2, second : second + 2] ** 2) / 2
        )
    result = {
        **details,
        "steps": settings["attack_steps"],
        "names": names,
        "shapes": shapes,
        "offsets": offsets,
        "parameters": total,
        "gram": gram.tolist(),
        "basis_actual_gram": actual_gram.tolist(),
        "planes": projections,
        "radius": float(
            max(settings.get("minimum_radius", 1e-3), 1.25 * np.sqrt(norm_sq.max()))
        ),
        "rerun_distance_from_original": {
            name: math.sqrt(value) for name, value in differences.items()
        },
        "random_transverse_overlap": overlap,
        "precision": "FP64 Gram/projections; FP32 stored orthonormal directions; BF16 model evaluations",
    }
    dump(study / "geometry/geometry.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "radius",
                    "attack_plane_energy_fraction",
                    "random_transverse_overlap",
                    "rerun_distance_from_original",
                )
            },
            indent=2,
        ),
        flush=True,
    )


def surfaces(study, settings):
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from safetensors import safe_open
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )

    geom = json.loads((study / "geometry/geometry.json").read_text())
    defense_root = Path(
        settings.get("defense_checkpoint_root", study / "models/defense")
    )
    reference = Path(settings.get("attack_anchor", study / "models/defense"))
    model = (
        AutoModelForCausalLM.from_pretrained(
            reference, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to("cuda:0")
        .eval()
    )
    parameters = dict(model.named_parameters())
    center = {name: value.detach().clone() for name, value in parameters.items()}
    tokenizer = AutoTokenizer.from_pretrained(reference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    helper = runpy.run_path(str(study / "source/misc/reference_get_ll.py"))
    fixed = [
        {key: value.to("cuda:0") for key, value in batch.items()}
        for batch in helper["get_dataloader"](
            "cais/wmdp-bio-forget-corpus",
            tokenizer,
            512,
            2,
            settings["probe_documents"],
        )
    ]
    corpus = load_dataset(
        "cais/wmdp-bio-forget-corpus",
        split="train",
        revision=settings.get("forget_revision"),
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if settings.get("fixed_source_indices") is not None:
        fixed_rows = [
            tokenizer(corpus[index]["text"], max_length=512, truncation=True)
            for index in settings["fixed_source_indices"]
        ]
        fixed = [
            {
                key: value.to("cuda:0")
                for key, value in collator(fixed_rows[i : i + 2]).items()
            }
            for i in range(0, len(fixed_rows), 2)
        ]
    records = [
        json.loads(line)
        for line in (study / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    maximum = settings.get("attack_max_steps", 128)
    assert [record["step"] for record in records] == list(range(1, maximum + 1))
    minibatches = [
        {
            key: torch.tensor(value, device="cuda:0")
            for key, value in record["batch"].items()
            if key in ("input_ids", "attention_mask", "labels")
        }
        for record in records
    ]
    seen = {
        tuple(batch["input_ids"][0][batch["attention_mask"][0].bool()].cpu().tolist())
        for batch in minibatches
    }
    heldout_rows, heldout_hashes, heldout_indices = [], [], []
    selected_indices = settings.get("heldout_source_indices")
    for index in (
        selected_indices if selected_indices is not None else range(1000, len(corpus))
    ):
        text = corpus[index]["text"]
        if not isinstance(text, str) or len(text) <= 50:
            continue
        tokens = tokenizer(text, max_length=512, truncation=True)
        if tuple(tokens["input_ids"]) in seen:
            if selected_indices is not None:
                raise ValueError(
                    f"Shared held-out document {index} was observed by this attacker"
                )
            continue
        heldout_rows.append(tokens)
        heldout_hashes.append(hashlib.sha256(text.encode()).hexdigest())
        heldout_indices.append(index)
        if len(heldout_rows) == settings["heldout_documents"]:
            break
    assert len(heldout_rows) == settings["heldout_documents"]
    heldout = [
        {
            key: value.to("cuda:0")
            for key, value in collator(heldout_rows[i : i + 2]).items()
        }
        for i in range(0, len(heldout_rows), 2)
    ]
    dump(
        study / "surfaces/probes.json",
        {
            "fixed_documents": settings["probe_documents"],
            "fixed_source_indices": settings.get("fixed_source_indices"),
            "heldout_sha256": heldout_hashes,
            "heldout_source_indices": heldout_indices,
            "heldout_excludes_observed_attack_documents": True,
            "batch_surface_steps": settings["minibatch_surface_steps"],
            "attention": "eager",
        },
    )

    @torch.inference_mode()
    def loss(batches):
        return float(np.mean([float(model(**batch).loss) for batch in batches]))

    @torch.inference_mode()
    def batch_losses(batches):
        result = []
        for start in range(0, len(batches), 4):
            group = {
                key: torch.cat([batch[key] for batch in batches[start : start + 4]])
                for key in ("input_ids", "attention_mask", "labels")
            }
            logits = model(**group).logits[:, :-1].float()
            labels = group["labels"][:, 1:]
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).reshape_as(labels)
            values = losses.sum(dim=1) / (labels != -100).sum(dim=1)
            result.extend(values.cpu().tolist())
        return result

    @torch.inference_mode()
    def load(path):
        path = Path(path)
        index = path / "model.safetensors.index.json"
        mapping = (
            json.loads(index.read_text())["weight_map"]
            if index.exists()
            else {name: "model.safetensors" for name in parameters}
        )
        with contextlib.ExitStack() as stack:
            handles = {
                filename: stack.enter_context(
                    safe_open(path / filename, framework="pt", device="cpu")
                )
                for filename in set(mapping.values())
            }
            for name, parameter in parameters.items():
                # Fault file-backed pages into ordinary CPU memory before HIP
                # copies them. Avoid handing the driver a cold filesystem mmap.
                parameter.copy_(handles[mapping[name]].get_tensor(name).clone())

    @torch.inference_mode()
    def perturb(anchor, basis, c0, c1):
        for name, parameter in parameters.items():
            start = geom["offsets"][name]
            count = parameter.numel()
            first = basis[0, start : start + count].reshape(parameter.shape)
            second = basis[1, start : start + count].reshape(parameter.shape)
            parameter.copy_(anchor[name].float() + c0 * first + c1 * second)

    states = [("defense", 0, Path(settings["base_model"]))]
    states += [
        ("defense", step, defense_root / f"checkpoint-{step}")
        for step in settings["defense_steps"]
    ]
    states += [("attack", 0, reference)]
    states += [
        ("attack", step, study / "models/attack" / f"checkpoint-{step}")
        for step in settings.get("actual_attack_steps", settings["attack_steps"])
    ]

    def cached(filename):
        path = study / "surfaces" / filename
        if not path.exists() or path.stat().st_size < 2:
            return []
        frame = pd.read_csv(path, float_precision="round_trip")
        return [row.dropna().to_dict() for _, row in frame.iterrows()]

    def save_rows(filename, rows):
        target = study / "surfaces" / filename
        temporary = target.with_suffix(".csv.tmp")
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(target)

    actual = cached("actual-losses.csv")
    matrix = []
    matrix_steps = settings.get(
        "matrix_steps",
        [0] + settings.get("actual_attack_steps", settings["attack_steps"]),
    )
    matrix_batch_steps = settings.get("matrix_batch_steps", list(range(1, maximum + 1)))
    matrix_batches = [minibatches[step - 1] for step in matrix_batch_steps]
    reuse_actual = (
        len(actual) == len(states)
        and (study / "surfaces/minibatch-losses.npz").exists()
    )
    if reuse_actual:
        print("Reusing all actual checkpoint and minibatch losses", flush=True)
    else:
        actual = []
    for phase, step, path in [] if reuse_actual else states:
        print("Loading actual checkpoint:", phase, step, flush=True)
        load(path)
        row = {
            "phase": phase,
            "step": step,
            "probe_loss": loss(fixed),
            "heldout_loss": loss(heldout),
        }
        if phase == "attack":
            values = {}
            if step in matrix_steps:
                measured = batch_losses(matrix_batches)
                matrix.append(measured)
                values = dict(zip(matrix_batch_steps, measured))
            if step:
                row["last_batch_loss"] = (
                    values[step] if step in values else loss([minibatches[step - 1]])
                )
            if step < maximum:
                row["next_batch_loss"] = (
                    values[step + 1]
                    if step + 1 in values
                    else loss([minibatches[step]])
                )
        actual.append(row)
        print("Actual checkpoint:", row, flush=True)
    if not reuse_actual:
        save_rows("actual-losses.csv", actual)
        np.savez(
            study / "surfaces/minibatch-losses.npz",
            losses=np.array(matrix),
            steps=matrix_steps,
            batch_steps=matrix_batch_steps,
        )
    chosen_batches = [
        minibatches[step - 1] for step in settings["minibatch_surface_steps"]
    ]
    all_points = cached("fixed-grids.csv")
    projected = cached("projected-losses.csv")
    local_points = cached("local-grids.csv")
    radius = geom["radius"]
    local_plane = settings.get("local_plane", "attack")
    local_radius = settings.get("local_radius", min(radius, max(radius * 0.2, 1e-3)))
    local_coords = np.linspace(
        -local_radius, local_radius, settings["local_surface_points"]
    )

    def point_key(plane, c0, c1):
        return plane, round(float(c0), 10), round(float(c1), 10)

    completed_points = {
        point_key(row["plane"], row["c0"], row["c1"]) for row in all_points
    }
    completed_projections = {(row["plane"], row["step"]) for row in projected}
    completed_local = {
        (
            row["phase"],
            row["step"],
            round(float(row["c0"]), 10),
            round(float(row["c1"]), 10),
        )
        for row in local_points
    }
    for plane in settings["planes"]:
        plane_radius = settings.get("plane_radii", {}).get(plane, radius)
        coordinates = np.linspace(
            -plane_radius, plane_radius, settings["surface_points"]
        )
        grid_done = all(
            point_key(plane, c0, c1) in completed_points
            for c0 in coordinates
            for c1 in coordinates
        )
        projections_done = all(
            (plane, step) in completed_projections
            for step in [0] + settings["attack_steps"]
        )
        if plane != local_plane and grid_done and projections_done:
            print("Reusing completed plane:", plane, flush=True)
            continue
        print("Loading visualization basis:", plane, flush=True)
        mapped = np.load(study / "geometry" / f"{plane}-basis.npy", mmap_mode="r")
        basis = torch.from_numpy(mapped).to("cuda:0")
        for i, c0 in enumerate(coordinates):
            for c1 in coordinates:
                key = point_key(plane, c0, c1)
                if key in completed_points:
                    continue
                perturb(center, basis, c0, c1)
                row = {"plane": plane, "c0": c0, "c1": c1, "probe_loss": loss(fixed)}
                values = batch_losses(chosen_batches)
                row.update(
                    {
                        f"batch_{step}_loss": value
                        for step, value in zip(
                            settings["minibatch_surface_steps"], values
                        )
                    }
                )
                all_points.append(row)
                completed_points.add(key)
            save_rows("fixed-grids.csv", all_points)
            print("Surface:", plane, i + 1, "/", len(coordinates), flush=True)
        for point in [
            {
                "step": 0,
                "c0": 0.0,
                "c1": 0.0,
                "distance": 0.0,
                "captured_fraction": 1.0,
                "off_plane_distance": 0.0,
            }
        ] + geom["planes"][plane]:
            if (plane, point["step"]) in completed_projections:
                continue
            perturb(center, basis, point["c0"], point["c1"])
            projected.append({"plane": plane, **point, "projected_loss": loss(fixed)})
            completed_projections.add((plane, point["step"]))
            save_rows("projected-losses.csv", projected)
        # Local neighborhoods keep their orientation and units fixed while their
        # centers follow the actual weights, including movement outside the plane.
        if plane == local_plane:
            frames = [
                (
                    "defense",
                    step,
                    (
                        Path(settings["base_model"])
                        if step == 0
                        else defense_root / f"checkpoint-{step}"
                    ),
                )
                for step in settings["defense_frames"]
            ]
            frames += [
                (
                    "attack",
                    step,
                    (
                        reference
                        if step == 0
                        else study / "models/attack" / f"checkpoint-{step}"
                    ),
                )
                for step in settings["attack_frames"]
            ]
            for phase, step, path in frames:
                if all(
                    (phase, step, round(float(c0), 10), round(float(c1), 10))
                    in completed_local
                    for c0 in local_coords
                    for c1 in local_coords
                ):
                    print("Reusing local frame:", phase, step, flush=True)
                    continue
                print("Loading local frame:", phase, step, flush=True)
                load(path)
                anchor = {
                    name: parameter.detach().clone()
                    for name, parameter in parameters.items()
                }
                for c0 in local_coords:
                    for c1 in local_coords:
                        key = (phase, step, round(float(c0), 10), round(float(c1), 10))
                        if key in completed_local:
                            continue
                        perturb(anchor, basis, c0, c1)
                        row = {
                            "phase": phase,
                            "step": step,
                            "c0": c0,
                            "c1": c1,
                            "probe_loss": loss(fixed),
                        }
                        if phase == "attack":
                            index = min(step, maximum - 1)
                            row["minibatch_step"] = index + 1
                            row["minibatch_loss"] = loss([minibatches[index]])
                        local_points.append(row)
                        completed_local.add(key)
                    save_rows("local-grids.csv", local_points)
                    print("Local row:", phase, step, float(c0), flush=True)
                del anchor
                print("Local frame:", phase, step, flush=True)
        del basis, mapped
        torch.cuda.empty_cache()
        save_rows("fixed-grids.csv", all_points)
        save_rows("projected-losses.csv", projected)
        save_rows("local-grids.csv", local_points)
    for row in all_points + projected + local_points + actual:
        assert all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, (float, int))
        )
    assert len(all_points) == len(settings["planes"]) * settings["surface_points"] ** 2
    assert len(projected) == len(settings["planes"]) * (
        len(settings["attack_steps"]) + 1
    )
    assert (
        len(local_points)
        == (len(settings["defense_frames"]) + len(settings["attack_frames"]))
        * settings["local_surface_points"] ** 2
    )
    dump(
        study / "surfaces/validation.json",
        {
            "state": "passed",
            "fixed_grid_points": len(all_points),
            "moving_grid_points": len(local_points),
            "actual_checkpoints": len(actual),
            "projected_checkpoints": len(projected),
            "local_radius": local_radius,
            "radius": radius,
            "plane_radii": {
                plane: settings.get("plane_radii", {}).get(plane, radius)
                for plane in settings["planes"]
            },
            "local_plane": local_plane,
        },
    )


def minibatch_matrix_diagnostics(losses, checkpoint_steps, batch_steps, records):
    """Compare sampled batch positions and distinguish future from unseen content."""
    losses = np.asarray(losses)
    checkpoint_steps, batch_steps = list(checkpoint_steps), list(batch_steps)
    assert losses.shape == (len(checkpoint_steps), len(batch_steps))
    assert checkpoint_steps[0] == 0 and np.isfinite(losses).all()
    assert len(set(batch_steps)) == len(batch_steps)
    assert all(1 <= step <= len(records) for step in batch_steps)
    reductions = losses[0] - losses
    recovery, observations = [], []
    for i, step in enumerate(checkpoint_steps):
        used = np.array(batch_steps) <= step
        seen = {row["batch_sha256"] for row in records if row["step"] <= step}
        unseen = np.array(
            [
                records[position - 1]["batch_sha256"] not in seen
                for position in batch_steps
            ]
        )
        recovery.append(
            {
                "step": int(step),
                "mean_loss_reduction_all_batches": float(reductions[i].mean()),
                "mean_loss_reduction_already_used_batches": (
                    float(reductions[i, used].mean()) if used.any() else None
                ),
                "mean_loss_reduction_future_batches": (
                    float(reductions[i, ~used].mean()) if (~used).any() else None
                ),
                "mean_loss_reduction_unseen_content": (
                    float(reductions[i, unseen].mean()) if unseen.any() else None
                ),
                "sampled_batches": len(batch_steps),
                "sampled_future_positions": int((~used).sum()),
                "sampled_unseen_content": int(unseen.sum()),
            }
        )
    for row in records:
        for step, field in (
            (row["step"] - 1, "loss_before_update"),
            (row["step"], "loss_after_update"),
        ):
            if step in checkpoint_steps and row["step"] in batch_steps:
                value = float(
                    losses[checkpoint_steps.index(step), batch_steps.index(row["step"])]
                )
                observations.append(
                    {
                        "batch_step": row["step"],
                        "checkpoint_step": step,
                        "field": field,
                        "recorded_loss": row[field],
                        "recomputed_loss": value,
                        "absolute_difference": abs(row[field] - value),
                    }
                )
    return recovery, observations


def report(study, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter

    def expanded_limits(low, high):
        if low == high:
            padding = max(abs(float(low)), 1.0) * 1e-6
            return float(low - padding), float(high + padding)
        return float(low), float(high)

    geom = json.loads((study / "geometry/geometry.json").read_text())
    grids = pd.read_csv(study / "surfaces/fixed-grids.csv")
    actual = pd.read_csv(study / "surfaces/actual-losses.csv")
    projected = pd.read_csv(study / "surfaces/projected-losses.csv")
    local = pd.read_csv(study / "surfaces/local-grids.csv")
    center_checks = []
    for label, source in (("fixed", grids), ("moving", local)):
        centers = source[(source.c0 == 0) & (source.c1 == 0)].copy()
        if label == "fixed":
            reference = float(
                actual[(actual.phase == "attack") & (actual.step == 0)].probe_loss.iloc[
                    0
                ]
            )
            differences = abs(centers.probe_loss - reference)
        else:
            centers = centers.merge(
                actual[["phase", "step", "probe_loss"]],
                on=["phase", "step"],
                suffixes=("", "_actual"),
            )
            differences = abs(centers.probe_loss - centers.probe_loss_actual)
        center_checks.append(
            {
                "kind": label,
                "count": len(centers),
                "maximum_absolute_difference": float(differences.max()),
            }
        )
    broad_file = study / "surfaces/random-evolution.csv"
    if broad_file.exists():
        broad = pd.read_csv(broad_file)
        centers = broad[(broad.c0 == 0) & (broad.c1 == 0)].merge(
            actual[["phase", "step", "probe_loss"]],
            on=["phase", "step"],
            suffixes=("", "_actual"),
        )
        center_checks.append(
            {
                "kind": "broad-random",
                "count": len(centers),
                "maximum_absolute_difference": float(
                    abs(centers.probe_loss - centers.probe_loss_actual).max()
                ),
            }
        )
    assert all(
        item["count"] > 0 and item["maximum_absolute_difference"] < 1e-5
        for item in center_checks
    ), center_checks
    dump(study / "center-validation.json", center_checks)
    attack_actual = actual[actual.phase == "attack"].sort_values("step")
    projected = projected.merge(attack_actual[["step", "probe_loss"]], on="step")
    projected["loss_gap"] = projected.projected_loss - projected.probe_loss
    projected.to_csv(study / "projection-quality.csv", index=False)
    for name in settings["planes"]:
        rows = projected[(projected.plane == name) & (projected.step > 0)]
        print(
            name,
            "minimum capture",
            rows.captured_fraction.min(),
            "maximum loss gap",
            rows.loss_gap.abs().max(),
            flush=True,
        )
    low, high = expanded_limits(grids.probe_loss.min(), grids.probe_loss.max())
    number_planes = len(settings["planes"])
    fig, axes = plt.subplots(
        1,
        number_planes,
        figsize=(5.3 * number_planes, 4.5),
        layout="constrained",
        squeeze=False,
    )
    axes = axes.ravel()
    for plane, ax in zip(settings["planes"], axes):
        data = grids[grids.plane == plane].pivot(
            index="c1", columns="c0", values="probe_loss"
        )
        image = ax.contourf(
            data.columns,
            data.index,
            data.values,
            levels=np.linspace(low, high, 18),
            cmap="viridis",
        )
        route = projected[projected.plane == plane].sort_values("step")
        ax.plot(route.c0, route.c1, "o-", color="red", linewidth=1)
        plane_radius = max(abs(data.columns.min()), abs(data.columns.max()))
        near_origin = (
            max(route.c0.abs().max(), route.c1.abs().max()) < 0.02 * plane_radius
        )
        labels = settings.get("path_label_steps", [0, 1, 4, 16, 48, 128])
        if not near_origin:
            for row in route.itertuples():
                if row.step in labels:
                    ax.annotate(
                        str(row.step),
                        (row.c0, row.c1),
                        fontsize=8,
                        xytext={
                            0: (-10, 10),
                            1: (-15, -14),
                            4: (6, 4),
                            16: (5, 5),
                            48: (5, -12),
                            128: (5, 6),
                        }.get(row.step, (5, 5)),
                        textcoords="offset points",
                    )
        else:
            ax.annotate(
                "All projected checkpoints\nlie near the origin",
                (0, 0),
                xytext=(12, 28),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->"},
                fontsize=8,
            )
        ax.set(
            title={
                "random": "Random plane",
                "attack": "Attack-fitted plane",
                "transverse": "Plane perpendicular to attack span",
                "reference": "Shared baseline attack plane",
            }.get(plane, plane),
            xlabel="First coordinate (global L2 units)",
            ylabel="Second coordinate (global L2 units)",
            aspect="equal",
            xlim=(data.columns.min(), data.columns.max()),
            ylim=(data.index.min(), data.index.max()),
        )
        outside = (route.c0.abs() > plane_radius) | (route.c1.abs() > plane_radius)
        if outside.any():
            ax.text(
                0.02,
                0.02,
                f"{int(outside.sum())} checkpoints outside window",
                transform=ax.transAxes,
                fontsize=8,
            )
    fig.colorbar(image, ax=axes, label="Fixed-probe cross-entropy (nats)", shrink=0.8)
    fig.suptitle("Fixed slices through the attack start; red line is a projection")
    for ext in ("png", "pdf"):
        fig.savefig(study / f"attack-paths.{ext}", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(5.3 * number_planes, 5), layout="constrained")
    for i, plane in enumerate(settings["planes"]):
        ax = fig.add_subplot(1, number_planes, i + 1, projection="3d")
        data = grids[grids.plane == plane].pivot(
            index="c1", columns="c0", values="probe_loss"
        )
        x, y = np.meshgrid(data.columns, data.index)
        ax.plot_surface(
            x, y, data.values, cmap="viridis", vmin=low, vmax=high, alpha=0.8
        )
        route = projected[projected.plane == plane].sort_values("step")
        ax.plot(
            route.c0,
            route.c1,
            route.projected_loss,
            "r.--",
            label="Loss of projected weights",
        )
        ax.scatter(
            route.c0,
            route.c1,
            route.probe_loss,
            c=route.step,
            cmap="plasma",
            s=18,
            label="Actual checkpoint loss at projected x,y",
        )
        ax.set(
            title=plane,
            xlabel="Coordinate 1",
            ylabel="Coordinate 2",
            zlabel="Cross-entropy",
            zlim=(min(low, attack_actual.probe_loss.min()), high),
            xlim=(data.columns.min(), data.columns.max()),
            ylim=(data.index.min(), data.index.max()),
        )
    axes = fig.axes
    axes[-1].legend(fontsize=6, loc="upper left")
    fig.suptitle(
        "Actual checkpoint loss shown at projected x,y: points can lie off the surface"
    )
    fig.savefig(study / "surfaces-and-checkpoints.png", dpi=180)
    fig.savefig(study / "surfaces-and-checkpoints.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(
        attack_actual.step, attack_actual.probe_loss, "ko-", label="Actual: fixed probe"
    )
    axes[0].plot(
        attack_actual.step,
        attack_actual.heldout_loss,
        "k--",
        label="Actual: held-out documents",
    )
    for plane in settings["planes"]:
        rows = projected[projected.plane == plane].sort_values("step")
        axes[0].plot(rows.step, rows.projected_loss, ".-", label=f"Projected: {plane}")
        nonzero = rows[rows.step > 0]
        moving = nonzero[nonzero.distance > 0]
        axes[1].plot(moving.step, moving.captured_fraction, ".-", label=plane)
        axes[2].plot(rows.step, rows.off_plane_distance, ".-", label=plane)
    for ax, ylabel in zip(
        axes,
        (
            "Cross-entropy (nats)",
            "Fraction of displacement captured",
            "Distance outside plane (global L2)",
        ),
    ):
        ax.set(xlabel="Attack step", ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.savefig(study / "projection-fidelity.png", dpi=180)
    fig.savefig(study / "projection-fidelity.pdf")
    plt.close(fig)

    matrix = np.load(study / "surfaces/minibatch-losses.npz")
    matrix_batch_steps = (
        matrix["batch_steps"].tolist()
        if "batch_steps" in matrix
        else list(range(1, matrix["losses"].shape[1] + 1))
    )
    records = [
        json.loads(line)
        for line in (study / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), layout="constrained")
    image = axes[0].imshow(
        matrix["losses"],
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=(0.5, matrix["losses"].shape[1] + 0.5, -0.5, len(matrix["steps"]) - 0.5),
    )
    axes[0].set(
        yticks=range(len(matrix["steps"])),
        yticklabels=matrix["steps"],
        xlabel="Minibatch's position in actual attack order",
        ylabel="Model checkpoint step",
    )
    fig.colorbar(image, ax=axes[0], label="Minibatch cross-entropy")
    reductions = matrix["losses"][0] - matrix["losses"]
    image = axes[1].imshow(
        reductions,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=(0.5, reductions.shape[1] + 0.5, -0.5, len(matrix["steps"]) - 0.5),
    )
    axes[1].plot(
        np.searchsorted(matrix_batch_steps, matrix["steps"], side="right") + 0.5,
        np.arange(len(matrix["steps"])),
        "r--",
        lw=1,
        label="Boundary: batches already used / not yet used",
    )
    axes[1].set(
        yticks=range(len(matrix["steps"])),
        yticklabels=matrix["steps"],
        xlabel="Minibatch's position in actual attack order",
        ylabel="Model checkpoint step",
    )
    axes[1].legend(fontsize=6)
    if matrix_batch_steps != list(range(1, len(matrix_batch_steps) + 1)):
        tick_positions = np.unique(
            np.linspace(0, len(matrix_batch_steps) - 1, 7).astype(int)
        )
        for ax in axes[:2]:
            ax.set_xticks(
                tick_positions + 1, [matrix_batch_steps[i] for i in tick_positions]
            )
            ax.set_xlabel("Sampled minibatch's actual attack step")
    fig.colorbar(image, ax=axes[1], label="Loss reduction from attack start (nats)")
    axes[2].plot(
        [row["step"] for row in records],
        [row["loss_before_update"] for row in records],
        label="Actual batch before its update",
    )
    axes[2].plot(
        [row["step"] for row in records],
        [row["loss_after_update"] for row in records],
        label="Same batch after its update",
        alpha=0.8,
    )
    axes[2].set(xlabel="Attack step", ylabel="Cross-entropy (nats)")
    axes[2].legend(fontsize=8)
    fig.savefig(study / "minibatch-dynamics.png", dpi=180)
    fig.savefig(study / "minibatch-dynamics.pdf")
    plt.close(fig)

    # The full evaluations can be reused only because geometry verifies that all
    # shared checkpoints are exactly equal to the original reproduction weights.
    score_file = Path(
        settings.get(
            "capability_scores_file", Path(settings["original_run"]) / "comparison.csv"
        )
    )
    original = (
        pd.read_csv(score_file)
        if score_file.exists()
        else pd.DataFrame(
            columns=["learning_rate", "attack_step", "task", "jwk_percent"]
        )
    )
    original = original[
        np.isclose(
            pd.to_numeric(original.learning_rate),
            float(settings["attack_learning_rate"]),
            rtol=1e-8,
            atol=0,
        )
    ]
    early = study / "early-comparison.csv"
    if early.exists():
        original = pd.concat([original, pd.read_csv(early)], ignore_index=True)
    original.sort_values(["attack_step", "task"]).to_csv(
        study / "capability-scores.csv", index=False
    )
    route = pd.DataFrame([{"step": 0, "distance": 0}] + geom["planes"]["attack"])
    route = route[["step", "distance"]].merge(attack_actual, on="step")
    route.to_csv(study / "actual-trajectory.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    axes[0].plot(
        attack_actual.step, attack_actual.probe_loss, "o-", label="Fixed 10 documents"
    )
    axes[0].plot(
        attack_actual.step,
        attack_actual.heldout_loss,
        "s--",
        label="32 held-out documents",
    )
    axes[0].set(xlabel="Attack step", ylabel="Cross-entropy (nats)")
    axes[1].plot(route.distance, route.probe_loss, "o-", label="Fixed 10 documents")
    axes[1].plot(
        route.distance, route.heldout_loss, "s--", label="32 held-out documents"
    )
    for row in route.itertuples():
        if row.step in (0, 1, 16, 48, 128):
            axes[1].annotate(
                str(row.step),
                (row.distance, row.probe_loss),
                xytext=((-26, 5) if row.step == 128 else (4, 4)),
                textcoords="offset points",
                fontsize=8,
            )
    axes[1].set(
        xlabel="Distance from attack start (global L2)", ylabel="Cross-entropy (nats)"
    )
    for task, label in (
        ("wmdp_bio_robust", "WMDP Bio Robust"),
        ("mmlu_no_bio", "MMLU excluding biology"),
    ):
        rows = original[original.task == task].sort_values("attack_step")
        axes[2].plot(rows.attack_step, rows.jwk_percent, "o-", label=label)
    axes[2].set(xlabel="Attack step", ylabel="Accuracy (%)")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Actual checkpoints: loss and capability recovery")
    for ext in ("png", "pdf"):
        fig.savefig(study / f"loss-and-capability.{ext}", dpi=180)
    plt.close(fig)

    batch_recovery, observation_checks = minibatch_matrix_diagnostics(
        matrix["losses"], matrix["steps"].tolist(), matrix_batch_steps, records
    )
    dump(
        study / "observation-validation.json",
        {
            "comparisons": observation_checks,
            "maximum_absolute_difference": max(
                (row["absolute_difference"] for row in observation_checks), default=None
            ),
            "note": "All use eager attention. Training spans four GPUs with batch size one; analysis groups four minibatches per forward on one GPU in a separate locked evaluation environment. Small numerical differences are possible.",
        },
    )

    pd.DataFrame(batch_recovery).to_csv(study / "minibatch-recovery.csv", index=False)

    objectives = ["probe_loss"] + [
        f"batch_{step}_loss" for step in settings["minibatch_surface_steps"]
    ]
    rows = grids[grids.plane == "attack"]
    fig, axes = plt.subplots(
        1, len(objectives), figsize=(5 * len(objectives), 4), layout="constrained"
    )
    limits = expanded_limits(rows[objectives].min().min(), rows[objectives].max().max())
    for objective, ax in zip(objectives, axes):
        data = rows.pivot(index="c1", columns="c0", values=objective)
        image = ax.contourf(
            data.columns,
            data.index,
            data.values,
            levels=np.linspace(*limits, 18),
            cmap="viridis",
        )
        route = projected[projected.plane == "attack"].sort_values("step")
        ax.plot(route.c0, route.c1, "r.-")
        ax.set(
            title=objective,
            aspect="equal",
            xlabel="Coordinate 1",
            ylabel="Coordinate 2",
        )
    fig.colorbar(image, ax=axes, label="Cross-entropy (nats)", shrink=0.8)
    fig.savefig(study / "minibatch-landscapes.png", dpi=180)
    fig.savefig(study / "minibatch-landscapes.pdf")
    plt.close(fig)

    selected_steps = settings.get("minibatch_comparison_steps", [0, 16, 48, 128])
    selected = local[(local.phase == "attack") & local.step.isin(selected_steps)]
    common_low, common_high = expanded_limits(
        selected[["probe_loss", "minibatch_loss"]].min().min(),
        selected[["probe_loss", "minibatch_loss"]].max().max(),
    )
    fig, axes = plt.subplots(
        len(selected_steps),
        2,
        figsize=(10, 4 * len(selected_steps)),
        layout="constrained",
    )
    for i, step in enumerate(selected_steps):
        rows = selected[selected.step == step]
        for objective, ax in zip(("probe_loss", "minibatch_loss"), axes[i]):
            data = rows.pivot(index="c1", columns="c0", values=objective)
            image = ax.contourf(
                data.columns,
                data.index,
                data.values,
                levels=np.linspace(common_low, common_high, 18),
                cmap="viridis",
            )
            ax.scatter([0], [0], marker="x", color="red")
            title = (
                "Fixed probe"
                if objective == "probe_loss"
                else f"Actual attack minibatch {int(rows.minibatch_step.iloc[0])}"
            )
            ax.set(
                title=f"Model step {step}: {title}",
                xlabel="Local coordinate 1",
                ylabel="Local coordinate 2",
                aspect="equal",
            )
    fig.colorbar(
        image, ax=axes.ravel().tolist(), label="Cross-entropy (nats)", shrink=0.6
    )
    fig.savefig(study / "local-minibatch-landscapes.png", dpi=160)
    fig.savefig(study / "local-minibatch-landscapes.pdf")
    plt.close(fig)

    frames = list(local.groupby(["phase", "step"], sort=False))
    low_local, high_local = expanded_limits(
        local.probe_loss.min(), local.probe_loss.max()
    )
    local_label = settings.get("local_plane_label", "attack-fitted directions")
    local_half_width = settings.get("local_radius", geom["radius"] * 0.2)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")

    def draw(index):
        ax.clear()
        (phase, step), rows = frames[index]
        data = rows.pivot(index="c1", columns="c0", values="probe_loss")
        x, y = np.meshgrid(data.columns, data.index)
        ax.plot_surface(
            x, y, data.values, cmap="viridis", vmin=low_local, vmax=high_local
        )
        ax.set(
            title=f"{phase.capitalize()} step {step} | {local_label}\nLocal half-width {local_half_width:.3f} global L2 units",
            xlabel="Local coordinate 1",
            ylabel="Local coordinate 2",
            zlabel="Fixed-probe CE",
            zlim=(low_local, high_local),
        )
        return []

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000)
    animation.save(study / "local-landscape-evolution.gif", writer=PillowWriter(fps=1))
    plt.close(fig)
    for phase in ("defense", "attack"):
        selected = [(step, rows) for (name, step), rows in frames if name == phase]
        fig, axes = plt.subplots(
            1,
            len(selected),
            figsize=(4 * len(selected), 4.5),
            subplot_kw={"projection": "3d"},
        )
        for (step, rows), ax in zip(selected, np.atleast_1d(axes)):
            data = rows.pivot(index="c1", columns="c0", values="probe_loss")
            x, y = np.meshgrid(data.columns, data.index)
            ax.plot_surface(
                x, y, data.values, cmap="viridis", vmin=low_local, vmax=high_local
            )
            ax.set(
                title=f"{phase} {step}",
                zlim=(low_local, high_local),
                xlabel="Local 1",
                ylabel="Local 2",
                zlabel="CE",
            )
        fig.suptitle(
            f"{phase.capitalize()} | {local_label} | local half-width {local_half_width:.3f} global L2 units"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(
            study / f"{phase}-landscape-evolution.png", dpi=160, bbox_inches="tight"
        )
        fig.savefig(study / f"{phase}-landscape-evolution.pdf", bbox_inches="tight")
        plt.close(fig)

    if broad_file.exists():
        import shutil

        archive = study / "renderings-v2"
        archive.mkdir(exist_ok=True)
        for ext in ("png", "pdf"):
            filename = f"random-landscape-evolution.{ext}"
            if (study / filename).exists() and not (archive / filename).exists():
                shutil.copy2(study / filename, archive / filename)
        broad_low, broad_high = broad.probe_loss.min(), broad.probe_loss.max()
        frame_count = len(broad.groupby(["phase", "step"], sort=False))
        columns = min(4, frame_count)
        rows_count = math.ceil(frame_count / columns)
        fig, axes = plt.subplots(
            rows_count,
            columns,
            figsize=(4.5 * columns, 4.5 * rows_count),
            subplot_kw={"projection": "3d"},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.02, right=0.9, bottom=0.04, top=0.9, wspace=0.17, hspace=0.12
        )
        for ((phase, step), rows), ax in zip(
            broad.groupby(["phase", "step"], sort=False), axes.flat
        ):
            grid = rows.pivot(index="c1", columns="c0", values="probe_loss")
            x, y = np.meshgrid(grid.columns, grid.index)
            surface = ax.plot_surface(
                x, y, grid.values, cmap="viridis", vmin=broad_low, vmax=broad_high
            )
            ax.set(
                title=f"{phase.capitalize()} step {step}", zlim=(broad_low, broad_high)
            )
            ax.set_xlabel("Random 1", fontsize=9, labelpad=0)
            ax.set_ylabel("Random 2", fontsize=9, labelpad=0)
            ax.set_zlabel("CE (nats)", fontsize=9, labelpad=0)
            ax.tick_params(labelsize=8, pad=0)
            ax.title.set_fontsize(11)
        for ax in axes.flat[frame_count:]:
            ax.set_axis_off()
        fig.suptitle(
            f"Random directions centered on each checkpoint | coordinate half-width {settings.get('broad_radius', 125):g} (global L2 units)",
            fontsize=13,
        )
        color_axis = fig.add_axes([0.94, 0.18, 0.015, 0.62])
        fig.colorbar(surface, cax=color_axis, label="Fixed-probe cross-entropy (nats)")
        for ext in ("png", "pdf"):
            fig.savefig(study / f"random-landscape-evolution.{ext}", dpi=180)
        plt.close(fig)
    summary = {
        "attack_rank": geom.get("attack_rank"),
        "attack_plane_energy_fraction": geom["attack_plane_energy_fraction"],
        "radius": geom["radius"],
        "random_transverse_overlap": geom["random_transverse_overlap"],
        "rerun_distance_from_original": geom["rerun_distance_from_original"],
        "planes": {
            name: {
                "minimum_capture": (
                    float(
                        projected[
                            (projected.plane == name)
                            & (projected.step > 0)
                            & (projected.distance > 0)
                        ].captured_fraction.min()
                    )
                    if (
                        (projected.plane == name)
                        & (projected.step > 0)
                        & (projected.distance > 0)
                    ).any()
                    else None
                ),
                "maximum_projection_loss_gap": float(
                    projected[projected.plane == name].loss_gap.abs().max()
                ),
            }
            for name in settings["planes"]
        },
        "actual_attack_start_loss": float(attack_actual.probe_loss.iloc[0]),
        "actual_attack_end_loss": float(attack_actual.probe_loss.iloc[-1]),
        "local_plane": settings.get("local_plane", "attack"),
        "local_radius": local_half_width,
        "matrix_batch_steps": matrix_batch_steps,
    }
    dump(study / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def random_frames(study, settings):
    """Follow fortress formation on the same broad random slice at every frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from matplotlib.animation import FuncAnimation, PillowWriter
    from safetensors import safe_open
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )
    from datasets import load_dataset

    geom = json.loads((study / "geometry/geometry.json").read_text())
    reference = Path(settings.get("attack_anchor", study / "models/defense"))
    defense_root = Path(
        settings.get("defense_checkpoint_root", study / "models/defense")
    )
    radius = settings.get("broad_radius", 125)
    resolution = settings.get("broad_points", 9)
    coordinates = np.linspace(-radius, radius, resolution)
    model = (
        AutoModelForCausalLM.from_pretrained(
            reference, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to("cuda:0")
        .eval()
    )
    parameters = dict(model.named_parameters())
    tokenizer = AutoTokenizer.from_pretrained(reference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    helper = runpy.run_path(str(study / "source/misc/reference_get_ll.py"))
    fixed = [
        {key: value.to("cuda:0") for key, value in batch.items()}
        for batch in helper["get_dataloader"](
            "cais/wmdp-bio-forget-corpus",
            tokenizer,
            512,
            2,
            settings["probe_documents"],
        )
    ]
    if settings.get("fixed_source_indices") is not None:
        corpus = load_dataset(
            "cais/wmdp-bio-forget-corpus",
            split="train",
            revision=settings.get("forget_revision"),
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        fixed_rows = [
            tokenizer(corpus[index]["text"], max_length=512, truncation=True)
            for index in settings["fixed_source_indices"]
        ]
        fixed = [
            {
                key: value.to("cuda:0")
                for key, value in collator(fixed_rows[i : i + 2]).items()
            }
            for i in range(0, len(fixed_rows), 2)
        ]
    mapped = np.load(study / "geometry/random-basis.npy", mmap_mode="r")
    basis = torch.from_numpy(mapped).to("cuda:0")
    frames = [
        (
            "defense",
            step,
            (
                Path(settings["base_model"])
                if step == 0
                else defense_root / f"checkpoint-{step}"
            ),
        )
        for step in settings.get("broad_defense_frames", [0, 8, 32, 125])
    ]
    frames += [
        (
            "attack",
            step,
            reference if step == 0 else study / "models/attack" / f"checkpoint-{step}",
        )
        for step in settings.get("broad_attack_frames", [0, 16, 48, 128])
    ]
    target = study / "surfaces/random-evolution.csv"
    rows = (
        pd.read_csv(target, float_precision="round_trip").to_dict("records")
        if target.exists()
        else []
    )
    completed = {
        (r["phase"], r["step"], round(float(r["c0"]), 10), round(float(r["c1"]), 10))
        for r in rows
    }
    with torch.inference_mode():
        for phase, step, path in frames:
            if all(
                (phase, step, round(float(c0), 10), round(float(c1), 10)) in completed
                for c0 in coordinates
                for c1 in coordinates
            ):
                print("Reusing broad random frame:", phase, step, flush=True)
                continue
            print("Loading broad random frame:", phase, step, flush=True)
            index = path / "model.safetensors.index.json"
            mapping = (
                json.loads(index.read_text())["weight_map"]
                if index.exists()
                else {name: "model.safetensors" for name in parameters}
            )
            with contextlib.ExitStack() as stack:
                handles = {
                    filename: stack.enter_context(
                        safe_open(path / filename, framework="pt", device="cpu")
                    )
                    for filename in set(mapping.values())
                }
                for name, parameter in parameters.items():
                    parameter.copy_(handles[mapping[name]].get_tensor(name).clone())
            anchor = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            for c0 in coordinates:
                for c1 in coordinates:
                    key = (phase, step, round(float(c0), 10), round(float(c1), 10))
                    if key in completed:
                        continue
                    for name, parameter in parameters.items():
                        offset = geom["offsets"][name]
                        count = parameter.numel()
                        first = basis[0, offset : offset + count].reshape(
                            parameter.shape
                        )
                        second = basis[1, offset : offset + count].reshape(
                            parameter.shape
                        )
                        parameter.copy_(anchor[name].float() + c0 * first + c1 * second)
                    value = float(
                        np.mean([float(model(**batch).loss) for batch in fixed])
                    )
                    assert math.isfinite(value)
                    rows.append(
                        {
                            "phase": phase,
                            "step": step,
                            "c0": c0,
                            "c1": c1,
                            "probe_loss": value,
                        }
                    )
                    completed.add(key)
                temporary = target.with_suffix(".csv.tmp")
                pd.DataFrame(rows).to_csv(temporary, index=False)
                temporary.replace(target)
                print("Broad row:", phase, step, float(c0), flush=True)
            del anchor
            print("Broad random frame:", phase, step, flush=True)
    data = pd.DataFrame(rows)
    assert len(rows) == len(frames) * resolution**2
    data.to_csv(study / "surfaces/random-evolution.csv", index=False)
    low, high = data.probe_loss.min(), data.probe_loss.max()
    if low == high:
        padding = max(abs(float(low)), 1) * 1e-6
        low, high = low - padding, high + padding
    columns = min(4, len(frames))
    rows_count = math.ceil(len(frames) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(4.5 * columns, 4.5 * rows_count),
        subplot_kw={"projection": "3d"},
        layout="constrained",
        squeeze=False,
    )
    for (phase, step, _), ax in zip(frames, axes.flat):
        grid = data[(data.phase == phase) & (data.step == step)].pivot(
            index="c1", columns="c0", values="probe_loss"
        )
        x, y = np.meshgrid(grid.columns, grid.index)
        ax.plot_surface(x, y, grid.values, cmap="viridis", vmin=low, vmax=high)
        ax.set(
            title=f"{phase} step {step} | radius {radius:g}",
            xlabel="Random coordinate 1",
            ylabel="Random coordinate 2",
            zlabel="Fixed-probe CE",
            zlim=(low, high),
        )
    for ax in axes.flat[len(frames) :]:
        ax.set_axis_off()
    fig.savefig(study / "random-landscape-evolution.png", dpi=180)
    fig.savefig(study / "random-landscape-evolution.pdf")
    plt.close(fig)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")

    def draw(i):
        phase, step, _ = frames[i]
        ax.clear()
        grid = data[(data.phase == phase) & (data.step == step)].pivot(
            index="c1", columns="c0", values="probe_loss"
        )
        x, y = np.meshgrid(grid.columns, grid.index)
        ax.plot_surface(x, y, grid.values, cmap="viridis", vmin=low, vmax=high)
        ax.set(
            title=f"{phase.capitalize()} {step} | fixed random axes | radius {radius:g}",
            xlabel="Random 1",
            ylabel="Random 2",
            zlabel="CE",
            zlim=(low, high),
        )
        return []

    FuncAnimation(fig, draw, frames=len(frames), interval=1000).save(
        study / "random-landscape-evolution.gif", writer=PillowWriter(fps=1)
    )
    plt.close(fig)
    dump(
        study / "surfaces/random-evolution-validation.json",
        {
            "state": "passed",
            "points": len(rows),
            "frames": len(frames),
            "radius": radius,
            "shared_zlim": [float(low), float(high)],
        },
    )


def main():
    import faulthandler
    import signal

    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(300, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("geometry", "surfaces", "report", "random-frames")
    )
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    settings = json.loads((args.study / "study.json").read_text())
    {
        "geometry": geometry,
        "surfaces": surfaces,
        "report": report,
        "random-frames": random_frames,
    }[args.stage](args.study, settings)


if __name__ == "__main__":
    main()
