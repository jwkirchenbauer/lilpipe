"""Measure baseline attack paths and fixed/moving loss slices on Tuolumne."""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import runpy
from pathlib import Path

import numpy as np


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def plane_coefficients(gram, n_attack):
    """Orthonormal random, attack-SVD, and attack-orthogonal planes."""
    gram = np.asarray(gram, dtype=np.float64)
    attack = gram[:n_attack, :n_attack]
    values, vectors = np.linalg.eigh(attack)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    valid = values > max(values[0] * 1e-11, 1e-14)
    if np.sum(valid) < 2:
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

    random = np.zeros((n_attack + 2, 2))
    random[n_attack:, :] = np.eye(2)
    aligned = np.zeros_like(random)
    aligned[:n_attack] = vectors[:, :2] / np.sqrt(values[:2])
    transverse = random.copy()
    transverse[:n_attack] = -inverse @ gram[:n_attack, n_attack:]
    planes = {
        "random": normalize(random),
        "attack": normalize(aligned),
        "transverse": normalize(transverse),
    }
    np.testing.assert_allclose(
        planes["transverse"].T @ gram[:, :n_attack], 0, atol=1e-7
    )
    return planes, {
        "attack_rank": int(np.sum(valid)),
        "attack_eigenvalues": values.tolist(),
        "attack_plane_energy_fraction": float(values[:2].sum() / values.sum()),
    }


def paths(study, settings):
    return [
        study / "models/attack" / f"checkpoint-{step}"
        for step in settings["attack_steps"]
    ]


def geometry(study, settings):
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = study / "models/defense"
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
    helper = runpy.run_path(str(study / "source/misc/reference_get_ll.py"))
    directions = helper["get_directions"](model, settings["direction_seed"])
    random = [
        {name: value.cpu() for name, value in direction.items()}
        for direction in directions
    ]
    del directions, model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoints = paths(study, settings)
    n = len(checkpoints)
    chunk = 4 * 1024 * 1024
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
    gram_gpu = torch.zeros((n + 2, n + 2), dtype=torch.float64, device=device)
    offsets, total = {}, 0
    for name in names:
        offsets[name] = total
        total += math.prod(shapes[name])

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

        def blocks(dtype):
            for name in names:
                center = base.get_tensor(name).reshape(-1)
                tensors = [handle.get_tensor(name).reshape(-1) for handle in handles]
                for start in range(0, center.numel(), chunk):
                    stop = min(start + chunk, center.numel())
                    anchor = center[start:stop].to(device=device, dtype=dtype)
                    deltas = [
                        tensor[start:stop].to(device=device, dtype=dtype) - anchor
                        for tensor in tensors
                    ]
                    vectors = deltas + [
                        direction[name]
                        .reshape(-1)[start:stop]
                        .to(device=device, dtype=dtype)
                        for direction in random
                    ]
                    yield name, start, stop, anchor, torch.stack(vectors)

        print("Computing FP64 trajectory Gram matrix", flush=True)
        last = None
        for name, start, stop, anchor, vectors in blocks(torch.float64):
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
        gram = gram_gpu.cpu().numpy()
        coefficients, details = plane_coefficients(gram, n)
        plane_order = list(coefficients)
        combined = np.concatenate([coefficients[name] for name in plane_order], axis=1)
        combined_gpu = torch.tensor(combined.T, dtype=torch.float64, device=device)
        files = {
            name: np.lib.format.open_memmap(
                study / "geometry" / f"{name}-basis.npy",
                mode="w+",
                dtype=np.float32,
                shape=(2, total),
            )
            for name in plane_order
        }
        actual_gram = torch.zeros((6, 6), dtype=torch.float64, device=device)
        actual_cross = torch.zeros((n, 6), dtype=torch.float64, device=device)
        print("Constructing and saving three visualization planes", flush=True)
        for name, start, stop, anchor, vectors in blocks(torch.float64):
            basis = (combined_gpu @ vectors).float()
            accurate = basis.double()
            actual_gram.add_(accurate @ accurate.T)
            actual_cross.add_(vectors[:n] @ accurate.T)
            host = basis.cpu().numpy()
            offset = offsets[name] + start
            for i, plane in enumerate(plane_order):
                files[plane][:, offset : offset + stop - start] = host[
                    2 * i : 2 * i + 2
                ]
        for file in files.values():
            file.flush()
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
        "radius": float(1.25 * np.sqrt(norm_sq.max())),
        "rerun_distance_from_original": {
            name: math.sqrt(value) for name, value in differences.items()
        },
        "random_transverse_overlap": float(np.sum(actual_gram[:2, 4:6] ** 2) / 2),
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
    reference = study / "models/defense"
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
    records = [
        json.loads(line)
        for line in (study / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [record["step"] for record in records] == list(range(1, 129))
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
    corpus = load_dataset("cais/wmdp-bio-forget-corpus", split="train")
    heldout_rows, heldout_hashes = [], []
    for index in range(1000, len(corpus)):
        text = corpus[index]["text"]
        if not isinstance(text, str) or len(text) <= 50:
            continue
        tokens = tokenizer(text, max_length=512, truncation=True)
        if tuple(tokens["input_ids"]) in seen:
            continue
        heldout_rows.append(tokens)
        heldout_hashes.append(hashlib.sha256(text.encode()).hexdigest())
        if len(heldout_rows) == settings["heldout_documents"]:
            break
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
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
            "heldout_sha256": heldout_hashes,
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
                parameter.copy_(handles[mapping[name]].get_tensor(name))

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
        ("defense", step, study / "models/defense" / f"checkpoint-{step}")
        for step in settings["defense_steps"]
    ]
    states += [("attack", 0, reference)]
    states += [
        ("attack", step, study / "models/attack" / f"checkpoint-{step}")
        for step in settings["attack_steps"]
    ]
    actual, matrix = [], []
    for phase, step, path in states:
        load(path)
        row = {
            "phase": phase,
            "step": step,
            "probe_loss": loss(fixed),
            "heldout_loss": loss(heldout),
        }
        if phase == "attack":
            values = batch_losses(minibatches)
            matrix.append(values)
            if step:
                row["last_batch_loss"] = values[step - 1]
            if step < 128:
                row["next_batch_loss"] = values[step]
        actual.append(row)
        print("Actual checkpoint:", row, flush=True)
    pd.DataFrame(actual).to_csv(study / "surfaces/actual-losses.csv", index=False)
    np.savez(
        study / "surfaces/minibatch-losses.npz",
        losses=np.array(matrix),
        steps=[0] + settings["attack_steps"],
    )
    chosen_batches = [
        minibatches[step - 1] for step in settings["minibatch_surface_steps"]
    ]
    all_points, projected, local_points = [], [], []
    radius = geom["radius"]
    coordinates = np.linspace(-radius, radius, settings["surface_points"])
    local_radius = min(radius, max(radius * 0.2, 1e-3))
    local_coords = np.linspace(
        -local_radius, local_radius, settings["local_surface_points"]
    )
    for plane in settings["planes"]:
        print("Loading visualization basis:", plane, flush=True)
        mapped = np.load(study / "geometry" / f"{plane}-basis.npy", mmap_mode="r")
        basis = torch.from_numpy(mapped).to("cuda:0")
        for i, c0 in enumerate(coordinates):
            for c1 in coordinates:
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
            perturb(center, basis, point["c0"], point["c1"])
            projected.append({"plane": plane, **point, "projected_loss": loss(fixed)})
        # Local neighborhoods keep their orientation and units fixed while their
        # centers follow the actual weights, including movement outside the plane.
        if plane == "attack":
            frames = [
                (
                    "defense",
                    step,
                    (
                        Path(settings["base_model"])
                        if step == 0
                        else study / "models/defense" / f"checkpoint-{step}"
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
                load(path)
                anchor = {
                    name: parameter.detach().clone()
                    for name, parameter in parameters.items()
                }
                for c0 in local_coords:
                    for c1 in local_coords:
                        perturb(anchor, basis, c0, c1)
                        row = {
                            "phase": phase,
                            "step": step,
                            "c0": c0,
                            "c1": c1,
                            "probe_loss": loss(fixed),
                        }
                        if phase == "attack":
                            index = min(step, 127)
                            row["minibatch_step"] = index + 1
                            row["minibatch_loss"] = loss([minibatches[index]])
                        local_points.append(row)
                del anchor
                print("Local frame:", phase, step, flush=True)
        del basis, mapped
        torch.cuda.empty_cache()
        pd.DataFrame(all_points).to_csv(study / "surfaces/fixed-grids.csv", index=False)
        pd.DataFrame(projected).to_csv(
            study / "surfaces/projected-losses.csv", index=False
        )
        pd.DataFrame(local_points).to_csv(
            study / "surfaces/local-grids.csv", index=False
        )
    for row in all_points + projected + local_points + actual:
        assert all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, (float, int))
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
        },
    )


def report(study, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.animation import FuncAnimation, PillowWriter

    geom = json.loads((study / "geometry/geometry.json").read_text())
    grids = pd.read_csv(study / "surfaces/fixed-grids.csv")
    actual = pd.read_csv(study / "surfaces/actual-losses.csv")
    projected = pd.read_csv(study / "surfaces/projected-losses.csv")
    local = pd.read_csv(study / "surfaces/local-grids.csv")
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
    low, high = grids.probe_loss.min(), grids.probe_loss.max()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
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
        for row in route.itertuples():
            ax.annotate(str(row.step), (row.c0, row.c1), fontsize=7)
        ax.set(
            title={
                "random": "Random plane",
                "attack": "Attack-fitted plane",
                "transverse": "Plane perpendicular to attack span",
            }[plane],
            xlabel="First coordinate (global L2 units)",
            ylabel="Second coordinate (global L2 units)",
            aspect="equal",
        )
    fig.colorbar(image, ax=axes, label="Fixed-probe cross-entropy (nats)", shrink=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(study / f"attack-paths.{ext}", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 5), layout="constrained")
    for i, plane in enumerate(settings["planes"]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
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
        axes[1].plot(rows.step, rows.captured_fraction, ".-", label=plane)
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
    records = [
        json.loads(line)
        for line in (study / "models/attack/observed-batches.jsonl")
        .read_text()
        .splitlines()
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    image = axes[0].imshow(
        matrix["losses"], aspect="auto", origin="lower", cmap="viridis"
    )
    axes[0].set(
        yticks=range(len(matrix["steps"])),
        yticklabels=matrix["steps"],
        xlabel="Minibatch's position in actual attack order",
        ylabel="Model checkpoint step",
    )
    fig.colorbar(image, ax=axes[0], label="Minibatch cross-entropy")
    axes[1].plot(
        [row["step"] for row in records],
        [row["loss_before_update"] for row in records],
        label="Actual batch before its update",
    )
    axes[1].plot(
        [row["step"] for row in records],
        [row["loss_after_update"] for row in records],
        label="Same batch after its update",
        alpha=0.8,
    )
    axes[1].set(xlabel="Attack step", ylabel="Cross-entropy (nats)")
    axes[1].legend(fontsize=8)
    fig.savefig(study / "minibatch-dynamics.png", dpi=180)
    fig.savefig(study / "minibatch-dynamics.pdf")
    plt.close(fig)

    objectives = ["probe_loss"] + [
        f"batch_{step}_loss" for step in settings["minibatch_surface_steps"]
    ]
    rows = grids[grids.plane == "attack"]
    fig, axes = plt.subplots(
        1, len(objectives), figsize=(5 * len(objectives), 4), layout="constrained"
    )
    limits = (float(rows[objectives].min().min()), float(rows[objectives].max().max()))
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

    selected_steps = [0, 16, 48, 128]
    selected = local[(local.phase == "attack") & local.step.isin(selected_steps)]
    common_low = selected[["probe_loss", "minibatch_loss"]].min().min()
    common_high = selected[["probe_loss", "minibatch_loss"]].max().max()
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
    low_local, high_local = local.probe_loss.min(), local.probe_loss.max()
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
            title=f"{phase.capitalize()} step {step} | same attack-fitted directions",
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
            figsize=(4 * len(selected), 4),
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
        fig.tight_layout()
        fig.savefig(study / f"{phase}-landscape-evolution.png", dpi=160)
        fig.savefig(study / f"{phase}-landscape-evolution.pdf")
        plt.close(fig)
    summary = {
        "attack_plane_energy_fraction": geom["attack_plane_energy_fraction"],
        "radius": geom["radius"],
        "random_transverse_overlap": geom["random_transverse_overlap"],
        "rerun_distance_from_original": geom["rerun_distance_from_original"],
        "planes": {
            name: {
                "minimum_capture": float(
                    projected[
                        (projected.plane == name) & (projected.step > 0)
                    ].captured_fraction.min()
                ),
                "maximum_projection_loss_gap": float(
                    projected[projected.plane == name].loss_gap.abs().max()
                ),
            }
            for name in settings["planes"]
        },
        "actual_attack_start_loss": float(attack_actual.probe_loss.iloc[0]),
        "actual_attack_end_loss": float(attack_actual.probe_loss.iloc[-1]),
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    geom = json.loads((study / "geometry/geometry.json").read_text())
    reference = study / "models/defense"
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
    mapped = np.load(study / "geometry/random-basis.npy", mmap_mode="r")
    basis = torch.from_numpy(mapped).to("cuda:0")
    frames = [
        (
            "defense",
            step,
            (
                Path(settings["base_model"])
                if step == 0
                else study / "models/defense" / f"checkpoint-{step}"
            ),
        )
        for step in (0, 8, 32, 125)
    ]
    frames += [
        (
            "attack",
            step,
            reference if step == 0 else study / "models/attack" / f"checkpoint-{step}",
        )
        for step in (0, 16, 48, 128)
    ]
    rows = []
    with torch.inference_mode():
        for phase, step, path in frames:
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
                    parameter.copy_(handles[mapping[name]].get_tensor(name))
            anchor = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            for c0 in np.linspace(-125, 125, 9):
                for c1 in np.linspace(-125, 125, 9):
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
            del anchor
            print("Broad random frame:", phase, step, flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(study / "surfaces/random-evolution.csv", index=False)
    low, high = data.probe_loss.min(), data.probe_loss.max()
    fig, axes = plt.subplots(
        2, 4, figsize=(18, 9), subplot_kw={"projection": "3d"}, layout="constrained"
    )
    for (phase, step, _), ax in zip(frames, axes.flat):
        grid = data[(data.phase == phase) & (data.step == step)].pivot(
            index="c1", columns="c0", values="probe_loss"
        )
        x, y = np.meshgrid(grid.columns, grid.index)
        ax.plot_surface(x, y, grid.values, cmap="viridis", vmin=low, vmax=high)
        ax.set(
            title=f"{phase} step {step} | radius 125",
            xlabel="Random coordinate 1",
            ylabel="Random coordinate 2",
            zlabel="Fixed-probe CE",
            zlim=(low, high),
        )
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
            title=f"{phase.capitalize()} {step} | fixed random axes | radius 125",
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
            "radius": 125,
            "shared_zlim": [float(low), float(high)],
        },
    )


def main():
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
