"""Native-FP32 weight slices anchored at each clean-room final defender.

Run this payload through launch_tuo.py. No attack checkpoint is loaded as a
second language model: CPU safetensor readers stream parameter displacements.
Random axes are shared across cases by seed and parameter order. Fitted axes
use an uncentered Gram-matrix SVD of actual saved attack displacements.
Projected attack states are evaluated separately and never drawn as a path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path

# Match the clean-room worker before importing the tensor runtime.
os.environ.setdefault("TORCH_BLAS_PREFER_CUBLASLT", "0")
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

CHUNK = 1_000_000


class CheckpointReader:
    """CPU-mapped safetensors; handles stay alive until the reader closes."""

    def __init__(self, folder, *, allow_anchor_cast=False):
        from safetensors import safe_open

        self.folder = Path(folder)
        self.allow_anchor_cast = allow_anchor_cast
        self.source_dtypes = {}
        self.stack = ExitStack()
        self.handles = {}
        index = self.folder / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            files = sorted(self.folder.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No safetensors in {self.folder}")
            self.weight_map = {}
            for filename in files:
                handle = self.stack.enter_context(
                    safe_open(filename, framework="pt", device="cpu")
                )
                self.handles[filename.name] = handle
                for name in handle.keys():
                    if name in self.weight_map:
                        raise ValueError(f"Duplicate checkpoint tensor {name}")
                    self.weight_map[name] = filename.name
        self.safe_open = safe_open

    def get(self, name):
        filename = self.weight_map[name]
        if filename not in self.handles:
            self.handles[filename] = self.stack.enter_context(
                self.safe_open(self.folder / filename, framework="pt", device="cpu")
            )
        value = self.handles[filename].get_tensor(name)
        self.source_dtypes[name] = str(value.dtype)
        if value.dtype != torch.float32:
            if self.allow_anchor_cast and value.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                return value.float()
            raise ValueError(f"Native FP32 required: {name} is {value.dtype}")
        return value

    def close(self):
        self.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def dot(left, right):
    """CPU FP64 scalar reduction with bounded temporary arrays."""
    if left.keys() != right.keys():
        raise ValueError("Direction keys do not match")
    terms = []
    for name, value in left.items():
        other = right[name]
        if value.device.type != "cpu" or other.device.type != "cpu":
            raise ValueError("Directions must remain on CPU")
        if value.shape != other.shape:
            raise ValueError(f"Direction shape mismatch for {name}")
        a, b = value.reshape(-1), other.reshape(-1)
        for start in range(0, a.numel(), CHUNK):
            terms.append(
                float(
                    torch.dot(
                        a[start : start + CHUNK].double(),
                        b[start : start + CHUNK].double(),
                    )
                )
            )
    return math.fsum(terms)


def normalize_(direction, target=1.0):
    norm = math.sqrt(dot(direction, direction))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize an empty or zero direction")
    for value in direction.values():
        value.mul_(target / norm)
    return norm


def random_basis(specs, seed=42):
    """Globally orthonormal FP32 Gaussians, identical for the same schema."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    axes = tuple(
        {
            name: torch.randn(shape, generator=generator, dtype=torch.float32)
            for name, shape in specs
        }
        for _ in range(2)
    )
    normalize_(axes[0])
    projection = dot(axes[0], axes[1])
    for name, value in axes[1].items():
        value.add_(axes[0][name], alpha=-projection)
    normalize_(axes[1])
    return axes


def displacement(reader, anchor, specs):
    return {name: reader.get(name) - anchor.get(name) for name, _ in specs}


def first_update_basis(first, shared_random):
    """Reuse/mutate the Gaussian; x=1 is the actual first attack update."""
    length = math.sqrt(dot(first, first))
    if length <= 0:
        raise ValueError("Actual first attack update has zero displacement")
    projection = dot(first, shared_random) / length**2
    for name, value in shared_random.items():
        value.add_(first[name], alpha=-projection)
    normalize_(shared_random, target=length)
    return (first, shared_random), length


def trajectory_gram(readers, anchor, specs):
    """Uncentered Gram matrix, without retaining full attack displacements."""
    gram = torch.zeros(len(readers), len(readers), dtype=torch.float64)
    for name, _ in specs:
        base = anchor.get(name)
        deltas = [(reader.get(name) - base).reshape(-1) for reader in readers]
        for start in range(0, base.numel(), CHUNK):
            block = torch.stack(
                [value[start : start + CHUNK] for value in deltas]
            ).double()
            gram.add_(block @ block.T)
    return gram


def fitted_coefficients(gram):
    """Coefficients for two axes scaled by the largest trajectory radius."""
    values, vectors = torch.linalg.eigh(gram.double())
    values, vectors = values.flip(0), vectors.flip(1)
    if len(values) < 2 or values[1] <= max(float(values[0]), 1.0) * 1e-12:
        raise ValueError("Two nonzero fitted directions require trajectory rank >= 2")
    scale = math.sqrt(float(gram.diag().max()))
    coefficients = vectors[:, :2] / values[:2].sqrt()[None, :] * scale
    return coefficients, scale, values


def fitted_basis(readers, anchor, specs, gram):
    coefficients, scale, eigenvalues = fitted_coefficients(gram)
    axes = ({}, {})
    for name, _ in specs:
        base = anchor.get(name)
        deltas = [reader.get(name) - base for reader in readers]
        for axis in range(2):
            value = torch.zeros_like(base)
            for i, delta in enumerate(deltas):
                value.add_(delta, alpha=float(coefficients[i, axis]))
            axes[axis][name] = value
    return axes, {
        "scale_l2": scale,
        "gram": gram.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "coefficients": coefficients.tolist(),
        "top_two_displacement_energy_fraction": float(
            eigenvalues[:2].sum() / eigenvalues.clamp_min(0).sum()
        ),
    }


def projection_coordinates(readers, anchor, specs, axes):
    """Actual least-squares coordinates/capture for the stored FP32 axes."""
    basis_gram = torch.tensor(
        [[dot(a, b) for b in axes] for a in axes], dtype=torch.float64
    )
    products = torch.zeros(len(readers), 2, dtype=torch.float64)
    norms = torch.zeros(len(readers), dtype=torch.float64)
    for name, _ in specs:
        base = anchor.get(name)
        local_axes = [{name: axis[name]} for axis in axes]
        for i, reader in enumerate(readers):
            delta = {name: reader.get(name) - base}
            norms[i] += dot(delta, delta)
            for j in range(2):
                products[i, j] += dot(delta, local_axes[j])
    coordinates = torch.linalg.solve(basis_gram, products.T).T
    captures = (coordinates * products).sum(-1) / norms.clamp_min(1e-300)
    return coordinates, captures, basis_gram


@torch.no_grad()
def set_point(model, anchor, axes=None, coordinates=(0.0, 0.0)):
    """Reset from saved native weights for every point; never accumulate drift."""
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("Model parameters must remain FP32")
        parameter.copy_(anchor.get(name))
        if axes is not None:
            for axis, coordinate in zip(axes, coordinates):
                if coordinate:
                    parameter.add_(
                        axis[name].to(parameter.device), alpha=float(coordinate)
                    )


def checkpoint_identity(folder):
    """Content hashes recorded once; compute-node I/O, not a model reload."""
    result = {}
    for path in sorted(Path(folder).glob("*.safetensors*")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    return result


def load_cleanroom(run):
    path = run / "source/tar_cleanroom.py"
    spec = importlib.util.spec_from_file_location(
        "frozen_tar_cleanroom_geometry_source", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot(result, folder, ce_limits=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(result["planes"])
    all_ce = [
        point[group]["ce"]
        for plane in result["planes"].values()
        for point in plane["grid"]
        for group in ("forget_train", "forget_heldout_subset")
    ]
    vmin, vmax = ce_limits if ce_limits is not None else (min(all_ce), max(all_ce))
    fig, plots = plt.subplots(
        2,
        len(names),
        figsize=(5 * len(names), 8),
        squeeze=False,
        constrained_layout=True,
    )
    for column, name in enumerate(names):
        plane = result["planes"][name]
        grid = plane["coordinates"]
        for row, group in enumerate(("forget_train", "forget_heldout_subset")):
            values = np.array([point[group]["ce"] for point in plane["grid"]]).reshape(
                len(grid), len(grid)
            )
            ax = plots[row, column]
            mesh = ax.pcolormesh(
                grid,
                grid,
                values,
                shading="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            ax.scatter([0], [0], marker="+", c="white", s=60)
            if name == "first":
                ax.scatter(
                    [1], [0], marker="x", c="red", s=45, label="Actual first update"
                )
                ax.legend(loc="best", fontsize=8)
            ax.set(
                title=f"{name}: {group}",
                xlabel=plane["axis_labels"][0],
                ylabel=plane["axis_labels"][1],
            )
    fig.colorbar(
        mesh, ax=plots.ravel().tolist(), label="Native FP32 next-token cross-entropy"
    )
    fig.suptitle(
        f"{result['case']}: all planes anchored at final defender; one shared CE scale"
    )
    fig.savefig(folder / "slices.png", dpi=180)
    fig.savefig(folder / "slices.pdf")
    plt.close(fig)
    return {"ce_min": vmin, "ce_max": vmax}


def run_geometry(args):
    run = args.run.resolve()
    clean = load_cleanroom(run)
    study = clean.load_study(run)
    if args.case != "control" and args.case not in study["cases"]:
        raise ValueError(f"Unknown case {args.case}")
    data = clean.prepare_data(run, study)
    folder = run / "geometry" / args.case
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(run / "cache/matplotlib"))
    target = folder / "geometry.json"
    if target.exists():
        raise FileExistsError(f"Refuse to overwrite {target}")
    base_folder = (
        Path(study["anchor"])
        if args.case == "control"
        else run / "cases" / args.case / "defense-125"
    )
    attack_folder = run / "cases" / args.case / "attacks/full-same"
    checkpoint_folders = [
        attack_folder / f"checkpoint-{step}" for step in args.trajectory_steps
    ]
    if "first" in args.planes and 1 not in args.trajectory_steps:
        raise ValueError("First-update plane requires trajectory step 1")
    if "fit" in args.planes and len(args.trajectory_steps) < 2:
        raise ValueError("Fitted plane requires at least two saved states")
    model, _ = clean.runtime(study, base_folder)
    specs = tuple(
        (name, tuple(value.shape)) for name, value in model.named_parameters()
    )
    probes = {
        "forget_train": data["forget_train"],
        "forget_heldout_subset": data["forget_heldout"][:8],
    }

    def evaluate():
        return {group: clean.evaluate(model, rows) for group, rows in probes.items()}

    result = {
        "schema": 1,
        "case": args.case,
        "anchor": str(base_folder),
        "weight_dtype": "float32",
        "forward_dtype": "float32",
        "scalar_reductions_dtype": "float64 on CPU",
        "grid_size": args.grid_size,
        "seed": args.seed,
        "trajectory_steps": args.trajectory_steps,
        "study_sha256": hashlib.sha256((run / "study.json").read_bytes()).hexdigest(),
        "data_content_sha256": data["content_sha256"],
        "geometry_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "cleanroom_source_sha256": hashlib.sha256(
            (run / "source/tar_cleanroom.py").read_bytes()
        ).hexdigest(),
        "parameter_schema_sha256": clean.digest(specs),
        "probes": {
            group: [
                {
                    "source_row": row["source_row"],
                    "source_text_sha256": row["source_text_sha256"],
                    "token_sha256": clean.digest(row["input_ids"]),
                }
                for row in rows
            ]
            for group, rows in probes.items()
        },
        "checkpoint_files": {},
        "actual_saved_states": [],
        "planes": {},
        "interpretation": "Grid values use native FP32 weights at the stated plane coordinates. Saved-state projections are reconstructions, not the actual attack trajectory; no trajectory curve is plotted.",
    }
    with ExitStack() as stack:
        anchor = stack.enter_context(
            CheckpointReader(base_folder, allow_anchor_cast=args.case == "control")
        )
        readers = [
            stack.enter_context(CheckpointReader(path)) for path in checkpoint_folders
        ]
        result["checkpoint_files"]["anchor"] = checkpoint_identity(base_folder)
        for step, path, reader in zip(
            args.trajectory_steps, checkpoint_folders, readers
        ):
            result["checkpoint_files"][str(step)] = checkpoint_identity(path)
            set_point(model, reader)
            result["actual_saved_states"].append({"step": step, "probes": evaluate()})
        set_point(model, anchor)
        result["anchor_source_dtypes"] = sorted(set(anchor.source_dtypes.values()))
        result["anchor_load_policy"] = (
            "Original stored anchor values expanded to FP32, exactly as runtime(); lost precision is not recovered"
            if args.case == "control"
            else "Native FP32 defender checkpoint required"
        )
        result["anchor_probes"] = evaluate()
        shared = None
        for name in args.planes:
            if name == "random":
                shared = random_basis(specs, args.seed)
                axes = shared
                extent, labels, extra = (
                    125.0,
                    ["Shared Gaussian axis 1 (L2)", "Shared Gaussian axis 2 (L2)"],
                    {},
                )
            elif name == "first":
                gaussian = (
                    shared[0]
                    if shared is not None
                    else random_basis(specs, args.seed)[0]
                )
                shared = None
                axes = None
                first = displacement(
                    readers[args.trajectory_steps.index(1)], anchor, specs
                )
                axes, scale = first_update_basis(first, gaussian)
                extent, labels, extra = (
                    1.0,
                    [
                        "Actual first update (multiples)",
                        "Orthogonal shared Gaussian (same L2)",
                    ],
                    {"scale_l2": scale},
                )
            else:
                shared, axes = None, None
                gram = trajectory_gram(readers, anchor, specs)
                axes, extra = fitted_basis(readers, anchor, specs, gram)
                extent, labels = 1.0, [
                    "Uncentered SVD axis 1 (scaled radius)",
                    "Uncentered SVD axis 2 (scaled radius)",
                ]
            coordinates, captures, basis_gram = projection_coordinates(
                readers, anchor, specs, axes
            )
            plane = {
                "extent": [-extent, extent],
                "axis_labels": labels,
                "basis_gram": basis_gram.tolist(),
                "projections": [],
                "coordinates": torch.linspace(
                    -extent, extent, args.grid_size, dtype=torch.float64
                ).tolist(),
                "grid": [],
                **extra,
            }
            for i, step in enumerate(args.trajectory_steps):
                xy = coordinates[i].tolist()
                set_point(model, anchor, axes, xy)
                projected = evaluate()
                actual = result["actual_saved_states"][i]["probes"]
                plane["projections"].append(
                    {
                        "step": step,
                        "coordinates": xy,
                        "displacement_energy_capture": float(captures[i]),
                        "probes": projected,
                        "ce_error_projected_minus_actual": {
                            group: projected[group]["ce"] - actual[group]["ce"]
                            for group in probes
                        },
                    }
                )
            result["planes"][name] = plane
            for y in plane["coordinates"]:
                for x in plane["coordinates"]:
                    set_point(model, anchor, axes, (x, y))
                    point = {"x": x, "y": y, **evaluate()}
                    plane["grid"].append(point)
                    clean.dump(folder / "geometry-progress.json", result)
                    print(
                        json.dumps(
                            {
                                "case": args.case,
                                "plane": name,
                                "x": x,
                                "y": y,
                                "train_ce": point["forget_train"]["ce"],
                                "heldout_ce": point["forget_heldout_subset"]["ce"],
                            }
                        ),
                        flush=True,
                    )
            axes = None
            if name == "first":
                first, gaussian = None, None
        set_point(model, anchor)
    try:
        result["figure_ce_scale"] = plot(result, folder, args.ce_limits)
    except ModuleNotFoundError as error:
        if error.name not in {"matplotlib", "numpy"}:
            raise
        result["figure_state"] = (
            f"Numerical grids complete; --plot-only requires {error.name}"
        )
        print(result["figure_state"], flush=True)
    clean.dump(target, result)


def plot_saved(args):
    """Render complete numerical results in an existing plotting environment."""
    folder = args.run.resolve() / "geometry" / args.case
    path = folder / "geometry.json"
    result = json.loads(path.read_text())
    os.environ.setdefault("MPLCONFIGDIR", str(args.run.resolve() / "cache/matplotlib"))
    result["figure_ce_scale"] = plot(result, folder, args.ce_limits)
    result["figure_state"] = "complete"
    result["plot_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    load_cleanroom(args.run.resolve()).dump(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--planes",
        nargs="+",
        choices=("random", "first", "fit"),
        default=["random", "first", "fit"],
    )
    parser.add_argument(
        "--trajectory-steps",
        nargs="+",
        type=int,
        choices=(1, 4, 16, 64),
        default=[1, 4, 16, 64],
    )
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ce-limits", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.grid_size < 3 or args.grid_size % 2 != 1:
        parser.error("--grid-size must be odd and at least 3")
    if args.ce_limits is not None and (
        not all(math.isfinite(value) for value in args.ce_limits)
        or args.ce_limits[0] >= args.ce_limits[1]
    ):
        parser.error("--ce-limits requires finite MIN < MAX")
    if len(set(args.planes)) != len(args.planes) or len(
        set(args.trajectory_steps)
    ) != len(args.trajectory_steps):
        parser.error("planes and trajectory steps must be unique")
    # Always process random first so its first axis can be reused without extra
    # persistent bases. This order is part of the recorded experiment metadata.
    args.planes = [name for name in ("random", "first", "fit") if name in args.planes]
    if args.plot_only:
        plot_saved(args)
    else:
        run_geometry(args)


if __name__ == "__main__":
    main()
