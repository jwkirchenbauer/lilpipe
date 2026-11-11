"""Native-FP32 weight slices anchored at each clean-room final defender.

Run this payload through launch_tuo.py. No attack checkpoint is loaded as a
second language model: CPU safetensor readers stream parameter displacements.
Random axes are shared across cases by seed and parameter order. Fitted axes
use an uncentered Gram-matrix SVD of actual saved attack displacements.
Projected attack states are evaluated separately and never drawn as a path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import time
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
REFERENCE_STEPS = (1, 4, 16, 64)
PLANE_ORDER = ("random", "random-step", "reference", "first", "fit")


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


def first_update_basis(first, shared_random, length=None):
    """Reuse/mutate the Gaussian; x=1 is the actual first attack update."""
    length = math.sqrt(dot(first, first)) if length is None else length
    if length <= 0:
        raise ValueError("Actual first attack update has zero displacement")
    projection = dot(first, shared_random) / length**2
    for name, value in shared_random.items():
        value.add_(first[name], alpha=-projection)
    normalize_(shared_random, target=length)
    return (first, shared_random), length


def displacement_norm(reader, anchor, specs):
    """Global L2 norm of a native first update, with bounded CPU temporaries."""
    terms = []
    for name, _ in specs:
        value = (reader.get(name) - anchor.get(name)).reshape(-1)
        for start in range(0, value.numel(), CHUNK):
            chunk = value[start : start + CHUNK].double()
            terms.append(float(torch.dot(chunk, chunk)))
    norm = math.sqrt(math.fsum(terms))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Actual first attack update must have positive finite L2 norm")
    return norm


def matched_radius_random_basis(pair, radius):
    """Scale the same Gaussian pair; preserve axis 1 unscaled for the first plane.

    Consumes the second input axis in place to bound CPU storage at three
    directions. Positive scaling preserves signs and orientation of both axes.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("First-update radius must be positive and finite")
    first = {name: value * radius for name, value in pair[0].items()}
    for value in pair[1].values():
        value.mul_(radius)
    return (first, pair[1]), pair[0]


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
    axes = basis_from_coefficients(readers, anchor, specs, coefficients)
    return axes, {
        "scale_l2": scale,
        "gram": gram.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "coefficients": coefficients.tolist(),
        "top_two_displacement_energy_fraction": float(
            eigenvalues[:2].sum() / eigenvalues.clamp_min(0).sum()
        ),
    }


def basis_from_coefficients(readers, anchor, specs, coefficients):
    """Reconstruct exactly the same FP32 physical axes from fixed coefficients."""
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64)
    if (
        coefficients.shape != (len(readers), 2)
        or not torch.isfinite(coefficients).all()
    ):
        raise ValueError(
            "Expected finite trajectory coefficients with shape [states, 2]"
        )
    axes = ({}, {})
    for name, _ in specs:
        base = anchor.get(name)
        deltas = [reader.get(name) - base for reader in readers]
        for axis in range(2):
            value = torch.zeros_like(base)
            for i, delta in enumerate(deltas):
                value.add_(delta, alpha=float(coefficients[i, axis]))
            axes[axis][name] = value
    return axes


def reference_definition(control, source_path):
    """A fixed coordinate definition, independent of the target defender."""
    if (
        control["case"] != "control"
        or tuple(control["trajectory_steps"]) != REFERENCE_STEPS
    ):
        raise ValueError("Reference requires control full-same states 1, 4, 16, 64")
    fit = control["planes"]["fit"]
    coefficients = torch.as_tensor(fit["coefficients"], dtype=torch.float64)
    if (
        coefficients.shape != (4, 2)
        or not torch.isfinite(coefficients).all()
        or not math.isfinite(fit["scale_l2"])
        or fit["scale_l2"] <= 0
    ):
        raise ValueError("Invalid stored control fit coefficients or scale")
    keys = ("anchor", *(str(step) for step in REFERENCE_STEPS))
    payload = {
        "attack_variant": "full-same",
        "control_anchor": control["anchor"],
        "trajectory_steps": list(REFERENCE_STEPS),
        "parameter_schema_sha256": control["parameter_schema_sha256"],
        "data_content_sha256": control["data_content_sha256"],
        "checkpoint_files": {key: control["checkpoint_files"][key] for key in keys},
        **{
            key: fit[key]
            for key in (
                "coefficients",
                "scale_l2",
                "gram",
                "eigenvalues",
                "top_two_displacement_energy_fraction",
            )
        },
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    source_path = Path(source_path)
    return {
        **payload,
        "identity_sha256": identity,
        "source_geometry": str(source_path.resolve()),
        "source_geometry_sha256": (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            if source_path.is_file()
            else None
        ),
        "coordinate_policy": "Exact control fitted physical axes and scale, translated to the target defender without rotation or rescaling",
    }


def control_reference_axes(run, definition, specs):
    """Read only the control trajectory; target weights never define the axes."""
    folders = [Path(definition["control_anchor"])] + [
        run / "cases/control/attacks/full-same" / f"checkpoint-{step}"
        for step in REFERENCE_STEPS
    ]
    with ExitStack() as stack:
        for key, folder in zip(
            ("anchor", *(str(step) for step in REFERENCE_STEPS)), folders
        ):
            if checkpoint_identity(folder) != definition["checkpoint_files"][key]:
                raise ValueError(f"Control reference checkpoint changed: {folder}")
        anchor = stack.enter_context(
            CheckpointReader(folders[0], allow_anchor_cast=True)
        )
        readers = [
            stack.enter_context(CheckpointReader(folder)) for folder in folders[1:]
        ]
        return basis_from_coefficients(
            readers, anchor, specs, definition["coefficients"]
        )


def archive_augmentation(target, result, clean):
    """Keep the previous JSON bytes and every original numerical plane."""
    previous = json.loads(target.read_text())
    old_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    archive = target.with_name(f"geometry-before-augmentation-{old_hash}.json")
    if not archive.exists():
        shutil.copy2(target, archive)
    for name, plane in previous["planes"].items():
        if result["planes"].get(name) != plane:
            raise ValueError(f"Plane augmentation changed existing plane {name}")
    result.setdefault("augmentations", []).append(
        {
            "added_planes": [
                name for name in result["planes"] if name not in previous["planes"]
            ],
            "previous_json": str(archive),
            "previous_sha256": old_hash,
            "previous_geometry_source_sha256": previous["geometry_source_sha256"],
            "augmentation_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            **helper_source_provenance(target.parents[2], clean),
        }
    )
    result["planes"] = {
        name: result["planes"][name] for name in PLANE_ORDER if name in result["planes"]
    }
    result.update(helper_source_provenance(target.parents[2], clean))
    clean.dump(target, result)


def add_control_reference(target, args, clean, study, data):
    """The control reference equals its fit, so reuse all measured numbers."""
    result = json.loads(target.read_text())
    if "reference" in result["planes"]:
        raise ValueError("Reference plane already exists")
    if (
        result["anchor"] != study["anchor"]
        or result["data_content_sha256"] != data["content_sha256"]
    ):
        raise ValueError("Existing control anchor/data differ from the current study")
    fit = result["planes"]["fit"]
    if (
        result["grid_size"] != args.grid_size
        or len(fit["grid"]) != args.grid_size**2
        or fit["extent"] != [-1.0, 1.0]
    ):
        raise ValueError("Control fit must be complete at the requested grid size")
    reference = copy.deepcopy(fit)
    reference["axis_labels"] = [
        "Fixed control attack axis 1",
        "Fixed control attack axis 2",
    ]
    reference["coordinate_scope"] = "shared physical axes and scale across cases"
    reference["reference_definition"] = reference_definition(result, target)
    reference["reused_from"] = {
        "plane": "fit",
        "reason": "Control reference and fitted axes/scale are identical; no new model evaluation",
    }
    result["schema"] = 3
    result["planes"]["reference"] = reference
    result["figure_state"] = (
        "Reference added from existing control fit; --plot-only refreshes the figure"
    )
    archive_augmentation(target, result, clean)


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


class DevicePointCache:
    """Immutable device copies of the anchor and one plane's FP32 axes."""

    @torch.no_grad()
    def __init__(self, model, anchor, axes):
        self.model = model
        self.parameters = tuple(model.named_parameters())
        self.anchor = {}
        self.axes = ({}, {})
        for name, parameter in self.parameters:
            base = anchor.get(name)
            values = (base, axes[0][name], axes[1][name])
            if parameter.dtype != torch.float32 or any(
                value.dtype != torch.float32 or value.shape != parameter.shape
                for value in values
            ):
                raise ValueError(f"FP32 cache shape/dtype mismatch: {name}")
            self.anchor[name] = base.to(parameter.device, copy=True)
            for index, axis in enumerate(axes):
                self.axes[index][name] = axis[name].to(parameter.device, copy=True)
        self.resident_bytes = sum(
            value.numel() * value.element_size()
            for mapping in (self.anchor, *self.axes)
            for value in mapping.values()
        )

    @torch.no_grad()
    def set(self, coordinates):
        for name, parameter in self.parameters:
            parameter.copy_(self.anchor[name])
            for axis, coordinate in zip(self.axes, coordinates):
                if coordinate:
                    # These are precisely the original separate in-place adds.
                    # No foreach, compiled graph, or fused combination is used.
                    parameter.add_(axis[name], alpha=float(coordinate))

    def close(self):
        self.anchor.clear()
        for axis in self.axes:
            axis.clear()


@torch.no_grad()
def validate_point_cache(cache, anchor, axes, coordinates, evaluate=None):
    """Compare real cached parameters against the original streamed formula.

    Only one parameter's reference/temporary arrays reside on a device at once.
    Optional CE checks run both full constructions; parameter equality is always
    checked, including the zero-displacement anchor and actual first-step axis.
    """
    records = []
    for xy in coordinates:
        baseline = None
        if evaluate is not None:
            set_point(cache.model, anchor, axes, xy)
            baseline = evaluate()
        cache.set(xy)
        maxima, error_terms, displacement_terms = {}, {}, {}
        for name, parameter in cache.parameters:
            reference = anchor.get(name).to(parameter.device, copy=True)
            for axis, coordinate in zip(axes, xy):
                if coordinate:
                    reference.add_(
                        axis[name].to(parameter.device), alpha=float(coordinate)
                    )
            difference = parameter - reference
            device = parameter.device
            maximum = difference.abs().max()
            error = difference.double().square().sum()
            displacement = (reference - cache.anchor[name]).double().square().sum()
            maxima[device] = (
                maximum
                if device not in maxima
                else torch.maximum(maxima[device], maximum)
            )
            error_terms[device] = (
                error if device not in error_terms else error_terms[device] + error
            )
            displacement_terms[device] = (
                displacement
                if device not in displacement_terms
                else displacement_terms[device] + displacement
            )
        maximum = max(float(value) for value in maxima.values())
        error = math.fsum(float(value) for value in error_terms.values())
        displacement = math.fsum(float(value) for value in displacement_terms.values())
        relative = math.sqrt(error / max(displacement, 1e-300))
        gaps = None
        if baseline is not None:
            observed = evaluate()
            gaps = {
                group: abs(observed[group]["ce"] - baseline[group]["ce"])
                for group in baseline
            }
        record = {
            "coordinates": list(xy),
            "parameter_max_abs_difference": maximum,
            "perturbation_relative_l2_difference": relative,
            "reference_perturbation_l2": math.sqrt(displacement),
            "ce_absolute_gaps": gaps,
        }
        records.append(record)
        if (
            not math.isfinite(maximum)
            or not math.isfinite(relative)
            or maximum != 0
            or (
                gaps is not None
                and any(not math.isfinite(gap) or gap != 0 for gap in gaps.values())
            )
        ):
            raise RuntimeError(f"Device point cache failed exact FP32 parity: {record}")
    return records


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
    # Use the helper shipped in this execution profile, including extensions.
    path = Path(__file__).with_name("tar_cleanroom.py")
    spec = importlib.util.spec_from_file_location(
        "frozen_tar_cleanroom_geometry_source", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper_source_provenance(run, clean):
    path = Path(clean.__file__).resolve()
    original = run / "source/tar_cleanroom.py"
    result = {
        "executing_cleanroom_source": str(path),
        "cleanroom_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "original_cleanroom_source": str(original.resolve()),
        "original_cleanroom_source_sha256": (
            hashlib.sha256(original.read_bytes()).hexdigest()
            if original.is_file()
            else None
        ),
    }
    extension = run / getattr(clean, "EXTENSIONS_FILE", "case-extensions.json")
    if extension.is_file():
        result["case_extensions_file"] = str(extension.resolve())
        result["case_extensions_sha256"] = hashlib.sha256(
            extension.read_bytes()
        ).hexdigest()
    return result


def plot(result, folder, ce_limits=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(result["planes"])
    plane_titles = {
        "random": "Broad random",
        "random-step": "Random at first-update radius",
        "reference": "Fixed control attack plane",
        "first": "First-update plane",
        "fit": "Fitted attack plane",
    }
    cohort_titles = {
        "forget_train": "Forget training examples",
        "forget_heldout_subset": "Held-out forget examples (8)",
    }

    def panel_title(name, plane, group):
        details = ""
        radius = plane.get("scale_l2")
        if name == "random":
            details = f"Shared unit axes; displayed ±{max(abs(value) for value in plane['coordinates']):.4g} L2"
        elif radius is not None:
            scope = (
                "Shared orientation; case radius"
                if name == "random-step"
                else (
                    "Shared control axes; fixed radius"
                    if name == "reference"
                    else "Case-specific axes; radius"
                )
            )
            details = f"{scope} = {radius:.4g} L2"
        return "\n".join(
            value
            for value in (plane_titles.get(name, name), cohort_titles[group], details)
            if value
        )

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
                title=panel_title(name, plane, group),
                xlabel=plane["axis_labels"][0],
                ylabel=plane["axis_labels"][1],
            )
            ax.title.set_fontsize(9)
    fig.colorbar(
        mesh, ax=plots.ravel().tolist(), label="Native FP32 next-token cross-entropy"
    )
    fig.suptitle(
        f"{result['case']}: all planes anchored at final defender; one shared CE scale"
    )
    fig.savefig(folder / "slices.png", dpi=180)
    fig.savefig(folder / "slices.pdf")
    plt.close(fig)

    # Use the identical measured arrays, CE normalization, and height range.
    # Facets connect grid nodes; they are not additional model evaluations.
    # 3D text extents are not fully accounted for by constrained_layout.
    # Reserve explicit row separation and a generous bottom margin instead.
    surface_fig = plt.figure(figsize=(5 * len(names), 14), constrained_layout=False)
    surface_fig.subplots_adjust(
        left=0.025, right=0.91, bottom=0.10, top=0.91, hspace=0.45, wspace=0.10
    )
    for column, name in enumerate(names):
        plane = result["planes"][name]
        grid = plane["coordinates"]
        xx, yy = np.meshgrid(grid, grid)
        for row, group in enumerate(("forget_train", "forget_heldout_subset")):
            values = np.array([point[group]["ce"] for point in plane["grid"]]).reshape(
                len(grid), len(grid)
            )
            axis = surface_fig.add_subplot(
                2,
                len(names),
                row * len(names) + column + 1,
                projection="3d",
                computed_zorder=False,
            )
            surface = axis.plot_surface(
                xx,
                yy,
                values,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0.3,
                edgecolor=(0, 0, 0, 0.2),
                antialiased=True,
                zorder=0,
            )
            axis.scatter(
                xx.ravel(),
                yy.ravel(),
                values.ravel(),
                c="black",
                s=3,
                depthshade=False,
                zorder=1,
            )
            measured = {
                (point["x"], point["y"]): point[group]["ce"] for point in plane["grid"]
            }
            if (0, 0) in measured:
                axis.scatter(
                    [0],
                    [0],
                    [measured[(0, 0)]],
                    facecolors="white",
                    edgecolors="black",
                    linewidths=1.0,
                    marker="*",
                    s=90,
                    depthshade=False,
                    label="Defender anchor",
                    zorder=10,
                )
            if name == "first" and (1, 0) in measured:
                axis.scatter(
                    [1],
                    [0],
                    [measured[(1, 0)]],
                    c="red",
                    marker="x",
                    s=50,
                    depthshade=False,
                    label="First-update point",
                    linewidths=1.6,
                    zorder=11,
                )
                axis.legend(loc="upper left", fontsize=7).set_zorder(20)
            if name == "random":
                labels_3d = ("Direction 1 (L2)", "Direction 2 (L2)")
            elif name == "first":
                labels_3d = (
                    "First update\n(multiples)",
                    "Orthogonal random\n(multiples)",
                )
            else:
                labels_3d = ("Direction 1 (radii)", "Direction 2 (radii)")
            axis.set(
                title=panel_title(name, plane, group),
                xlabel=labels_3d[0],
                ylabel=labels_3d[1],
                zlabel="CE (nats)",
                xlim=(grid[0], grid[-1]),
                ylim=(grid[0], grid[-1]),
                zlim=(vmin, vmax),
            )
            axis.title.set_fontsize(9)
            axis.view_init(elev=30, azim=-135)
            axis.tick_params(labelsize=7)
            axis.xaxis.label.set_size(8)
            axis.yaxis.label.set_size(8)
            axis.zaxis.label.set_size(8)
            axis.xaxis.labelpad = 2
            axis.yaxis.labelpad = 2
            axis.zaxis.labelpad = 2
    surface_fig.colorbar(
        surface,
        cax=surface_fig.add_axes([0.94, 0.17, 0.014, 0.66]),
        label="Native FP32 next-token cross-entropy; same scale as heatmap",
    )
    surface_fig.suptitle(
        f"{result['case']}: measured grid nodes and connecting facets; no trajectory shown"
    )
    surface_fig.savefig(
        folder / "slices-3d.png", dpi=180, bbox_inches="tight", pad_inches=0.2
    )
    surface_fig.savefig(folder / "slices-3d.pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close(surface_fig)

    if {"random-step", "first"} <= result["planes"].keys():
        random_radius = result["planes"]["random-step"]["scale_l2"]
        first_radius = result["planes"]["first"]["scale_l2"]
        if (
            not math.isfinite(first_radius)
            or first_radius <= 0
            or not math.isclose(random_radius, first_radius, rel_tol=1e-10)
        ):
            raise ValueError(
                "Matched-radius sections require equal positive first-update radii"
            )
        section_fig, section_axes = plt.subplots(
            2, 1, figsize=(8, 7), sharex=True, sharey=True, constrained_layout=True
        )
        for axis, group in zip(section_axes, ("forget_train", "forget_heldout_subset")):
            for name, label, color in (
                ("random-step", "Shared Gaussian direction", "#2878b5"),
                ("first", "First-update direction", "#d62728"),
            ):
                points = sorted(
                    (
                        point
                        for point in result["planes"][name]["grid"]
                        if point["y"] == 0
                    ),
                    key=lambda point: point["x"],
                )
                axis.plot(
                    [point["x"] * first_radius for point in points],
                    [point[group]["ce"] for point in points],
                    marker="o",
                    markersize=4,
                    color=color,
                    label=label,
                )
                if name == "first":
                    for point in points:
                        if point["x"] in (0, 1):
                            axis.scatter(
                                [point["x"] * first_radius],
                                [point[group]["ce"]],
                                marker="*" if point["x"] == 0 else "x",
                                c="black" if point["x"] == 0 else "red",
                                s=70,
                            )
            axis.set(title=cohort_titles[group], ylabel="CE (nats)", ylim=(vmin, vmax))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        section_axes[-1].set_xlabel(
            "Signed displacement along axis 1 (global parameter L2 units)"
        )
        section_fig.suptitle(
            f"{result['case']}: y=0 sections at shared radius {first_radius:.4g} L2\nShared Gaussian orientation versus case-specific first update; lines connect measured nodes"
        )
        section_fig.savefig(folder / "matched-radius-sections.png", dpi=180)
        section_fig.savefig(folder / "matched-radius-sections.pdf")
        plt.close(section_fig)
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
    additions = list(getattr(args, "add_planes", None) or [])
    if getattr(args, "add_reference", False) and "reference" not in additions:
        additions.append("reference")
    augment = bool(additions)
    prior = json.loads(target.read_text()) if target.exists() else None
    if augment and prior is None:
        raise FileNotFoundError(
            "--add-planes requires an existing complete geometry.json"
        )
    if prior is not None and not augment:
        raise FileExistsError(f"Refuse to overwrite {target}")
    plane_names = (
        [
            name
            for name in PLANE_ORDER
            if name in additions and name not in prior["planes"]
        ]
        if augment
        else args.planes
    )
    if augment and not plane_names:
        raise ValueError("All requested planes already exist")
    if (
        augment
        and args.case == "control"
        and plane_names == ["reference"]
        and "fit" in prior["planes"]
    ):
        add_control_reference(target, args, clean, study, data)
        return
    fixed_reference = None
    if "reference" in plane_names and args.case != "control":
        control_path = run / "geometry/control/geometry.json"
        fixed_reference = reference_definition(
            json.loads(control_path.read_text()), control_path
        )
        if (
            fixed_reference["control_anchor"] != study["anchor"]
            or fixed_reference["data_content_sha256"] != data["content_sha256"]
        ):
            raise ValueError(
                "Control reference anchor/data differ from the current study"
            )
    base_folder = (
        Path(study["anchor"])
        if args.case == "control"
        else run / "cases" / args.case / "defense-125"
    )
    attack_folder = run / "cases" / args.case / "attacks/full-same"
    checkpoint_folders = [
        attack_folder / f"checkpoint-{step}" for step in args.trajectory_steps
    ]
    if (
        any(name in plane_names for name in ("first", "random-step"))
        and 1 not in args.trajectory_steps
    ):
        raise ValueError(
            "First-update and matched-radius random planes require trajectory step 1"
        )
    if "fit" in plane_names and len(args.trajectory_steps) < 2:
        raise ValueError("Fitted plane requires at least two saved states")
    model, _ = clean.runtime(study, base_folder)
    specs = tuple(
        (name, tuple(value.shape)) for name, value in model.named_parameters()
    )
    if fixed_reference is not None and fixed_reference[
        "parameter_schema_sha256"
    ] != clean.digest(specs):
        raise ValueError("Control reference and target parameter schemas differ")
    probes = {
        "forget_train": data["forget_train"],
        "forget_heldout_subset": data["forget_heldout"][:8],
    }

    def evaluate():
        return {group: clean.evaluate(model, rows) for group, rows in probes.items()}

    result = {
        "schema": 3,
        "case": args.case,
        "anchor": str(base_folder),
        "weight_dtype": "float32",
        "forward_dtype": "float32",
        "scalar_reductions_dtype": "float64 CPU SVD/projection; float64 parameter-device cache validation",
        "point_cache": getattr(args, "point_cache", "device"),
        "grid_size": args.grid_size,
        "seed": args.seed,
        "trajectory_steps": args.trajectory_steps,
        "study_sha256": hashlib.sha256((run / "study.json").read_bytes()).hexdigest(),
        "data_content_sha256": data["content_sha256"],
        "geometry_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        **helper_source_provenance(run, clean),
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
        "plane_comparison": "Broad random uses shared unit Gaussian axes at ±125 L2. Random-step uses those same orientations scaled by each case's actual first-update L2 norm, matching the first plane radius. Reference axes and scale are fixed across cases; first/fit axes are case-specific. All centers are case defenders.",
    }
    if prior is not None:
        for key in (
            "case",
            "anchor",
            "data_content_sha256",
            "parameter_schema_sha256",
            "grid_size",
            "seed",
            "trajectory_steps",
            "probes",
        ):
            if result[key] != prior[key]:
                raise ValueError(
                    f"Cannot augment incompatible existing geometry: {key}"
                )
        result["planes"] = copy.deepcopy(prior["planes"])
        result["augmentations"] = copy.deepcopy(prior.get("augmentations", []))
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
        if (
            prior is not None
            and result["checkpoint_files"] != prior["checkpoint_files"]
        ):
            raise ValueError(
                "Cannot augment geometry after anchor or attack checkpoint bytes changed"
            )
        set_point(model, anchor)
        result["anchor_source_dtypes"] = sorted(set(anchor.source_dtypes.values()))
        result["anchor_load_policy"] = (
            "Original stored anchor values expanded to FP32, exactly as runtime(); lost precision is not recovered"
            if args.case == "control"
            else "Native FP32 defender checkpoint required"
        )
        result["anchor_probes"] = evaluate()
        shared = None
        first_radius = None
        if any(name in plane_names for name in ("random-step", "first")):
            first_radius = displacement_norm(
                readers[args.trajectory_steps.index(1)], anchor, specs
            )
        # Consume the shared random axis in the first-update plane before
        # constructing reference axes; only two full CPU direction maps remain.
        for name in (
            name
            for name in ("random", "random-step", "first", "reference", "fit")
            if name in plane_names
        ):
            if (
                name == "fit"
                and args.case == "control"
                and "reference" in result["planes"]
            ):
                plane = copy.deepcopy(result["planes"]["reference"])
                plane.pop("reference_definition")
                plane["axis_labels"] = [
                    "Uncentered SVD axis 1 (scaled radius)",
                    "Uncentered SVD axis 2 (scaled radius)",
                ]
                plane["coordinate_scope"] = "case-specific physical axes"
                plane["reused_from"] = {
                    "plane": "reference",
                    "reason": "Control fitted and reference axes are identical",
                }
                result["planes"][name] = plane
                continue
            if (
                name == "reference"
                and args.case == "control"
                and "fit" in result["planes"]
            ):
                plane = copy.deepcopy(result["planes"]["fit"])
                plane["reference_definition"] = reference_definition(result, target)
                plane["axis_labels"] = [
                    "Fixed control attack axis 1",
                    "Fixed control attack axis 2",
                ]
                plane["coordinate_scope"] = (
                    "shared physical axes and scale across cases"
                )
                plane["reused_from"] = {
                    "plane": "fit",
                    "reason": "Control reference equals existing control fit; no new evaluations",
                }
                result["planes"][name] = plane
                continue
            if name == "random":
                shared = random_basis(specs, args.seed)
                axes = shared
                extent, labels, extra = (
                    125.0,
                    ["Shared Gaussian axis 1 (L2)", "Shared Gaussian axis 2 (L2)"],
                    {},
                )
            elif name == "random-step":
                if shared is None or len(shared) != 2:
                    shared = random_basis(specs, args.seed)
                axes, first_gaussian = matched_radius_random_basis(shared, first_radius)
                shared = (first_gaussian,)
                first_gaussian = None
                extent, labels, extra = (
                    1.0,
                    [
                        "Shared Gaussian axis 1 (first-update L2 units)",
                        "Shared Gaussian axis 2 (first-update L2 units)",
                    ],
                    {
                        "scale_l2": first_radius,
                        "actual_first_update_l2": first_radius,
                        "radius_policy": "Per-axis L2 radius equals the actual first full-same update norm and first-plane radius",
                    },
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
                axes, scale = first_update_basis(first, gaussian, length=first_radius)
                extent, labels, extra = (
                    1.0,
                    [
                        "Actual first update (multiples)",
                        "Orthogonal shared Gaussian (same L2)",
                    ],
                    {"scale_l2": scale, "actual_first_update_l2": first_radius},
                )
            elif name == "reference":
                shared, axes = None, None
                if args.case == "control":
                    if tuple(args.trajectory_steps) != REFERENCE_STEPS:
                        raise ValueError(
                            "Control reference requires steps 1, 4, 16, 64"
                        )
                    gram = trajectory_gram(readers, anchor, specs)
                    axes, extra = fitted_basis(readers, anchor, specs, gram)
                    fixed_reference = reference_definition(
                        {**result, "planes": {"fit": extra}}, target
                    )
                else:
                    axes = control_reference_axes(run, fixed_reference, specs)
                    extra = {
                        key: fixed_reference[key]
                        for key in (
                            "scale_l2",
                            "gram",
                            "eigenvalues",
                            "coefficients",
                            "top_two_displacement_energy_fraction",
                        )
                    }
                extra["reference_definition"] = fixed_reference
                extent, labels = 1.0, [
                    "Fixed control attack axis 1",
                    "Fixed control attack axis 2",
                ]
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
                "coordinate_scope": (
                    "shared physical axes and scale across cases"
                    if name in {"random", "reference"}
                    else (
                        "shared Gaussian orientations; case-specific first-update L2 scale"
                        if name == "random-step"
                        else "case-specific physical axes"
                    )
                ),
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
            cache = None
            try:
                if getattr(args, "point_cache", "device") == "device":
                    setup_started = time.monotonic()
                    cache = DevicePointCache(model, anchor, axes)
                    checks = [(0.0, 0.0), (1.0, 0.0), (-0.375 * extent, 0.625 * extent)]
                    plane["point_construction"] = {
                        "mode": "device cache",
                        "resident_cache_bytes": cache.resident_bytes,
                        "model_parameter_bytes": sum(
                            p.numel() * p.element_size() for p in model.parameters()
                        ),
                        "operation_order": "copy saved anchor, then separate axis1 add_(alpha=x), axis2 add_(alpha=y)",
                        "validation": validate_point_cache(
                            cache,
                            anchor,
                            axes,
                            checks,
                            (
                                evaluate
                                if getattr(args, "cache_check_ce", False)
                                else None
                            ),
                        ),
                    }
                    plane["point_construction"]["setup_and_validation_seconds"] = (
                        time.monotonic() - setup_started
                    )
                    point_setter = cache.set
                else:
                    plane["point_construction"] = {
                        "mode": "streaming",
                        "resident_cache_bytes": 0,
                    }
                    point_setter = lambda xy: set_point(model, anchor, axes, xy)
                print(
                    json.dumps(
                        {
                            "case": args.case,
                            "plane": name,
                            "point_construction": plane["point_construction"],
                        }
                    ),
                    flush=True,
                )
                for i, step in enumerate(args.trajectory_steps):
                    xy = coordinates[i].tolist()
                    point_setter(xy)
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
                if cache is not None:
                    # The Gaussian's first axis is needed once more by the
                    # case-specific first-update plane; release every other
                    # full CPU basis before evaluating the cached grid.
                    shared = (
                        shared
                        if name == "random" and "random-step" in plane_names
                        else (
                            (shared[0],)
                            if name in {"random", "random-step"}
                            and "first" in plane_names
                            else None
                        )
                    )
                    axes = None
                    if name == "first":
                        first, gaussian = None, None
                for y in plane["coordinates"]:
                    for x in plane["coordinates"]:
                        point_started = time.monotonic()
                        point_setter((x, y))
                        point = {"x": x, "y": y, **evaluate()}
                        point["seconds"] = time.monotonic() - point_started
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
            finally:
                if cache is not None:
                    cache.close()
            axes = None
            if name == "first":
                first, gaussian = None, None
        set_point(model, anchor)
    result["planes"] = {
        name: result["planes"][name] for name in PLANE_ORDER if name in result["planes"]
    }
    try:
        result["figure_ce_scale"] = plot(result, folder, args.ce_limits)
    except ModuleNotFoundError as error:
        if error.name not in {"matplotlib", "numpy"}:
            raise
        result["figure_state"] = (
            f"Numerical grids complete; --plot-only requires {error.name}"
        )
        print(result["figure_state"], flush=True)
    if augment:
        archive_augmentation(target, result, clean)
    else:
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
        choices=PLANE_ORDER,
        default=list(PLANE_ORDER),
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
    parser.add_argument("--add-reference", action="store_true")
    parser.add_argument("--add-planes", nargs="+", choices=PLANE_ORDER)
    parser.add_argument(
        "--point-cache", choices=("device", "streaming"), default="device"
    )
    parser.add_argument(
        "--cache-check-ce",
        action="store_true",
        help="Also compare CE for cached/streaming validation coordinates",
    )
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
    if args.add_planes is not None and len(set(args.add_planes)) != len(
        args.add_planes
    ):
        parser.error("--add-planes must contain distinct plane names")
    if args.plot_only and (args.add_reference or args.add_planes):
        parser.error("--plot-only and plane augmentation are separate operations")
    args.planes = [name for name in PLANE_ORDER if name in args.planes]
    if args.plot_only:
        plot_saved(args)
    else:
        run_geometry(args)


if __name__ == "__main__":
    main()
