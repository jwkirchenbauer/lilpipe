"""Measure TAR inner displacements against fixed checkpoints using aggregate inner products."""

import hashlib
import json
import os
import random
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch

from tar_basis_structure import WeightReader, classify_parameter
from tar_dynamics_training import RecordedTARMETrainer
from trainers.tr_tar import ROLE_FIELDS


def _rng_state(devices):
    np_rng = np.random.get_state()
    return (
        random.getstate(),
        (*np_rng[:1], np_rng[1].tobytes(), *np_rng[2:]),
        torch.get_rng_state().numpy().tobytes(),
        tuple(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
            for device in devices
        ),
    )


@contextmanager
def _observe(devices):
    before = _rng_state(devices)
    active_device = torch.cuda.current_device() if devices else None
    with torch.no_grad():
        yield
    if _rng_state(devices) != before:
        raise RuntimeError("Inner trajectory observation changed an RNG state")
    if devices and torch.cuda.current_device() != active_device:
        raise RuntimeError(
            "Inner trajectory observation changed the active CUDA device"
        )


def _cpu_parameters(model):
    # Copy individually, including CPU-resident parameters. Never alias the live model.
    return {
        name: parameter.detach().to(
            device="cpu", copy=True, memory_format=torch.contiguous_format
        )
        for name, parameter in model.named_parameters()
    }


def _slice_fp32(tensor, section, representation):
    part = tensor.view(-1)[section]
    if representation == "bf16":
        part = part.to(torch.bfloat16)
    return part.float()


def _steps(variable):
    values = [int(value) for value in os.environ[variable].split(",")]
    if not values or min(values) < 1 or sorted(set(values)) != values:
        raise ValueError(f"{variable} requires strictly increasing positive steps")
    return values


def _representations(matrices, split):
    output = {}
    for rep, matrix in matrices.items():
        matrix = matrix.cpu()
        if not torch.isfinite(matrix).all():
            raise ValueError(f"Nonfinite {rep} Gram matrix")
        norms = matrix.diagonal().sqrt()
        output[rep] = {
            "inner_gram": matrix[:split, :split].tolist(),
            "cross_gram": matrix[:split, split:].tolist(),
            "reference_gram": matrix[split:, split:].tolist(),
            "inner_norms": norms[:split].tolist(),
            "reference_norms": norms[split:].tolist(),
        }
    return output


class TransientInnerRecorder:
    """Hold selected endpoints for one call and publish only full-coordinate Gram matrices."""

    def __init__(
        self,
        root,
        inner_steps,
        references,
        *,
        chunk_elements=131072,
        spec_path=None,
        reduction_device="cpu",
    ):
        self.root = Path(root)
        self.inner_steps = tuple(inner_steps)
        self.references = references
        self.chunk_elements = chunk_elements
        self.spec_path = spec_path
        self.reduction_device = torch.device(reduction_device)
        if self.reduction_device.type not in {"cpu", "cuda"}:
            raise ValueError("Inner Gram reduction requires a CPU or CUDA/ROCm device")
        if self.reduction_device.type == "cuda" and self.reduction_device.index is None:
            self.reduction_device = torch.device("cuda", torch.cuda.current_device())
        names = [ref["name"] for ref in references]
        if not names or len(set(names)) != len(names):
            raise ValueError("Reference directions must have unique names")
        if len({ref["anchor"] for ref in references}) != 1:
            raise ValueError(
                "Reference directions must share their archived starting checkpoint"
            )
        if (
            not self.inner_steps
            or tuple(sorted(set(inner_steps))) != self.inner_steps
            or min(inner_steps) < 1
            or chunk_elements < 1
        ):
            raise ValueError("Invalid selected steps or chunk size")

    def run(self, function, model, inputs, outer_call, **kwargs):
        destination = self.root / f"outer-{outer_call:06d}.json"
        pending = destination.with_suffix(".partial")
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        # This file reserves the call across writers. Keep it on failure as evidence.
        with pending.open("x") as stream:
            json.dump(
                {"schema_version": 1, "outer_call": outer_call, "status": "incomplete"},
                stream,
            )
        if destination.exists():
            raise FileExistsError(destination)

        devices = {p.device for p in model.parameters() if p.is_cuda}
        if self.reduction_device.type == "cuda":
            devices.add(self.reduction_device)
        devices = sorted(devices, key=str)
        print(
            f"Inner recorder outer {outer_call}: reading pre-call parameters",
            flush=True,
        )
        with _observe(devices):
            anchor = _cpu_parameters(model)
            if not anchor or any(
                t.dtype not in {torch.bfloat16, torch.float16, torch.float32}
                for t in anchor.values()
            ):
                raise ValueError("Expected float32, float16 or bfloat16 parameters")
            serial = {key: inputs[key].detach().cpu().tolist() for key in ROLE_FIELDS}
            batch_sha256 = hashlib.sha256(
                json.dumps(serial, sort_keys=True).encode()
            ).hexdigest()
        endpoints = {}
        completed = 0

        def capture(step):
            nonlocal completed
            if step != completed + 1:
                raise ValueError(f"Out-of-order inner step {step} after {completed}")
            completed = step
            if step in self.inner_steps:
                with _observe(devices):
                    endpoints[step] = _cpu_parameters(model)
                print(
                    f"Inner recorder outer {outer_call}: captured inner {step} in memory",
                    flush=True,
                )

        result = function(model, inputs, on_step_end=capture, **kwargs)
        if tuple(endpoints) != self.inner_steps:
            raise ValueError(
                f"Expected inner steps {self.inner_steps}, observed {tuple(endpoints)}"
            )
        reduction_context = (
            torch.cuda.device(self.reduction_device)
            if self.reduction_device.type == "cuda"
            else nullcontext()
        )
        with _observe(devices), reduction_context:
            representations, families, source_files = self._grams(
                anchor, endpoints, outer_call
            )
        record = {
            "schema_version": 1,
            "status": "complete",
            "outer_call": outer_call,
            "anchor": "copied model immediately before the inner call of this outer step",
            "completed_inner_steps": completed,
            "inner_steps": self.inner_steps,
            "reference_names": [ref["name"] for ref in self.references],
            "references": self.references,
            "reference_spec_path": self.spec_path,
            "comparison_anchor": self.references[0]["anchor"],
            "references_sha256": hashlib.sha256(
                json.dumps(self.references, sort_keys=True).encode()
            ).hexdigest(),
            "reference_files": source_files,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "observed_batches_step": outer_call,
            "inner_batch_sha256": batch_sha256,
            "inner_batch_fields": {
                key: {"dtype": str(inputs[key].dtype), "shape": list(inputs[key].shape)}
                for key in ROLE_FIELDS
            },
            "layout": [
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                for name, value in anchor.items()
            ],
            "coordinates": sum(t.numel() for t in anchor.values()),
            "reduction_device": str(self.reduction_device),
            "chunk_elements": self.chunk_elements,
            "representations": representations,
            "families": families,
            "rng_unchanged_during_observation": True,
        }
        with pending.open("w") as stream:
            json.dump(record, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.link(pending, destination)
        pending.unlink()
        print(f"Inner recorder outer {outer_call}: published {destination}", flush=True)
        return result

    def _grams(self, anchor, endpoints, outer_call):
        rows = len(endpoints) + len(self.references)
        matrices = {
            key: torch.zeros(
                (rows, rows), dtype=torch.float64, device=self.reduction_device
            )
            for key in ("raw", "bf16")
        }
        family_matrices = {}
        coordinates = {}
        gaps = {key: 0.0 for key in matrices}
        archived_anchor = self.references[0]["anchor"]
        paths = sorted(
            {ref[field] for ref in self.references for field in ("anchor", "path")}
        )
        files = {}
        with ExitStack() as stack:
            readers = {path: WeightReader(path, stack) for path in paths}
            for folder, reader in readers.items():
                sources = set(reader.handles)
                if reader.mapping is not None:
                    sources.add("model.safetensors.index.json")
                for name in sorted(sources):
                    path = Path(folder) / name
                    stat = path.stat()
                    files[str(path)] = {
                        "bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
            for number, (name, initial) in enumerate(anchor.items(), 1):
                family = classify_parameter(name)[0]
                if family not in family_matrices:
                    family_matrices[family] = {
                        key: torch.zeros_like(value) for key, value in matrices.items()
                    }
                    coordinates[family] = 0
                coordinates[family] += initial.numel()
                saved = {path: reader.tensor(name) for path, reader in readers.items()}
                observed = [values[name] for values in endpoints.values()]
                if any(t.shape != initial.shape for t in (*observed, *saved.values())):
                    raise ValueError(f"Parameter shape mismatch for {name!r}")
                if any(
                    t.dtype not in {torch.bfloat16, torch.float16, torch.float32}
                    for t in saved.values()
                ):
                    raise ValueError(
                        f"Unsupported archived parameter dtype for {name!r}"
                    )
                # BF16 is applied to each actual endpoint before FP32 subtraction.
                # Keeping the transient endpoints avoids a potentially lossy delta round trip.
                for start in range(0, initial.numel(), self.chunk_elements):
                    section = slice(start, start + self.chunk_elements)
                    count = min(self.chunk_elements, initial.numel() - start)
                    block = torch.empty(
                        (rows, count), dtype=torch.float64, device="cpu"
                    )
                    for rep, matrix in family_matrices[family].items():
                        base = _slice_fp32(initial, section, rep)
                        for i, value in enumerate(observed):
                            block[i].copy_(_slice_fp32(value, section, rep) - base)
                        refs = {
                            path: _slice_fp32(value, section, rep)
                            for path, value in saved.items()
                        }
                        difference = (base - refs[archived_anchor]).double()
                        gaps[rep] += torch.dot(difference, difference).item()
                        for i, ref in enumerate(self.references, start=len(observed)):
                            block[i].copy_(refs[ref["path"]] - refs[ref["anchor"]])
                        reduction_block = block.to(self.reduction_device)
                        matrix.addmm_(reduction_block, reduction_block.T)
                        del reduction_block
                if number % 8 == 0 or number == len(anchor):
                    print(
                        f"Inner recorder outer {outer_call}: scanned {number}/{len(anchor)} parameters through {name} on {self.reduction_device}",
                        flush=True,
                    )

        split = len(endpoints)
        families = {}
        for family, group in family_matrices.items():
            for rep, matrix in group.items():
                matrices[rep] += matrix
            families[family] = {
                "coordinates": coordinates[family],
                "representations": _representations(group, split),
            }
        overall = _representations(matrices, split)
        for rep, value in gaps.items():
            overall[rep]["anchor_gap_squared_norm"] = value
        return overall, families, files


class RecordedInnerTARMETrainer(RecordedTARMETrainer):
    """Observe selected inner calls alongside the existing mixed-batch recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.world_size != 1:
            raise ValueError("Aggregate inner recording expects one trainer process")
        root = os.environ["TAR_INNER_RECORD_ROOT"]
        spec_path = Path(os.environ["TAR_INNER_REFERENCE_SPEC"]).resolve()
        references = json.loads(spec_path.read_text())["references"]
        outer_steps = frozenset(_steps("TAR_INNER_OUTER_STEPS"))
        recorder = TransientInnerRecorder(
            root,
            _steps("TAR_INNER_SAVE_STEPS"),
            references,
            spec_path=str(spec_path),
            reduction_device=os.environ.get("TAR_INNER_REDUCTION_DEVICE", "cpu"),
            chunk_elements=int(os.environ.get("TAR_INNER_CHUNK_ELEMENTS", "131072")),
        )
        inner = self.sample_attack_fn

        def observed(model, inputs, **kwargs):
            step = self.state.global_step + 1
            if step in outer_steps and model.training and torch.is_grad_enabled():
                return recorder.run(inner, model, inputs, step, **kwargs)
            return inner(model, inputs, **kwargs)

        self.sample_attack_fn = observed
