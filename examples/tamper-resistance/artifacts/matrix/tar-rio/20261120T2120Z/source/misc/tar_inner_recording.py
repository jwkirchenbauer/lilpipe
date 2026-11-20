"""Measure TAR inner displacements against fixed checkpoints using aggregate inner products."""

import copy
import gc
import hashlib
import json
import math
import os
import random
from collections import Counter
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from transformers import TrainerCallback

from tar_basis_structure import WeightReader, classify_parameter
from tar_dynamics_training import (
    RecordedRelearnTrainer,
    RecordedTARMETrainer,
    load_attack_trace,
)
from trainers.tr_tar import ROLE_FIELDS, adamw_attack


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
        build_device="cpu",
        materialize_references=False,
    ):
        self.root = Path(root)
        self.inner_steps = tuple(inner_steps)
        self.references = references
        self.chunk_elements = chunk_elements
        self.spec_path = spec_path
        if not isinstance(materialize_references, bool):
            raise ValueError("materialize_references must be a boolean")
        self.materialize_references = materialize_references
        self.reduction_device = torch.device(reduction_device)
        if self.reduction_device.type not in {"cpu", "cuda"}:
            raise ValueError("Inner Gram reduction requires a CPU or CUDA/ROCm device")
        if self.reduction_device.type == "cuda" and self.reduction_device.index is None:
            self.reduction_device = torch.device("cuda", torch.cuda.current_device())
        self.build_device = torch.device(build_device)
        if (
            self.build_device == torch.device("cuda")
            and self.reduction_device.type == "cuda"
        ):
            self.build_device = self.reduction_device
        if self.build_device != torch.device("cpu") and (
            self.build_device.type != "cuda"
            or self.build_device != self.reduction_device
        ):
            raise ValueError(
                "Block building requires CPU or the selected reduction GPU"
            )
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
            "build_device": str(self.build_device),
            "materialize_references": self.materialize_references,
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

    def _accumulate_device_chunk(
        self, initial, observed, saved, section, matrices, gaps
    ):
        """Transfer native slices once and form both displacement blocks on the device."""

        def native(tensor):
            return tensor.view(-1)[section].to(self.build_device, non_blocking=True)

        initial = native(initial).float()
        archived = native(saved[self.references[0]["anchor"]]).float()
        # All supported endpoint dtypes convert exactly to FP32.
        values = torch.empty(
            (len(observed) + len(self.references), initial.numel()),
            dtype=torch.float32,
            device=self.build_device,
        )
        for i, tensor in enumerate(observed):
            values[i].copy_(native(tensor))
        for i, ref in enumerate(self.references, start=len(observed)):
            values[i].copy_(native(saved[ref["path"]]))
        # Use the original FP32 buffer last so the raw branch can subtract in place.
        for rep in ("bf16", "raw"):
            if rep == "bf16":
                updates = values.to(torch.bfloat16).float()
                start_inner = initial.to(torch.bfloat16).float()
                start_actual = archived.to(torch.bfloat16).float()
            else:
                updates, start_inner, start_actual = values, initial, archived
            updates[: len(observed)].sub_(start_inner)
            updates[len(observed) :].sub_(start_actual)
            gap = (start_inner - start_actual).double()
            gaps[rep].add_(torch.dot(gap, gap))
            del gap
            block = updates.double()
            matrices[rep].addmm_(block, block.T)
            del block, updates

    def _reference_parameters(self, readers, name):
        values = {path: reader.tensor(name) for path, reader in readers.items()}
        if self.materialize_references:
            for path in values:
                values[path] = values[path].clone(memory_format=torch.contiguous_format)
        return values

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
        gaps = {
            key: (
                torch.zeros((), dtype=torch.float64, device=self.build_device)
                if self.build_device.type == "cuda"
                else 0.0
            )
            for key in matrices
        }
        archived_anchor = self.references[0]["anchor"]
        paths = sorted(
            {ref[field] for ref in self.references for field in ("anchor", "path")}
        )
        files = {}
        with ExitStack() as stack:
            readers = {path: WeightReader(path, stack) for path in paths}
            transfer_stream = None
            if self.materialize_references and self.build_device.type == "cuda":
                transfer_stream = torch.cuda.current_stream(self.build_device)
                # An interrupted parameter may still have host-to-device copies in flight.
                stack.callback(transfer_stream.synchronize)
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
                saved = self._reference_parameters(readers, name)
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
                    if self.build_device.type == "cuda":
                        self._accumulate_device_chunk(
                            initial,
                            observed,
                            saved,
                            section,
                            family_matrices[family],
                            gaps,
                        )
                        continue
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
                if transfer_stream is not None:
                    transfer_stream.synchronize()
                del saved  # Release all owned reference copies before the next parameter.
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
            overall[rep]["anchor_gap_squared_norm"] = float(value)
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
        materialize = os.environ.get("TAR_INNER_MATERIALIZE_REFERENCES", "0")
        if materialize not in {"0", "1"}:
            raise ValueError("TAR_INNER_MATERIALIZE_REFERENCES must be 0 or 1")
        recorder = TransientInnerRecorder(
            root,
            _steps("TAR_INNER_SAVE_STEPS"),
            references,
            spec_path=str(spec_path),
            reduction_device=os.environ.get("TAR_INNER_REDUCTION_DEVICE", "cpu"),
            build_device=os.environ.get("TAR_INNER_BUILD_DEVICE", "cpu"),
            materialize_references=materialize == "1",
            chunk_elements=int(os.environ.get("TAR_INNER_CHUNK_ELEMENTS", "131072")),
        )
        inner = self.sample_attack_fn

        def observed(model, inputs, **kwargs):
            step = self.state.global_step + 1
            if step in outer_steps and model.training and torch.is_grad_enabled():
                return recorder.run(inner, model, inputs, step, **kwargs)
            return inner(model, inputs, **kwargs)

        self.sample_attack_fn = observed


def parity_parameter_statistics(parameters, expected, anchor, chunk_elements=1048576):
    """Compare native saved coordinates, normalized by the expected attack update."""
    squared_error, squared_update, maximum, coordinates = 0.0, 0.0, 0.0, 0
    dtype_mismatches = {}
    for name, parameter in parameters.items():
        reference = expected(name)
        initial = anchor[name]
        if parameter.shape != reference.shape or parameter.shape != initial.shape:
            raise ValueError(f"Parity parameter shape differs: {name}")
        if parameter.dtype != reference.dtype:
            dtype_mismatches[name] = [str(parameter.dtype), str(reference.dtype)]
        current = parameter.detach().to(device="cpu", copy=True).reshape(-1)
        reference = reference.reshape(-1)
        initial = initial.reshape(-1)
        for start in range(0, current.numel(), chunk_elements):
            stop = start + chunk_elements
            difference = current[start:stop].double() - reference[start:stop].double()
            update = reference[start:stop].double() - initial[start:stop].double()
            squared_error += float(difference.square().sum())
            squared_update += float(update.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
        coordinates += current.numel()
    if not all(
        math.isfinite(value) for value in (squared_error, squared_update, maximum)
    ):
        raise ValueError("Nonfinite parity statistics")
    return {
        "coordinates": coordinates,
        "difference_l2": math.sqrt(squared_error),
        "expected_update_l2": math.sqrt(squared_update),
        "relative_update_error": (
            math.sqrt(squared_error / squared_update)
            if squared_update
            else (0.0 if not squared_error else None)
        ),
        "maximum_absolute_difference": maximum,
        "exact_equal": squared_error == 0 and not dtype_mismatches,
        "dtype_mismatches": dtype_mismatches,
    }


class _ReplayParityCallback(TrainerCallback):
    """Stop a genuine 1000-step-schedule attack after its checked 16-step prefix."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.trainer
        trainer._check_actual_parity(state.global_step)
        if state.global_step == trainer.parity_steps[-1]:
            control.should_save = True
            control.should_training_stop = True
            trainer._publish_parity()
        return control


class RecordedReplayParityTrainer(RecordedRelearnTrainer):
    """Compare a copied inner simulation with a genuine Axolotl attacker prefix.

    The actual trainer retains its 1000-step LR schedule and original sampler.
    The copied simulation runs before the actual first update, in the same model
    representation and autocast context. Its selected saved iterates are retained.
    """

    def __init__(
        self,
        *args,
        parity_trace_path,
        parity_reference_root,
        parity_steps=(1, 2, 4, 8, 16),
        **kwargs,
    ):
        self.parity_steps = tuple(parity_steps)
        if (
            not self.parity_steps
            or self.parity_steps != tuple(sorted(set(self.parity_steps)))
            or self.parity_steps[0] < 1
            or self.parity_steps[-1] != 16
        ):
            raise ValueError(
                "Parity requires ordered positive saves ending at update 16"
            )
        self.parity_trace_path = Path(parity_trace_path)
        self.parity_reference_root = Path(parity_reference_root)
        self._parity_trace = load_attack_trace(
            self.parity_trace_path, expected_steps=16
        )
        self._parity_started = False
        self._parity_checks, self._parity_simulated_losses = [], []
        super().__init__(*args, **kwargs)
        if self.args.max_steps != 1000 or self.args.gradient_accumulation_steps != 1:
            raise ValueError(
                "Parity must preserve the standard 1000-step scheduler and accumulation=1"
            )
        self.add_callback(_ReplayParityCallback(self))

    def _save_parity_model(self, model, name):
        from safetensors.torch import save_file

        folder = Path(self.args.output_dir) / "parity" / name
        folder.mkdir(parents=True, exist_ok=False)
        save_file(_cpu_parameters(model), folder / "model.safetensors")

    def _simulate_prefix(self, model):
        self._parity_started = True
        if self.args.max_grad_norm != 1.0:
            raise ValueError(
                "Parity requires the actual attack's clipping threshold of one"
            )
        for group in self.optimizer.param_groups:
            if (
                tuple(group["betas"]) != (0.9, 0.999)
                or group["eps"] != 1e-8
                or group["weight_decay"] != 0
            ):
                raise ValueError(
                    "Actual optimizer settings differ from the declared matched simulation"
                )
        self._parity_initial = _cpu_parameters(model)
        self._save_parity_model(model, "anchor")
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        groups = [
            [names[id(parameter)] for parameter in group["params"]]
            for group in self.optimizer.param_groups
        ]
        self._parity_optimizer_groups = [
            {
                "names": group,
                **{key: value for key, value in original.items() if key != "params"},
            }
            for group, original in zip(groups, self.optimizer.param_groups)
        ]
        devices = sorted(
            {p.device.index for p in model.parameters() if p.device.type == "cuda"}
        )
        with torch.random.fork_rng(devices=devices):
            copied = copy.deepcopy(model)
            copied.forward = MethodType(type(copied).forward, copied)
            batches, rates, _ = self._parity_trace
            device = next(model.parameters()).device
            batches = [
                {field: value.to(device) for field, value in batch.items()}
                for batch in batches
            ]

            def saved(step):
                if step in self.parity_steps:
                    self._save_parity_model(copied, f"simulated-{step}")

            def measured(step, batch, loss, learning_rate, grad_norm):
                self._parity_simulated_losses.append(
                    {
                        "step": step,
                        "loss_before_update": loss,
                        "learning_rate_used": learning_rate,
                        "preclip_grad_norm": grad_norm,
                    }
                )

            with self.accelerator.autocast():
                adamw_attack(
                    copied,
                    batches[0],
                    attack_batches=batches,
                    attack_epochs=1,
                    attack_max_steps=16,
                    attack_learning_rate=rates[0],
                    attack_learning_rates=rates,
                    attack_optimizer_mode="matched",
                    attack_parameter_groups=groups,
                    on_step_end=saved,
                    on_inner_step=measured,
                )
            del copied
        gc.collect()

    def training_step(self, model, inputs, *args, **kwargs):
        if not self._parity_started:
            self._simulate_prefix(model)
        step = self.state.global_step + 1
        expected_batch, rates, hashes = self._parity_trace
        batch = {field: inputs[field].detach().cpu().tolist() for field in ROLE_FIELDS}
        actual_hash = hashlib.sha256(
            json.dumps(batch, sort_keys=True).encode()
        ).hexdigest()
        if actual_hash != hashes[step - 1]:
            raise ValueError(f"Actual parity attack batch differs at step {step}")
        if not math.isclose(
            self.optimizer.param_groups[0]["lr"], rates[step - 1], rel_tol=1e-12
        ):
            raise ValueError(f"Actual parity attack LR differs at step {step}")
        return super().training_step(model, inputs, *args, **kwargs)

    def _check_actual_parity(self, step):
        if step not in self.parity_steps:
            return
        with ExitStack() as stack:
            simulated = WeightReader(
                Path(self.args.output_dir) / "parity" / f"simulated-{step}", stack
            )
            archived = WeightReader(
                self.parity_reference_root / f"checkpoint-{step}", stack
            )
            parameters = dict(self.model.named_parameters())
            self._parity_checks.append(
                {
                    "step": step,
                    "simulated_vs_actual": parity_parameter_statistics(
                        parameters, simulated.tensor, self._parity_initial
                    ),
                    "archived_vs_actual": parity_parameter_statistics(
                        parameters, archived.tensor, self._parity_initial
                    ),
                }
            )
        print("Replay parity checked actual update", step, flush=True)

    def _publish_parity(self):
        path = Path(self.args.output_dir) / "observed-batches.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if [record["step"] for record in records] != list(range(1, 17)):
            raise ValueError("Parity actual trajectory must contain exactly 16 updates")
        loss_errors = [
            abs(actual["loss_before_update"] - simulated["loss_before_update"])
            for actual, simulated in zip(records, self._parity_simulated_losses)
        ]
        relative_errors = [
            check["simulated_vs_actual"]["relative_update_error"]
            for check in self._parity_checks
        ]
        verdict = (
            "exact"
            if all(
                check["simulated_vs_actual"]["exact_equal"]
                for check in self._parity_checks
            )
            and max(loss_errors) == 0
            else (
                "within_declared_tolerance"
                if all(value is not None and value <= 1e-3 for value in relative_errors)
                and max(loss_errors) <= 1e-4
                else "outside_declared_tolerance"
            )
        )
        value = {
            "state": "complete",
            "trajectory_parity": verdict,
            "scope": "One unchanged RIO defender, fresh optimizer, exact actual first16 data/order and 1000-step LR schedule. Copied inner model compared with genuine Axolotl updates. Not continuous-state matching across outer defense calls.",
            "tolerances": {"relative_update_l2": 1e-3, "loss_absolute": 1e-4},
            "trace": str(self.parity_trace_path),
            "reference_root": str(self.parity_reference_root),
            "parameter_checks": self._parity_checks,
            "simulated_losses": self._parity_simulated_losses,
            "actual_loss_errors": loss_errors,
            "optimizer_groups": self._parity_optimizer_groups,
            "model_dtype_counts": dict(
                Counter(str(p.dtype) for p in self.model.parameters())
            ),
            "model_device_counts": dict(
                Counter(str(p.device) for p in self.model.parameters())
            ),
            "autocast": str(self.accelerator.mixed_precision),
            "actual_horizon": self.args.max_steps,
            "actual_updates": 16,
            "torch": torch.__version__,
            "flux_job_id": os.environ.get("FLUX_JOB_ID"),
        }
        (Path(self.args.output_dir) / "parity" / "validation.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
