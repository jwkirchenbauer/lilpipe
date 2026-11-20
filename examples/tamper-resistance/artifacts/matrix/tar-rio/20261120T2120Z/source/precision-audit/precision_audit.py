#!/usr/bin/env python3
"""Streaming precision audit for the two frozen RIO replay anchors.

This reads safetensors only. It does not instantiate a model, run inference, or
train. Tensor arithmetic is performed in bounded chunks on one selected device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import __version__ as safetensors_version
from safetensors import safe_open

STEPS = (1, 2, 4, 8, 16)
ORIGIN = Path(
    "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z"
)


def module_bucket(name: str) -> str:
    lower = name.lower()
    if "norm" in lower or "layernorm" in lower:
        return "norm"
    if "embed" in lower or "lm_head" in lower:
        return "embedding_or_head"
    return "other"


def bf16_roundtrip(anchor: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Return anchor minus native cast through BF16 and back to native dtype."""
    casted = native.to(torch.bfloat16).to(native.dtype)
    return anchor.to(torch.float64) - casted.to(torch.float64)


def straight_through_reconstruction(
    anchor: torch.Tensor, endpoint: torch.Tensor
) -> torch.Tensor:
    """Apply the two explicit parameter-dtype operations used by TAR's ST value."""
    update = endpoint - anchor
    return anchor + update


def _tensor_chunks(handle: Any, key: str, max_elements: int) -> Iterator[torch.Tensor]:
    view = handle.get_slice(key)
    shape = tuple(view.get_shape())
    if not shape:
        yield view[()]
        return
    row_size = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, max_elements // max(1, row_size))
    for start in range(0, shape[0], rows):
        yield view[start : min(shape[0], start + rows)]


def _finite(tensor: torch.Tensor, label: str) -> None:
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise ValueError(f"Non-finite tensor values in {label}")


class Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.changed = 0
        self.sq = 0.0
        self.ref_sq = 0.0
        self.max_abs = 0.0
        self.dot = 0.0

    def add(self, delta: torch.Tensor, reference: torch.Tensor) -> None:
        d = delta.detach().to(dtype=torch.float64)
        r = reference.detach().to(dtype=torch.float64)
        if not torch.isfinite(d).all().item() or not torch.isfinite(r).all().item():
            raise ValueError("Non-finite arithmetic result")
        self.n += d.numel()
        self.changed += torch.count_nonzero(d).item()
        self.sq += torch.sum(d * d).item()
        self.ref_sq += torch.sum(r * r).item()
        if d.numel():
            self.max_abs = max(self.max_abs, torch.max(torch.abs(d)).item())
            self.dot += torch.sum(d * r).item()

    def finish(self) -> dict[str, Any]:
        l2 = math.sqrt(self.sq)
        ref_l2 = math.sqrt(self.ref_sq)
        return {
            "coordinates": self.n,
            "changed_coordinates": self.changed,
            "difference_l2": l2,
            "reference_l2": ref_l2,
            "relative_to_reference_l2": l2 / ref_l2 if ref_l2 else None,
            "maximum_absolute_difference": self.max_abs,
            "cosine_difference_to_reference": (
                self.dot / (l2 * ref_l2) if l2 and ref_l2 else None
            ),
        }


def _metadata(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {
            key: (
                tuple(f.get_slice(key).get_shape()),
                str(f.get_slice(key).get_dtype()),
            )
            for key in f.keys()
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pair_inputs(
    native: Path, parity_dir: Path, validation_path: Path
) -> tuple[dict[str, Any], list[Path], dict[str, tuple[tuple[int, ...], str]]]:
    validation = json.loads(validation_path.read_text())
    if (
        validation.get("state") != "complete"
        or validation.get("trajectory_parity") != "exact"
    ):
        raise ValueError(
            f"{validation_path}: expected complete exact same-backend parity"
        )
    losses = validation.get("actual_loss_errors")
    if (
        not isinstance(losses, list)
        or len(losses) != 16
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value != 0.0
            for value in losses
        )
    ):
        raise ValueError(
            f"{validation_path}: expected 16 exact finite same-backend loss matches"
        )
    checks = {
        int(item["step"]): item for item in validation.get("parameter_checks", [])
    }
    if not set(STEPS).issubset(checks):
        raise ValueError(f"{validation_path}: missing checked states {STEPS}")
    for step in STEPS:
        comparison = checks[step].get("simulated_vs_actual", {})
        if (
            comparison.get("exact_equal") is not True
            or comparison.get("difference_l2") != 0.0
        ):
            raise ValueError(
                f"{validation_path}: simulated_vs_actual state {step} is not bitwise exact"
            )
        # Archive checkpoints were produced on ROCm. Their comparison is retained
        # as a cross-backend diagnostic and must never gate same-backend parity.
        archived = checks[step].get("archived_vs_actual", {})
        if (
            not isinstance(archived.get("difference_l2"), (int, float))
            or not math.isfinite(archived["difference_l2"])
            or archived["difference_l2"] < 0
        ):
            raise ValueError(
                f"{validation_path}: missing finite archive diagnostic at state {step}"
            )
    ref_root = Path(validation.get("reference_root", ""))
    if not ref_root.is_absolute():
        raise ValueError(f"{validation_path}: reference_root must be absolute")
    required = [native, parity_dir / "anchor/model.safetensors"]
    for step in STEPS:
        required.extend(
            [
                parity_dir / f"simulated-{step}/model.safetensors",
                ref_root / f"checkpoint-{step}/model.safetensors",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required parity audit inputs are missing:\n" + "\n".join(missing)
        )
    metadata = [_metadata(path) for path in required]
    base_meta = metadata[0]
    for path, meta in zip(required[1:], metadata[1:], strict=True):
        if set(meta) != set(base_meta):
            raise ValueError(f"Tensor-key mismatch: {path}")
        for key in base_meta:
            if meta[key][0] != base_meta[key][0]:
                raise ValueError(f"Tensor shape mismatch in {path}: {key}")
            if meta[key][1] != base_meta[key][1]:
                raise ValueError(
                    f"Tensor dtype mismatch in {path}: {key} ({base_meta[key][1]} vs {meta[key][1]})"
                )
    return validation, required, base_meta


def _open_inputs(paths: list[Path]) -> list[Any]:
    return [safe_open(str(path), framework="pt", device="cpu") for path in paths]


def analyze_pair(
    label: str,
    native: Path,
    parity_dir: Path,
    validation_path: Path,
    device: torch.device,
    max_elements: int,
    *,
    anchor_backend: str = "unspecified",
    simulated_backend: str = "unspecified",
    reference_backend: str = "unspecified",
    prepared: (
        tuple[dict[str, Any], list[Path], dict[str, tuple[tuple[int, ...], str]]] | None
    ) = None,
) -> list[dict[str, Any]]:
    validation, required, base_meta = prepared or validate_pair_inputs(
        native, parity_dir, validation_path
    )

    files = _open_inputs(required)
    try:
        # Required order: native, anchor, then for each state simulated, actual.
        accumulators: dict[tuple[str, str, str, int | None], Accumulator] = defaultdict(
            Accumulator
        )
        names = sorted(base_meta)
        for key in names:
            dtype_label = base_meta[key][1]
            bucket = module_bucket(key)
            streams = [
                _tensor_chunks(files[0], key, max_elements),
                _tensor_chunks(files[1], key, max_elements),
            ]
            for step_index, _step in enumerate(STEPS):
                streams.extend(
                    [
                        _tensor_chunks(files[2 + 2 * step_index], key, max_elements),
                        _tensor_chunks(files[3 + 2 * step_index], key, max_elements),
                    ]
                )
            for parts in zip(*streams, strict=True):
                native_cpu, anchor_cpu = parts[:2]
                n = native_cpu.to(device=device, non_blocking=True)
                a = anchor_cpu.to(device=device, non_blocking=True)
                _finite(n, f"native:{key}")
                _finite(a, f"anchor:{key}")

                cast_acc = accumulators[
                    ("native_bf16_roundtrip", dtype_label, bucket, None)
                ]
                cast_acc.add(bf16_roundtrip(a, n), n)
                raw_acc = accumulators[("native_to_reload", dtype_label, bucket, None)]
                raw_delta = a.to(torch.float64) - n.to(torch.float64)
                raw_acc.add(raw_delta, n.to(torch.float64))
                for step_index, step in enumerate(STEPS):
                    sim_cpu, actual_cpu = parts[2 + 2 * step_index : 4 + 2 * step_index]
                    sim = sim_cpu.to(device=device, non_blocking=True)
                    actual = actual_cpu.to(device=device, non_blocking=True)
                    _finite(sim, f"simulated-{step}:{key}")
                    _finite(actual, f"actual-{step}:{key}")
                    sim_update = sim.to(torch.float64) - a.to(torch.float64)
                    archived_difference = actual.to(torch.float64) - sim.to(
                        torch.float64
                    )
                    accumulators[
                        ("reference_vs_simulated", dtype_label, bucket, step)
                    ].add(archived_difference, sim_update)
                    if step in (1, 16):
                        accumulators[
                            ("reload_denominator", dtype_label, bucket, step)
                        ].ref_sq += torch.sum(sim_update * sim_update).item()
                    effective = straight_through_reconstruction(a, sim)
                    effective_delta = effective.to(torch.float64) - sim.to(
                        torch.float64
                    )
                    st_acc = accumulators[
                        (
                            "straight_through_reconstruction",
                            dtype_label,
                            bucket,
                            step,
                        )
                    ]
                    st_acc.add(effective_delta, sim_update)

        # Build normalized summary rows from accumulated raw statistics and denominators.
        rows: list[dict[str, Any]] = []
        for (measure, dtype_label, bucket, step), acc in sorted(accumulators.items()):
            if measure == "reload_denominator":
                continue
            stat = acc.finish()
            stat["relative_to_actual_attack_update_1_l2"] = None
            stat["relative_to_actual_attack_update_16_l2"] = None
            stat["exact_equal_to_native_bf16_roundtrip"] = None
            if measure == "native_to_reload":
                for denom_step in (1, 16):
                    den = (
                        accumulators[
                            ("reload_denominator", dtype_label, bucket, denom_step)
                        ].ref_sq
                        ** 0.5
                    )
                    stat[f"relative_to_actual_attack_update_{denom_step}_l2"] = (
                        stat["difference_l2"] / den if den else None
                    )
            if measure == "native_bf16_roundtrip":
                stat["exact_equal_to_native_bf16_roundtrip"] = (
                    stat["changed_coordinates"] == 0
                )
            rows.append(
                {
                    "anchor": label,
                    "anchor_backend": anchor_backend,
                    "simulated_backend": simulated_backend,
                    "reference_backend": reference_backend,
                    "comparison_backend": (
                        f"{simulated_backend}-vs-{reference_backend}"
                        if measure == "reference_vs_simulated"
                        else None
                    ),
                    "analysis_device": str(device),
                    "measure": measure,
                    "dtype": dtype_label,
                    "module_bucket": bucket,
                    "step": step,
                    **stat,
                }
            )
        return rows
    finally:
        for f in files:
            f.__exit__(None, None, None)


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        temp = Path(f.name)
    os.replace(temp, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(f.name)
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-elements", type=int, default=1_000_000)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.chunk_elements <= 0:
        parser.error("--chunk-elements must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A single visible GPU is required for bounded GPU arithmetic"
        )
    device = torch.device(args.device)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise ValueError(
            f"Invalid requested device {device}; visible CUDA count={torch.cuda.device_count()}"
        )

    pairs = [
        {
            "name": "archived_rio_v2",
            "anchor_backend": "archived-rocm",
            "simulated_backend": "archived-rocm",
            "reference_backend": "archived-rocm",
            "native": ORIGIN.parent.parent
            / "tar-hparams/20261116T2310Z/cases/hparam-rio-defense/models/defense/model.safetensors",
            "parity": ORIGIN / "pilots/rio-prefix-v2/models/attack/parity",
        },
        {
            "name": "forward_matched_new_anchor",
            "anchor_backend": "fresh-cuda",
            "simulated_backend": "fresh-cuda",
            "reference_backend": "archived-rocm",
            "native": run
            / "cases/rio-prefix-forward-matched-defense/models/defense/model.safetensors",
            "parity": run / "pilots/rio-prefix-new-anchor/models/attack/parity",
        },
    ]
    # Validate both full input families before reading tensor contents or writing outputs.
    for pair in pairs:
        validation = pair["parity"] / "validation.json"
        if not validation.is_file():
            raise FileNotFoundError(
                f"Required parity validation is missing: {validation}"
            )
        if not pair["native"].is_file():
            raise FileNotFoundError(
                f"Native defender checkpoint is missing: {pair['native']}"
            )
    prepared_pairs = [
        validate_pair_inputs(
            pair["native"], pair["parity"], pair["parity"] / "validation.json"
        )
        for pair in pairs
    ]

    all_rows: list[dict[str, Any]] = []
    input_provenance = []
    for pair, prepared in zip(pairs, prepared_pairs, strict=True):
        validation = pair["parity"] / "validation.json"
        rows = analyze_pair(
            pair["name"],
            pair["native"],
            pair["parity"],
            validation,
            device,
            args.chunk_elements,
            anchor_backend=pair["anchor_backend"],
            simulated_backend=pair["simulated_backend"],
            reference_backend=pair["reference_backend"],
            prepared=prepared,
        )
        all_rows.extend(rows)
        reference_root = Path(prepared[0]["reference_root"])
        input_provenance.extend(
            [pair["native"], validation, pair["parity"] / "anchor/model.safetensors"]
        )
        for step in STEPS:
            input_provenance.extend(
                [
                    pair["parity"] / f"simulated-{step}/model.safetensors",
                    reference_root / f"checkpoint-{step}/model.safetensors",
                ]
            )
    small_provenance_files = [
        Path(__file__).resolve(),
        Path(__file__).with_name("worker.sh"),
        Path(__file__).with_name("README.md"),
        *(pair["parity"] / "validation.json" for pair in pairs),
    ]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("FLUX_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hip": torch.version.hip,
        "safetensors": safetensors_version,
        "device": str(device),
        "visible_gpu_count": torch.cuda.device_count(),
        "chunk_elements": args.chunk_elements,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in input_provenance
        ],
        "source_files": [
            {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in small_provenance_files
        ],
        "scope": "CUDA same-backend simulated-versus-genuine exact parity is a prerequisite; native reload drift and straight-through dtype arithmetic are measured at the saved parity anchors. Archived ROCm states are retained as cross-backend diagnostics, not equality gates. No inference or training.",
        "same_backend_parity_requirement": "simulated_vs_actual exact_equal=true and difference_l2=0 at steps 1,2,4,8,16; all 16 actual loss errors are exactly zero",
        "reference_comparison_scope": "The archived_rio_v2 pair compares AMD simulation states against AMD archive states. The forward_matched_new_anchor pair compares fresh CUDA simulated/genuine states against archived AMD states. Only the latter is cross-backend; neither reference comparison gates same-backend parity.",
        "anchor_backend_by_pair": {
            pair["name"]: pair["anchor_backend"] for pair in pairs
        },
        "simulated_backend_by_pair": {
            pair["name"]: pair["simulated_backend"] for pair in pairs
        },
        "reference_backend_by_pair": {
            pair["name"]: pair["reference_backend"] for pair in pairs
        },
        "archived_parity_diagnostics_by_pair": {
            pair["name"]: {
                str(step): next(
                    item["archived_vs_actual"]
                    for item in prepared[0]["parameter_checks"]
                    if int(item["step"]) == step
                )
                for step in STEPS
            }
            for pair, prepared in zip(pairs, prepared_pairs, strict=True)
        },
        "straight_through_scope": "anchor + (simulated_endpoint - anchor), compared with saved simulated endpoint; not native training inner endpoints",
    }
    out_dir = run / "source/precision-audit/results"
    _atomic_csv(out_dir / "precision-audit.csv", all_rows)
    _atomic_json(
        out_dir / "precision-audit.json",
        {
            "state": "complete",
            "schema_version": 1,
            "provenance": provenance,
            "rows": all_rows,
        },
    )
    print(
        json.dumps(
            {"state": "complete", "rows": len(all_rows), "output": str(out_dir)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
