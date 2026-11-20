"""Check schedule/data identity and quantify CUDA-versus-ROCm K4 updates."""

import json
import math
from pathlib import Path
import sys

import torch
from safetensors import safe_open


ORIGIN = Path(
    "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-rio/20261119T1620Z"
)


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    root = Path(sys.argv[1]) / "pilots/cuda-training"
    original = ORIGIN / "cases/rio-k4-defense/models/defense"
    cuda = root / "k4/models/defense"
    reference = read_lines(original / "observed-batches.jsonl")[:4]
    actual = read_lines(cuda / "observed-batches.jsonl")
    assert [r["step"] for r in actual] == [1, 2, 3, 4]
    assert [r["batch"] for r in actual] == [r["batch"] for r in reference]
    assert [r["learning_rate_used"] for r in actual] == [
        r["learning_rate_used"] for r in reference
    ]
    loss_error = max(
        abs(a["loss_before_update"] - b["loss_before_update"])
        for a, b in zip(actual, reference)
    )
    statistics = {
        k: []
        for k in ("reference_squared", "cuda_squared", "difference_squared", "dot")
    }
    with (
        safe_open(original / "checkpoint-1/model.safetensors", framework="pt") as base,
        safe_open(cuda / "checkpoint-1/model.safetensors", framework="pt") as initial,
        safe_open(original / "checkpoint-4/model.safetensors", framework="pt") as old,
        safe_open(cuda / "checkpoint-4/model.safetensors", framework="pt") as new,
    ):
        assert (
            set(base.keys())
            == set(initial.keys())
            == set(old.keys())
            == set(new.keys())
        )
        for name in base.keys():
            b, a, x, y = [
                handle.get_tensor(name).clone() for handle in (base, initial, old, new)
            ]
            assert (
                b.dtype == a.dtype == x.dtype == y.dtype
                and b.shape == a.shape == x.shape == y.shape
            )
            assert torch.equal(b, a), "Initial saved weights differ: " + name
            b, x, y = [v.reshape(-1) for v in (b, x, y)]
            for start in range(0, b.numel(), 1024 * 1024):
                stop = start + 1024 * 1024
                anchor, left, right = [
                    v[start:stop].to(device="cuda:0", dtype=torch.float64)
                    for v in (b, x, y)
                ]
                dl, dr = left - anchor, right - anchor
                for field, value in (
                    ("reference_squared", dl.square().sum()),
                    ("cuda_squared", dr.square().sum()),
                    ("difference_squared", (right - left).square().sum()),
                    ("dot", (dl * dr).sum()),
                ):
                    statistics[field].append(float(value))
    totals = {k: math.fsum(v) for k, v in statistics.items()}
    reference_norm, cuda_norm = [
        math.sqrt(totals[k]) for k in ("reference_squared", "cuda_squared")
    ]
    relative = (
        math.sqrt(totals["difference_squared"]) / reference_norm
        if reference_norm
        else None
    )
    cosine = (
        totals["dot"] / (reference_norm * cuda_norm)
        if reference_norm and cuda_norm
        else None
    )
    large = root / "k256/models/defense"
    records = read_lines(large / "observed-batches.jsonl")
    assert len(records) == 1 and records[0]["batch"] == reference[0]["batch"]
    assert records[0]["learning_rate_used"] == 0
    state = json.loads((large / "checkpoint-1/trainer_state.json").read_text())
    inner = [
        r["inner_step_total"] for r in state["log_history"] if "inner_step_total" in r
    ]
    assert inner == list(range(256)), "K256 simulation budget changed"
    assert all(math.isfinite(r["loss_before_update"]) for r in actual + records)
    # Generous migration checks, declared before this first CUDA training pilot.
    # Cross-backend update equality is not required; same data/schedules are.
    passed = (
        loss_error <= 0.05
        and relative is not None
        and relative <= 0.25
        and cosine >= 0.95
    )
    result = {
        "state": "complete" if passed else "failed",
        "initial_weights_bitwise_equal": True,
        "outer_batches_and_rates_exact": True,
        "max_loss_error": loss_error,
        "reference_update_l2": reference_norm,
        "cuda_update_l2": cuda_norm,
        "relative_update_error": relative,
        "update_cosine": cosine,
        "declared_migration_limits": {
            "max_loss_error": 0.05,
            "relative_update_error": 0.25,
            "minimum_cosine": 0.95,
        },
        "k256_inner_updates": len(inner),
        "k256_outer_step_seconds": records[0]["forward_backward_seconds"],
        "k256_125_step_time_estimate_seconds": 125
        * records[0]["forward_backward_seconds"],
        "k4_memory": json.loads((cuda / "matrix-runtime.json").read_text()),
        "k256_memory": json.loads((large / "matrix-runtime.json").read_text()),
        "scope": "Short prefixes of unchanged 125-step schedules; approximate cross-backend training agreement, not bitwise trajectory equality",
    }
    target = root / "status/validation.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    assert passed, "Training migration pilot is outside its predeclared limits"


if __name__ == "__main__":
    main()
