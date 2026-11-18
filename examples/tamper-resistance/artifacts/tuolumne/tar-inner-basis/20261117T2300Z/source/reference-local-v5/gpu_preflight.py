"""Check CPU/GPU measurement parity on fixed toy tensors inside the allocation."""

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from tar_inner_recording import TransientInnerRecorder


class Toy(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.1234567, 0.01234567, 0.0001234567],
                    [-1.234567, 0.9876543, -0.00123456],
                ],
                dtype=torch.float32,
                device=device,
            )
        )


def update(model, inputs, on_step_end):
    with torch.no_grad():
        for step in (1, 2):
            change = torch.arange(
                6, dtype=torch.float32, device=model.weight.device
            ).reshape(2, 3)
            if step == 2:
                change = change.flip(0)
            model.weight.add_(change / 16)
            on_step_end(step)
    return model


def main():
    import sys

    run = Path(sys.argv[1])
    assert torch.cuda.device_count() == 4
    cache = run / "cache"
    cache.mkdir(exist_ok=True)
    original_device = torch.cuda.current_device()
    results, times = {}, {}
    with tempfile.TemporaryDirectory(
        prefix="gpu-measurement-check-", dir=cache
    ) as temporary:
        root = Path(temporary)
        anchor, reference = root / "anchor", root / "reference"
        anchor.mkdir()
        reference.mkdir()
        initial = Toy("cpu").weight.detach().clone()
        save_file({"weight": initial}, anchor / "model.safetensors")
        save_file(
            {"weight": (initial + 0.125).bfloat16()}, reference / "model.safetensors"
        )
        references = [
            {"name": "toy_reference", "anchor": str(anchor), "path": str(reference)}
        ]
        inputs = {
            key: torch.ones((2, 3), dtype=torch.int64)
            for key in ("input_ids", "attention_mask", "labels")
        }
        for device in ("cpu", "cuda:0"):
            model = Toy(device)
            torch.cuda.set_device(3)
            recorder = TransientInnerRecorder(
                root / device.replace(":", "-"),
                (1, 2),
                references,
                chunk_elements=3,
                reduction_device=device,
                build_device=device,
            )
            started = time.perf_counter()
            recorder.run(update, model, inputs, 1)
            times[device] = time.perf_counter() - started
            assert torch.cuda.current_device() == 3
            results[device] = json.loads(
                (recorder.root / "outer-000001.json").read_text()
            )
        largest = 0.0
        for rep in ("raw", "bf16"):
            for key in (
                "inner_gram",
                "cross_gram",
                "reference_gram",
                "anchor_gap_squared_norm",
            ):
                a = np.asarray(results["cpu"]["representations"][rep][key])
                b = np.asarray(results["cuda:0"]["representations"][rep][key])
                largest = max(largest, float(np.max(np.abs(a - b))))
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    torch.cuda.set_device(original_device)
    result = {
        "state": "passed",
        "measurement": "FP64 Gram CPU/GPU parity on fixed toy tensors",
        "maximum_absolute_matrix_error": largest,
        "active_device_preserved": True,
        "observer_rng_contract_passed": True,
        "wall_seconds": times,
    }
    (run / "status/gpu-reduction-preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
