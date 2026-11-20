"""Inspect an existing CUDA prefix and optionally exercise one Matrix GPU."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    versions = {}
    for name in (
        "torch", "numpy", "safetensors", "transformers", "accelerate",
        "datasets", "axolotl", "lm_eval", "matplotlib", "scipy", "pandas",
        "pillow", "tokenizers", "peft", "trl", "wandb",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "state": "inventory", "host": socket.gethostname(),
        "python": platform.python_version(), "prefix": sys.prefix,
        "versions": versions, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "created_epoch": time.time(),
    }
    if args.gpu:
        import torch

        assert result["slurm_job_id"], "GPU probe requires a launcher allocation"
        assert torch.version.cuda == "12.6" and torch.version.hip is None
        assert torch.cuda.device_count() == 1
        properties = torch.cuda.get_device_properties(0)
        result["device"] = {
            "name": properties.name, "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        sys.argv = ["matrix_cuda_probe.py", "--expected_visible_gpus=1"]
        runpy.run_path("/p/vast1/kirchenb/llnl-tools/matrix_cuda_probe.py", run_name="__main__")
        left = torch.ones((256, 256), device="cuda:0", dtype=torch.bfloat16)
        product = left @ left
        assert torch.equal(product, torch.full_like(product, 256))
        torch.cuda.synchronize()
        result["bf16_matmul"] = "passed"
        result["state"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
