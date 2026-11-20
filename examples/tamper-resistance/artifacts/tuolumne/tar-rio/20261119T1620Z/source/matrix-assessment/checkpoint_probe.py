"""Compare nine RIO random-plane points with the completed Tuolumne control."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import time


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["phase"], int(row["step"]), float(row["c0"]), float(row["c1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamics", type=Path, required=True)
    args = parser.parse_args()
    import torch

    assert os.environ.get("SLURM_JOB_ID"), "Use launch_matrix.py"
    assert torch.__version__ == "2.12.1+cu126"
    assert torch.version.hip is None and torch.cuda.device_count() == 1
    # Separate outputs prevent backend results from silently entering Tuo caches.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "surfaces").mkdir()
    for name in ("source", "geometry"):
        (args.output / name).symlink_to((args.control / name).resolve())
    settings_path = args.control / "study.json"
    settings = json.loads(settings_path.read_text())
    settings.update(
        cpu_landscape_anchor=True,
        broad_points=3,
        broad_defense_frames=[],
        broad_attack_frames=[0],
    )
    (args.output / "study.json").write_text(json.dumps(settings, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("matrix_dynamics", args.dynamics)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Check that offloading changes storage only, including BF16 rounding.
    parameters = {
        "weight": torch.nn.Parameter(
            torch.tensor([1.01, -0.7], device="cuda", dtype=torch.bfloat16)
        )
    }
    basis = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], device="cuda")
    resident = module._copy_landscape_anchor(parameters)
    host = module._copy_landscape_anchor(parameters, cpu=True)
    for c0, c1 in ((0.0, 0.0), (0.157, -0.08), (-125.0, 125.0)):
        module._perturb_landscape(parameters, resident, basis, {"weight": 0}, c0, c1)
        expected = parameters["weight"].detach().clone()
        module._perturb_landscape(parameters, host, basis, {"weight": 0}, c0, c1)
        assert torch.equal(
            parameters["weight"], expected
        ), "Offload changed perturbation arithmetic"
    del parameters, basis, resident, host, expected
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    module.random_frames(args.output, settings)
    torch.cuda.synchronize()
    actual = rows(args.output / "surfaces/random-evolution.csv")
    expected_path = args.control / "surfaces/random-evolution.csv"
    expected = {key(row): float(row["probe_loss"]) for row in rows(expected_path)}
    assert len(actual) == 9 and len({key(row) for row in actual}) == 9
    comparisons = []
    for row in actual:
        reference = expected[key(row)]
        measured = float(row["probe_loss"])
        assert math.isfinite(measured) and math.isfinite(reference)
        comparisons.append(
            {**row, "tuolumne_loss": reference, "abs_error": abs(measured - reference)}
        )
    with (args.output / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    maximum_error = max(row["abs_error"] for row in comparisons)
    # Fixed in advance; investigate disagreements rather than relaxing this gate.
    tolerance = 0.01
    result = {
        "state": "complete" if maximum_error <= tolerance else "failed",
        "scope": "Nine end-of-defense random-plane CE points; not training or trajectory equivalence",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cpu_anchor_gpu_rounding_check": "bitwise_equal",
        "points": 9,
        "absolute_ce_tolerance_nats": tolerance,
        "maximum_absolute_ce_error_nats": maximum_error,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_sha256": hashlib.sha256(args.dynamics.read_bytes()).hexdigest(),
        "control_settings_sha256": hashlib.sha256(
            settings_path.read_bytes()
        ).hexdigest(),
        "control_reference_csv_sha256": hashlib.sha256(
            expected_path.read_bytes()
        ).hexdigest(),
        "settings": settings,
    }
    target = args.output / "result.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps(result, indent=2), flush=True)
    assert result["state"] == "complete", "Cross-backend CE comparison failed"


if __name__ == "__main__":
    main()
