"""Time the original CPU-anchor and peer layouts on the archived RIO K16 grid."""

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import resource
import runpy
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from runtime import BASELINE, PEER, SOURCE, matrix_runtime, sha256, source_hashes


ARCHIVE = Path(
    "artifacts/tuolumne/tar-rio/20261119T1620Z/matrix-assessment/checkpoint-probe"
).resolve()
CE_TOLERANCE = 1e-4
MIN_SPEEDUP = 2.0
DEVICES = ("cuda:0", "cuda:1")


def topology():
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def file_identity(path):
    path = path.resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def tensor_sha256(tensors):
    """Hash tensor names, layout and raw evaluated bytes one tensor at a time."""
    digest = hashlib.sha256()
    for name, tensor in tensors.items():
        digest.update(f"{name}:{tensor.dtype}:{list(tensor.shape)}\n".encode())
        host_bytes = tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy()
        digest.update(memoryview(host_bytes))
    return digest.hexdigest()


def synchronize():
    for device in DEVICES:
        torch.cuda.synchronize(device)


def reset_peaks():
    for device in DEVICES:
        torch.cuda.reset_peak_memory_stats(device)


def memory():
    return {
        "cuda": {
            device: {
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            for device in DEVICES
        },
        "peak_host_rss_kib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def fixed_probes(settings, checkpoint, helper_path):
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    helper = runpy.run_path(str(helper_path))
    dataset = "cais/wmdp-bio-forget-corpus"
    fixed = helper["get_dataloader"](
        dataset, tokenizer, 512, 2, settings["probe_documents"]
    )
    if settings.get("fixed_source_indices") is not None:
        corpus = load_dataset(
            dataset, split="train", revision=settings["forget_revision"]
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        rows = [
            tokenizer(corpus[index]["text"], max_length=512, truncation=True)
            for index in settings["fixed_source_indices"]
        ]
        fixed = [collator(rows[i : i + 2]) for i in range(0, len(rows), 2)]
    return [
        {name: tensor.to(DEVICES[0]) for name, tensor in batch.items()}
        for batch in fixed
    ]


@torch.inference_mode()
def measure(model, parameters, perturb, anchor, basis, offsets, points, probes):
    started = time.perf_counter()

    def point_loss(point):
        perturb(parameters, anchor, basis, offsets, *point)
        return float(np.mean([float(model(**batch).loss) for batch in probes]))

    # Both directions contribute at this point. Compare every evaluated parameter.
    check_point = points[2]
    point_loss(check_point)
    synchronize()
    warmed = time.perf_counter()
    evaluated_hash = tensor_sha256(parameters)
    synchronize()
    hashed = time.perf_counter()
    rows = {}
    for point in points:
        synchronize()
        start = time.perf_counter()
        loss = point_loss(point)
        synchronize()
        elapsed = time.perf_counter() - start
        rows[point] = {
            "ce_nats": loss if math.isfinite(loss) else None,
            "seconds": elapsed,
        }
        print(f"Point {point}: CE {loss:.8f}, {elapsed:.3f} s", flush=True)
    return rows, {
        "warmup_seconds": warmed - started,
        "parameter_digest_seconds": hashed - warmed,
        "perturb_and_forward_seconds": sum(row["seconds"] for row in rows.values()),
        "measured_phase_wall_seconds": time.perf_counter() - started,
        "parameter_digest_point": check_point,
        "evaluated_parameters_sha256": evaluated_hash,
        "memory": memory(),
    }


def run(output):
    started = time.perf_counter()
    runtime_sources = source_hashes()
    pilot_sources = {name: sha256(SOURCE / name) for name in ("pilot.py", "pilot.sh")}
    runtime = matrix_runtime()
    runtime["nvidia_smi_topology"] = topology()
    print(json.dumps(runtime, indent=2), flush=True)

    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    old_result = ARCHIVE / "result.json"
    settings_path = ARCHIVE / "study.json"
    old_comparison = ARCHIVE / "comparison.csv"
    geometry_path = ARCHIVE / "geometry/geometry.json"
    helper_path = ARCHIVE / "source/misc/reference_get_ll.py"
    settings = json.loads(settings_path.read_text())
    assert settings == json.loads(old_result.read_text())["settings"]
    assert settings["case_name"] == "hparam-rio-lr1e-6"
    assert settings["cpu_landscape_anchor"] is True
    assert settings["broad_defense_frames"] == [] and settings[
        "broad_attack_frames"
    ] == [0]
    coordinates = np.linspace(
        -settings["broad_radius"], settings["broad_radius"], settings["broad_points"]
    )
    points = [(c0, c1) for c0 in coordinates for c1 in coordinates]
    with old_comparison.open() as stream:
        reference = {
            (float(row["c0"]), float(row["c1"])): float(row["probe_loss"])
            for row in csv.DictReader(stream)
            if row["phase"] == "attack" and int(row["step"]) == 0
        }
    assert len(points) == 9 and set(points) == set(reference)
    assert all(math.isfinite(loss) for loss in reference.values())
    offsets = json.loads(geometry_path.read_text())["offsets"]
    checkpoint = Path(settings["attack_anchor"]).resolve(strict=True)
    basis_path = ARCHIVE / "geometry/random-basis.npy"
    assert basis_path.resolve(strict=True) == Path(
        settings["shared_random_basis"]
    ).resolve(strict=True)
    baseline = runpy.run_path(str(BASELINE))
    peer = runpy.run_path(str(PEER))
    assert peer["_landscape_storage"](settings) == (DEVICES[1], False)

    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="eager"
        )
        .to(DEVICES[0])
        .eval()
    )
    parameters = dict(model.named_parameters())
    assert all(str(parameter.device) == DEVICES[0] for parameter in parameters.values())
    probes = fixed_probes(settings, checkpoint, helper_path)
    probe_hash = tensor_sha256(
        {
            f"batch{index}/{name}": tensor
            for index, batch in enumerate(probes)
            for name, tensor in sorted(batch.items())
        }
    )
    index = checkpoint / "model.safetensors.index.json"
    mapping = (
        json.loads(index.read_text())["weight_map"]
        if index.exists()
        else {name: "model.safetensors" for name in parameters}
    )
    shards = {
        filename: checkpoint / filename for filename in sorted(set(mapping.values()))
    }
    input_paths = [basis_path, *shards.values()]
    input_identities = [file_identity(path) for path in input_paths]
    # Match the archived broad frame's explicit checkpoint load and model cast.
    with torch.inference_mode(), contextlib.ExitStack() as stack:
        handles = {
            filename: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for filename, path in shards.items()
        }
        for name, parameter in parameters.items():
            parameter.copy_(handles[mapping[name]].get_tensor(name).clone())
    mapped = np.load(basis_path, mmap_mode="r")
    synchronize()
    common_setup = {"seconds": time.perf_counter() - started, "memory": memory()}

    print("Measuring CPU anchor and GPU0 directions", flush=True)
    reset_peaks()
    setup_started = time.perf_counter()
    with torch.inference_mode():
        host_anchor = baseline["_copy_landscape_anchor"](parameters, cpu=True)
    basis = baseline["_basis_to_device"](mapped, DEVICES[0])
    synchronize()
    cpu_setup = time.perf_counter() - setup_started
    cpu_rows, cpu_mode = measure(
        model,
        parameters,
        baseline["_perturb_landscape"],
        host_anchor,
        basis,
        offsets,
        points,
        probes,
    )
    del basis
    gc.collect()
    torch.cuda.empty_cache()

    print("Measuring GPU0 anchor and GPU1 directions", flush=True)
    reset_peaks()
    setup_started = time.perf_counter()
    # The model still contains the final CPU-anchor grid point.
    with torch.inference_mode():
        for name, parameter in parameters.items():
            parameter.copy_(host_anchor[name])
        peer_anchor = peer["_copy_landscape_anchor"](parameters, cpu=False)
    del host_anchor
    basis = peer["_basis_to_device"](mapped, DEVICES[1])
    synchronize()
    peer_setup = time.perf_counter() - setup_started
    peer_rows, peer_mode = measure(
        model,
        parameters,
        peer["_perturb_landscape"],
        peer_anchor,
        basis,
        offsets,
        points,
        probes,
    )

    assert [file_identity(path) for path in input_paths] == input_identities
    assert source_hashes() == runtime_sources, "Runtime source changed during the pilot"
    assert {name: sha256(SOURCE / name) for name in pilot_sources} == pilot_sources
    comparisons = []
    for c0, c1 in points:
        cpu, candidate = cpu_rows[c0, c1], peer_rows[c0, c1]
        cpu_ce, peer_ce, old_ce = (
            cpu["ce_nats"],
            candidate["ce_nats"],
            reference[c0, c1],
        )
        comparisons.append(
            {
                "c0": float(c0),
                "c1": float(c1),
                "cpu_anchor": cpu,
                "peer": candidate,
                "abs_ce_error_nats": (
                    None if cpu_ce is None or peer_ce is None else abs(cpu_ce - peer_ce)
                ),
                "archived_cuda_ce_nats": old_ce,
                "cpu_anchor_vs_archive_abs_ce_error_nats": (
                    None if cpu_ce is None else abs(cpu_ce - old_ce)
                ),
            }
        )
    errors = [point["abs_ce_error_nats"] for point in comparisons]
    finite = all(error is not None for error in errors)
    maximum = max(errors) if finite else None
    speedup = (
        cpu_mode["perturb_and_forward_seconds"]
        / peer_mode["perturb_and_forward_seconds"]
    )
    gates = {
        "all_nine_pairs_finite_and_within_ce_tolerance": finite
        and maximum <= CE_TOLERANCE,
        "full_model_parameter_bytes_equal_at_warmup_point": (
            cpu_mode["evaluated_parameters_sha256"]
            == peer_mode["evaluated_parameters_sha256"]
        ),
        "grid_speedup_strictly_greater_than_minimum": speedup > MIN_SPEEDUP,
    }
    small_inputs = (
        old_result,
        settings_path,
        old_comparison,
        geometry_path,
        helper_path,
    )
    result = {
        "state": "complete" if all(gates.values()) else "do_not_switch",
        "gates": gates,
        "ce_tolerance_nats": CE_TOLERANCE,
        "maximum_absolute_ce_error_nats": maximum,
        "all_nine_ce_values_equal": finite and maximum == 0,
        "minimum_speedup_exclusive": MIN_SPEEDUP,
        "perturb_and_forward_speedup": speedup,
        "points": comparisons,
        "common_setup": common_setup,
        "cpu_anchor": {"setup_seconds": cpu_setup, **cpu_mode},
        "peer": {"setup_seconds": peer_setup, **peer_mode},
        "end_to_end_seconds": time.perf_counter() - started,
        "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime": runtime,
        "runtime_source_sha256": runtime_sources,
        "pilot_source_sha256": pilot_sources,
        "archived_input_sha256": {str(path): sha256(path) for path in small_inputs},
        "archived_large_input_identities": input_identities,
        "checkpoint": str(checkpoint),
        "fixed_probe_tensor_sha256": probe_hash,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    temporary = output / "result.json.tmp"
    temporary.write_text(serialized)
    temporary.replace(output / "result.json")
    print(serialized, flush=True)
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    raise SystemExit(run(parser.parse_args().output))
