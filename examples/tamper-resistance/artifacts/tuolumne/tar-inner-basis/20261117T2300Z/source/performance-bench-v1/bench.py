"""Benchmark one resident synthetic chunk for three TAR Gram measurement paths."""

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
import tar_inner_recording as recorder


ROWS = 32
INNER = 6
WIDTH = 1_048_576
REPS = ("raw", "bf16")
SOURCE_HASH = "94b2a3a77e89c90ec034ecece17e4344a98b149dae8a0399b9df6f47e7647d90"


def endpoints(width):
    rng = torch.Generator().manual_seed(1749)
    tensors = []
    for i in range(ROWS + 2):
        # A dot of up to 2**20 bounded dyadic products fits exactly into FP64.
        values = torch.randint(-2048, 2048, (width,), generator=rng, dtype=torch.int16)
        tensors.append(
            (values.float() / 2048).to(torch.bfloat16 if i % 2 else torch.float32)
        )
    initial, archived, *values = tensors
    return initial, archived, values


def cast(values, rep):
    return values.to(torch.bfloat16).float() if rep == "bf16" else values.float()


def cpu_build(initial, archived, values, device, grams, gaps):
    # Same endpoint arithmetic/allocation order as the frozen CPU-build branch.
    block = torch.empty((ROWS, initial.numel()), dtype=torch.float64)
    for rep in REPS:
        base = cast(initial, rep)
        for i, source in enumerate(values[:INNER]):
            block[i].copy_(cast(source, rep) - base)
        refs = [cast(source, rep) for source in [archived, *values[INNER:]]]
        gap = (base - refs[0]).double()
        gaps[rep] += torch.dot(gap, gap).item()
        for i, source in enumerate(refs[1:], start=INNER):
            block[i].copy_(source - refs[0])
        transferred = block.to(device)
        grams[rep].addmm_(transferred, transferred.T)
        del transferred


def packed_build(initial, archived, values, device, grams, gaps):
    # copy_ converts native FP32/BF16 in one pass into a contiguous host buffer.
    host = torch.empty((ROWS + 2, initial.numel()), dtype=torch.float32)
    for i, source in enumerate([initial, archived, *values]):
        host[i].copy_(source)
    packed = host.to(device, non_blocking=True)
    for rep in ("bf16", "raw"):
        coords = cast(packed, rep)
        inner_base, actual_base = coords[0], coords[1]
        updates = coords[2:]
        updates[:INNER].sub_(inner_base)
        updates[INNER:].sub_(actual_base)
        gap = (inner_base - actual_base).double()
        gaps[rep].add_(torch.dot(gap, gap))
        del gap
        block = updates.double()
        grams[rep].addmm_(block, block.T)
        del block, coords, updates, inner_base, actual_base


class Gram:
    """Capture elapsed device events around the two actual FP64 GEMMs."""

    def __init__(self, device):
        self.value = torch.zeros((ROWS, ROWS), dtype=torch.float64, device=device)
        self.events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            if device.type == "cuda"
            else []
        )

    def addmm_(self, left, right):
        if self.events:
            self.events[0].record()
        self.value.addmm_(left, right)
        if self.events:
            self.events[1].record()


def standalone_gram(values, device, repetitions):
    """Time full and explicitly split reductions with inputs already on device."""
    block = torch.empty((ROWS, values[0].numel()), dtype=torch.float64, device=device)
    for i, source in enumerate(values):
        block[i].copy_(source)
    gpu = device.type == "cuda"
    report = {}
    expected = None
    for splits in (1, 64, 128):
        samples = []
        if gpu:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for i in range(repetitions + 1):
            if gpu:
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if gpu:
                begin.record()
            if splits == 1:
                gram = block @ block.T
            else:
                tiles = block.reshape(ROWS, splits, -1).transpose(0, 1)
                gram = torch.bmm(tiles, tiles.transpose(1, 2)).sum(0)
            if gpu:
                end.record()
                end.synchronize()
            elapsed = 1000 * (time.perf_counter() - started)
            sample = {"wall_ms": elapsed}
            if gpu:
                sample["gpu_stream_span_ms"] = begin.elapsed_time(end)
            if i:
                samples.append(sample)
            if expected is None:
                expected = gram.cpu()
            actual = gram.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            max_error = float((actual - expected).abs().max())
        medians = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        report[str(splits)] = {
            "samples": samples,
            "median": medians,
            "maximum_absolute_gram_error": max_error,
            "tile_width": block.shape[1] // splits,
        }
        if gpu:
            report[str(splits)]["peak_gpu_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
        print("standalone_gram", splits, json.dumps(medians), flush=True)
    return report


def run(width, repetitions, device):
    source_hash = hashlib.sha256(Path(recorder.__file__).read_bytes()).hexdigest()
    if source_hash != SOURCE_HASH:
        raise ValueError(f"Unexpected frozen recorder {source_hash}")
    initial, archive, values = endpoints(width)
    refs = [
        {"name": str(i), "anchor": "anchor", "path": str(i)}
        for i in range(ROWS - INNER)
    ]
    saved = {
        "anchor": archive,
        **{
            ref["path"]: source
            for ref, source in zip(refs, values[INNER:], strict=True)
        },
    }
    observer = recorder.TransientInnerRecorder(
        ".", [1], refs, reduction_device=device, build_device=device
    )
    gpu = device.type == "cuda"

    def native_build(initial, archive, values, device, grams, gaps):
        observer._accumulate_device_chunk(
            initial, values[:INNER], saved, slice(None), grams, gaps
        )

    modes = {
        "cpu_build_gpu_gram": cpu_build,
        "native_slice_gpu_build": native_build,
        "packed_fp32_host_gpu_build": packed_build,
    }
    result = dict(
        recorder_sha256=source_hash,
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=str(torch.__version__),
        hip_version=torch.version.hip,
        cpu_threads=torch.get_num_threads(),
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if gpu else "CPU check",
        rows=ROWS,
        inner_rows=INNER,
        coordinates=width,
        measured_repetitions=repetitions,
        synthetic_endpoint_bytes=sum(
            t.numel() * t.element_size() for t in [initial, archive, *values]
        ),
        timing="Wall and GPU stream span both include time the host can leave the stream idle. GEMM event times isolate the two queued matrix products. One untimed warmup precedes each mode.",
        scope="Generated resident CPU storage. This test measures neither checkpoint paging nor model/optimizer memory contention.",
        modes={},
    )
    reference = {}
    for mode, build in modes.items():
        gc.collect()
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for rep_no in range(repetitions + 1):
            grams = {k: Gram(device) for k in REPS}
            gaps = {
                k: (
                    0.0
                    if build is cpu_build
                    else torch.zeros((), dtype=torch.float64, device=device)
                )
                for k in REPS
            }
            if gpu:
                torch.cuda.synchronize(device)
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            started = time.perf_counter()
            if gpu:
                begin.record()
            build(initial, archive, values, device, grams, gaps)
            if gpu:
                end.record()
                end.synchronize()
            elapsed = 1000 * (time.perf_counter() - started)
            sample = {"wall_ms": elapsed}
            if gpu:
                sample["gpu_stream_span_ms"] = begin.elapsed_time(end)
                sample["gpu_gemm_ms"] = sum(
                    g.events[0].elapsed_time(g.events[1]) for g in grams.values()
                )
            if rep_no:
                samples.append(sample)
            actual = {
                **{k: g.value.cpu() for k, g in grams.items()},
                **{
                    k + "_gap": torch.as_tensor(float(val), dtype=torch.float64)
                    for k, val in gaps.items()
                },
            }
            if not reference:
                reference = actual
            for key in reference:
                if not torch.equal(reference[key], actual[key]):
                    error = (reference[key] - actual[key]).abs().max().item()
                    raise ValueError(f"{mode} {key} parity failed: {error}")
        medians = {k: statistics.median(s[k] for s in samples) for k in samples[0]}
        result["modes"][mode] = {
            "samples": samples,
            "median": medians,
            "exact_gram_and_gap_parity": True,
        }
        if gpu:
            result["modes"][mode]["peak_gpu_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
            result["modes"][mode]["peak_gpu_reserved_bytes"] = (
                torch.cuda.max_memory_reserved(device)
            )
        print(mode, json.dumps(medians), flush=True)
    if gpu:
        torch.cuda.empty_cache()
    result["standalone_gram_split_k"] = standalone_gram(values, device, repetitions)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--cpu-check",
        action="store_true",
        help="Check synthetic formulas on 256 coordinates",
    )
    args = p.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if not args.cpu_check:
        if not args.output:
            p.error("GPU benchmark requires --output")
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert (
            torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
        )
    device = torch.device("cpu" if args.cpu_check else "cuda:0")
    with torch.no_grad():
        result = run(
            256 if args.cpu_check else WIDTH, 1 if args.cpu_check else 5, device
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix(".partial")
        with temp.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.link(temp, args.output)
        temp.unlink()


if __name__ == "__main__":
    main()
