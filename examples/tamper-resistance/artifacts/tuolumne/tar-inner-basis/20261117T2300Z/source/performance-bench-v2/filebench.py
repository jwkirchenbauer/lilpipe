"""Compare pageable host inputs to an identical synthetic TAR Gram calculation."""

import argparse
import gc
import hashlib
import json
import os
import resource
import socket
import statistics
import struct
import tempfile
import time
from pathlib import Path

import torch
from safetensors import safe_open

import bench as v1
import tar_inner_recording as recorder


def counters():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    io = dict(
        line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
    )
    return {
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "storage_read_bytes": int(io["read_bytes"]),
    }


def phase(started, before):
    duration = 1000 * (time.perf_counter() - started)
    return {
        "wall_ms": duration,
        **{k: val - before[k] for k, val in counters().items()},
    }


def sparse_file(path, values, width):
    # Keep every possible mapped read within 127.5 MiB even on tmpfs kernels that
    # allocate physical pages on reads from sparse holes. Tail padding is untouched.
    length = width + width // 4
    offset = width // 8
    logical = width * 1024
    header, pos = {}, 0
    for i, tensor in enumerate(values):
        size = length * tensor.element_size()
        header[str(i)] = {
            "dtype": "BF16" if tensor.dtype == torch.bfloat16 else "F32",
            "shape": [length],
            "data_offsets": [pos, pos + size],
        }
        pos += size
    header["padding"] = {
        "dtype": "U8",
        "shape": [logical - pos],
        "data_offsets": [pos, logical],
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        start = stream.tell()
        stream.truncate(start + logical)
        for i, tensor in enumerate(values):
            stream.seek(
                start
                + header[str(i)]["data_offsets"][0]
                + offset * tensor.element_size()
            )
            stream.write(memoryview(tensor.view(torch.uint8).numpy()))
        stream.flush()
        os.fsync(stream.fileno())
    return slice(offset, offset + width), sum(
        t.numel() * t.element_size() for t in values
    )


def mapping_info(values, backing):
    mappings = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        a, b = (int(x, 16) for x in fields[0].split("-"))
        mappings.append((a, b, fields[5] if len(fields) > 5 else ""))
    matches = 0
    spans = set()
    for t in values:
        address = t.data_ptr()
        a, b, name = next(m for m in mappings if m[0] <= address < m[1])
        spans.add(b - a)
        matches += name == str(backing)
    return {
        "tensors": len(values),
        "mapped_to_backing": matches,
        "process_vma_sizes": sorted(spans),
        "tensor_storage_sizes": sorted({t.untyped_storage().nbytes() for t in values}),
    }


def whole_gram(observer, values, section, device):
    initial, archive, *rows = values
    saved = {
        "anchor": archive,
        **{
            ref["path"]: source
            for ref, source in zip(observer.references, rows[v1.INNER :], strict=True)
        },
    }
    grams = {rep: v1.Gram(device) for rep in v1.REPS}
    gaps = {rep: torch.zeros((), dtype=torch.float64, device=device) for rep in v1.REPS}
    gpu = device.type == "cuda"
    if gpu:
        torch.cuda.synchronize(device)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    before = counters()
    started = time.perf_counter()
    if gpu:
        begin.record()
    observer._accumulate_device_chunk(
        initial, rows[: v1.INNER], saved, section, grams, gaps
    )
    if gpu:
        end.record()
        end.synchronize()
    stats = phase(started, before)
    if gpu:
        stats["gpu_stream_span_ms"] = begin.elapsed_time(end)
        stats["gpu_gemm_ms"] = sum(
            g.events[0].elapsed_time(g.events[1]) for g in grams.values()
        )
        stats["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    outputs = {
        **{rep: g.value.cpu() for rep, g in grams.items()},
        **{rep + "_gap": g.cpu() for rep, g in gaps.items()},
    }
    return outputs, stats


def check(actual, expected):
    for key in expected:
        if not torch.equal(actual[key], expected[key]):
            raise ValueError(
                f"{key} differs by {float((actual[key]-expected[key]).abs().max())}"
            )


def file_trial(mode, path, recorder_obj, section, width, device, expected):
    before = counters()
    started = time.perf_counter()
    with safe_open(path, framework="pt", device="cpu") as reader:
        sources = [reader.get_tensor(str(i)) for i in range(v1.ROWS + 2)]
        opened = phase(started, before)
        before = counters()
        started = time.perf_counter()
        if mode == "full_clone":
            values = [t.clone() for t in sources]
            selection = section
        elif mode == "slice_clone":
            values = [t[section].clone() for t in sources]
            selection = slice(0, width)
        else:
            values, selection = sources, section
        materialized = phase(started, before)
        src_map = mapping_info(sources, path)
        input_map = src_map if values is sources else mapping_info(values, path)
        timings = []
        for _ in range(2):
            actual, stats = whole_gram(recorder_obj, values, selection, device)
            check(actual, expected)
            timings.append(stats)
        return {
            "open_and_get_tensor": opened,
            "materialization": materialized,
            "builds": timings,
            "source_mapping": src_map,
            "build_input_mapping": input_map,
        }


def median(rows, f):
    return statistics.median(f(r) for r in rows)


def run(width, device, roots, trials):
    assert (
        hashlib.sha256(Path(recorder.__file__).read_bytes()).hexdigest()
        == v1.SOURCE_HASH
    )
    initial, archive, rows = v1.endpoints(width)
    resident = [initial, archive, *rows]
    refs = [
        {"name": str(i), "anchor": "anchor", "path": str(i)}
        for i in range(v1.ROWS - v1.INNER)
    ]
    observer = recorder.TransientInnerRecorder(
        ".", [1], refs, reduction_device=device, build_device=device
    )
    expected, _ = whole_gram(observer, resident, slice(None), device)  # untimed warmup
    baseline = []
    for _ in range(trials):
        actual, stats = whole_gram(observer, resident, slice(None), device)
        check(actual, expected)
        baseline.append(stats)
    result = {
        "recorder_sha256": v1.SOURCE_HASH,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "v1_benchmark_sha256": hashlib.sha256(
            Path(v1.__file__).read_bytes()
        ).hexdigest(),
        "host": socket.gethostname(),
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "rows": v1.ROWS,
        "coordinates": width,
        "parameter_length": width + width // 4,
        "fresh_map_trials_per_mode": trials,
        "resident_builds": baseline,
        "resident_median_wall_ms": median(baseline, lambda r: r["wall_ms"]),
        "scope": "Generated sparse files, one map per fresh trial. Each trial builds twice from the same inputs. Files are populated immediately before the trials; kernel page cache is not dropped. The unselected tail padding stays untouched. Storage-read counters exclude cache hits. Production has more files, larger parameters and concurrent training allocations.",
        "tiers": {},
    }
    for tier, root in roots.items():
        with tempfile.TemporaryDirectory(prefix="tar-filebench-", dir=root) as folder:
            path = (Path(folder) / "synthetic.safetensors").resolve()
            section, written = sparse_file(path, resident, width)
            record = {
                "root": str(root),
                "file_logical_bytes": path.stat().st_size,
                "tensor_bytes_written": written,
                "allocated_bytes_after_write": path.stat().st_blocks * 512,
                "modes": {k: [] for k in ("direct", "full_clone", "slice_clone")},
            }
            modes = list(record["modes"])
            for i in range(trials):
                # Rotate the first method so each has an opportunity at first touch.
                for mode in modes[i % 3 :] + modes[: i % 3]:
                    gc.collect()
                    trial = file_trial(
                        mode, path, observer, section, width, device, expected
                    )
                    record["modes"][mode].append(trial)
                    print(
                        tier,
                        mode,
                        i + 1,
                        "build_ms",
                        [round(s["wall_ms"], 3) for s in trial["builds"]],
                        flush=True,
                    )
            summaries = {}
            for mode, records in record["modes"].items():
                summaries[mode] = {
                    "get_tensor_ms": median(
                        records, lambda r: r["open_and_get_tensor"]["wall_ms"]
                    ),
                    "materialize_ms": median(
                        records, lambda r: r["materialization"]["wall_ms"]
                    ),
                    "first_build_ms": median(
                        records, lambda r: r["builds"][0]["wall_ms"]
                    ),
                    "repeat_build_ms": median(
                        records, lambda r: r["builds"][1]["wall_ms"]
                    ),
                    "first_total_ms": median(
                        records,
                        lambda r: r["open_and_get_tensor"]["wall_ms"]
                        + r["materialization"]["wall_ms"]
                        + r["builds"][0]["wall_ms"],
                    ),
                }
            record.update(
                medians=summaries,
                allocated_bytes_after_read=path.stat().st_blocks * 512,
                exact_gram_and_gap_parity=True,
            )
            if (
                width == v1.WIDTH
                and max(
                    record["allocated_bytes_after_write"],
                    record["allocated_bytes_after_read"],
                )
                > 128 * 1024**2
            ):
                raise ValueError(
                    "Sparse backing exceeded its 128 MiB physical allocation budget"
                )
            result["tiers"][tier] = record
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("ssd", "ram", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--cpu-check", action="store_true")
    args = p.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    with torch.no_grad():
        if args.cpu_check:
            with tempfile.TemporaryDirectory() as root:
                result = run(256, torch.device("cpu"), {"smoke": Path(root)}, 1)
        else:
            if not args.ssd or not args.ram or not args.output:
                p.error("GPU benchmark requires --ssd, --ram and --output")
            if args.ssd.stat().st_dev == args.ram.stat().st_dev:
                raise ValueError("SSD and RAM roots resolve to the same filesystem")
            assert (
                torch.cuda.device_count() == 1
                and torch.backends.cuda.preferred_blas_library()
                == torch._C._BlasBackend.Cublas
            )
            result = run(
                v1.WIDTH,
                torch.device("cuda:0"),
                {"ssd": args.ssd, "tmpfs": args.ram},
                3,
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
