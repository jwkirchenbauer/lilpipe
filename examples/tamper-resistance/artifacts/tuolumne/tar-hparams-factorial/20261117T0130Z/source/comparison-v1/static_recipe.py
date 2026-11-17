"""Original static-recipe worker; numerical and rendering processes are separate."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RADII = [1000, 500, 200, 125, 100, 10, 1, 0.1]
DATASET = "cais/wmdp-bio-forget-corpus"
CONVENTION = (
    "Original seed42 independent globally normalized BF16 random directions, "
    "sequential BF16 perturbations and default attention. These are separate "
    "from shared FP32 orthonormal dynamics bases with eager attention."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_identity(folder):
    result = []
    for path in sorted(Path(folder).glob("*")):
        if path.is_file() and (
            path.suffix in {".safetensors", ".json", ".model"}
            or path.name.startswith("tokenizer")
        ):
            stat = path.stat()
            result.append(
                {
                    "path": str(path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    assert any(item["path"].endswith(".safetensors") for item in result)
    return result


def training_proof(run, name):
    program = read(run / "study.json")
    assert program["kind"] == "fixed-sampler-hparam-study"
    case = program["cases"][name]
    assert case["role"] == "defense" and case["max_steps"] == 125
    folder = run / "cases" / name
    assert read(folder / "status/train.json")["state"] == "complete"
    proof = read(folder / "training-validation.json")
    assert proof["state"] == "passed" and proof["updates"] == 125
    assert proof["input_order_validation"]["state"] == "passed"
    assert proof["input_order_validation"]["batches"] == 125
    assert len(proof["input_order_validation"]["canonical_step_batch_sha256"]) == 64
    return program, case, folder


def validate_grid(path):
    with Path(path).open(newline="") as stream:
        rows = [
            {key: float(row[key]) for key in ("radius", "c0", "c1", "loss")}
            for row in csv.DictReader(stream)
        ]
    assert len(rows) == 648 and all(
        math.isfinite(value) for row in rows for value in row.values()
    )
    assert len({(row["radius"], row["c0"], row["c1"]) for row in rows}) == 648
    centers = []
    for radius in RADII:
        selected = [row for row in rows if row["radius"] == radius]
        assert len(selected) == 81
        for coordinate in ("c0", "c1"):
            values = sorted({row[coordinate] for row in selected})
            assert len(values) == 9
            assert all(
                abs(value - (-radius + index * radius / 4)) <= 1e-5 * max(1, radius)
                for index, value in enumerate(values)
            )
        center = [row["loss"] for row in selected if row["c0"] == row["c1"] == 0]
        assert len(center) == 1
        centers.extend(center)
    assert max(centers) - min(centers) < 1e-5
    return rows


def check_protocol(previous, current):
    if previous != current:
        raise ValueError(
            "Static inputs/protocol changed; do not reuse or overwrite this grid"
        )


def prepare_inputs(run, name, code):
    import torch
    import yaml
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling

    program, case, folder = training_proof(run, name)
    config = yaml.safe_load((folder / "config.yml").read_text())
    model = Path(config["output_dir"]).resolve()
    assert model == (folder / "models/defense").resolve()
    probes = read(run / "probes.json")
    original = (
        Path(program["original_run"])
        / "loss-landscape/defense-step125-random-seed42-v3"
    )
    prior = read(original / "provenance.json")
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
    assert torch.__version__ == prior["torch"] and torch.version.hip == prior["hip"]
    corpus = load_dataset(DATASET, split="train", revision=probes["forget_revision"])
    texts = [corpus[index]["text"] for index in probes["fixed_source_indices"]]
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    assert len(texts) == 10 and hashes == prior["document_sha256"]
    default = load_dataset(DATASET, split="train")
    original_texts = list(
        itertools.islice((row["text"] for row in default if len(row["text"]) > 50), 10)
    )
    assert (
        original_texts == texts
    ), "Original get_ll default dataset does not select the frozen fixed10"

    def batches(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoded = [tokenizer(text, truncation=True, max_length=512) for text in texts]
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        return [
            {key: tensor.tolist() for key, tensor in batch.items()}
            for batch in DataLoader(
                encoded, batch_size=2, shuffle=False, collate_fn=collator
            )
        ]

    evaluated_batches = batches(model)
    baseline_model = Path(program["original_run"]) / "models/defense"
    assert evaluated_batches == batches(
        baseline_model
    ), "Static tokenizer/collation differs from original baseline"
    sources = {
        filename: sha(code / filename)
        for filename in ("get_ll.py", "plot_ll_2d.py", "plot_ll_3d.py")
    }
    assert sources == {
        filename: prior["source_sha256"][filename] for filename in sources
    }
    protocol = {
        "version": 1,
        "case": name,
        "condition": case["condition"],
        "checkpoint": str(model),
        "checkpoint_files": file_identity(model),
        "defense_config_sha256": sha(folder / "config.yml"),
        "training_validation_sha256": sha(folder / "training-validation.json"),
        "source_sha256": sources,
        "worker_sha256": sha(code / "static_worker.py"),
        "dataset": DATASET,
        "dataset_revision": probes["forget_revision"],
        "fixed_source_indices": probes["fixed_source_indices"],
        "document_sha256": hashes,
        "default_dataset_fixed10_matches_pinned": True,
        "evaluated_batch_sha256": [canonical_sha(batch) for batch in evaluated_batches],
        "baseline_tokenizer_and_collation_match": True,
        "radii": RADII,
        "n_pts": 9,
        "seed": 42,
        "batch_size": 2,
        "max_seq_len": 512,
        "probe_documents": 10,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "convention": CONVENTION,
        "original_static": str(original),
    }
    output = run / "static/cases" / name
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        check_protocol(read(output / "protocol.json"), protocol)
    else:
        assert not (output / "grid.csv").exists(), "Unbound existing numerical grid"
        dump(output / "protocol.json", protocol)
    dump(
        output / "provenance.json",
        {
            **protocol,
            "flux_job_id": os.environ["FLUX_JOB_ID"],
            "gpu": torch.cuda.get_device_name(0),
            "validated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(protocol, indent=2), flush=True)


def run_case(run, name, code):
    _, _, folder = training_proof(run, name)
    output = run / "static/cases" / name
    status = folder / "status/static.json"
    started = time.monotonic()
    dump(status, {"state": "running", "stage": "inputs"})
    try:
        subprocess.run(
            [sys.executable, str(code / "static_worker.py"), "inputs", str(run), name],
            check=True,
        )
        protocol = read(output / "protocol.json")
        if (output / "grid.csv").exists():
            validate_grid(output / "grid.csv")
        else:
            dump(status, {"state": "running", "stage": "values"})
            subprocess.run(
                [
                    sys.executable,
                    str(code / "get_ll.py"),
                    "--model",
                    protocol["checkpoint"],
                    "--dataset",
                    DATASET,
                    "--radii",
                    *map(str, RADII),
                    "--n-pts",
                    "9",
                    "--batch-size",
                    "2",
                    "--limit",
                    "10",
                    "--max-seq-len",
                    "512",
                    "--seed",
                    "42",
                    "--outdir",
                    str(output),
                ],
                check=True,
            )
        rows = validate_grid(output / "grid.csv")
        # No trajectory checkpoints are evaluated; an empty table supports the unchanged plotter.
        (output / "trajectory.csv").write_text("checkpoint,c0,c1\n")
        dump(
            output / "validation.json",
            {
                "state": "passed",
                "points": len(rows),
                "protocol_sha256": sha(output / "protocol.json"),
                "grid_sha256": sha(output / "grid.csv"),
                "center_ce": next(
                    row["loss"] for row in rows if row["c0"] == row["c1"] == 0
                ),
                "no_trajectory_evaluations": True,
            },
        )
        dump(
            status,
            {"state": "running", "stage": "rendering", "numerical_complete": True},
        )
        render_environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        }
        for dimension in ("2d", "3d"):
            for extension in ("png", "pdf"):
                subprocess.run(
                    [
                        sys.executable,
                        str(code / f"plot_ll_{dimension}.py"),
                        "--indir",
                        str(output),
                        "--outpath",
                        str(output / f"landscape-{dimension}.{extension}"),
                    ],
                    check=True,
                    env=render_environment,
                )
        dump(
            status,
            {
                "state": "complete",
                "points": len(rows),
                "wall_seconds": time.monotonic() - started,
                "includes_loading_and_io": True,
            },
        )
    except BaseException as error:
        dump(
            status,
            {
                "state": "failed",
                "error": str(error),
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("case", "inputs"))
    parser.add_argument("run", type=Path)
    parser.add_argument("case")
    args = parser.parse_args()
    assert os.environ.get("FLUX_JOB_ID"), "Allocated launcher job required"
    (run_case if args.mode == "case" else prepare_inputs)(
        args.run.resolve(), args.case, Path(__file__).resolve().parent
    )


if __name__ == "__main__":
    main()
