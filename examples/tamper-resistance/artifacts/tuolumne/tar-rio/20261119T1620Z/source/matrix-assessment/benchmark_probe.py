"""Run one full standard WMDP group and compare with the saved RIO control."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    import torch

    root = Path(__file__).resolve().parents[2]
    control = root.parent.parent / "tar-hparams/20261116T2310Z/cases/hparam-rio-lr1e-6"
    output = root / "matrix-assessment/benchmark-probe"
    assert os.environ.get("SLURM_JOB_ID"), "Use launch_matrix.py"
    assert torch.__version__ == "2.12.1+cu126" and torch.version.hip is None
    assert torch.cuda.device_count() == 4
    source = root / "source/misc/tar_compute_evaluation.py"
    spec = importlib.util.spec_from_file_location("frozen_evaluation", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    study = module.load_study(root)
    groups = module.frozen_groups(root, study)
    task_sources = {
        name: expected
        for name, expected in study["source_sha256"].items()
        if name.startswith("source/lm_eval_tasks/")
    }
    assert task_sources and all(
        hashlib.sha256((control.parent.parent / name).read_bytes()).hexdigest()
        == expected
        for name, expected in task_sources.items()
    ), "Tuolumne control used different frozen task definitions"
    model = control / "models/defense"
    module.require_model(model)
    reference, rejected = module.collect_results(
        control / "evals/checkpoint-0", model, groups
    )
    task = "wmdp_bio_robust"
    assert task in reference, "Missing complete Tuolumne reference"
    output.mkdir(parents=True, exist_ok=False)
    command = module.evaluation_command(
        model, output / "evals", root / "source/lm_eval_tasks", [task]
    )
    module.dump(
        output / "invocation.json",
        {"argv": command, "reference": reference[task], "reference_rejected": rejected},
    )
    environment = os.environ.copy()
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"):
        environment.pop(name, None)
    started = time.monotonic()
    with (output / "benchmark.log").open("w") as log:
        subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    actual, rejected = module.collect_results(output / "evals", model, groups)
    assert task in actual, "Benchmark group did not pass full-sample validation"
    old = reference[task]
    new = actual[task]
    assert [row[:2] for row in old["signature"]] == [
        row[:2] for row in new["signature"]
    ]
    old_data = module.read_json(old["files"][0])
    new_data = module.read_json(new["files"][0])
    old_hashes = old_data.get("task_hashes", {})
    new_hashes = new_data.get("task_hashes", {})
    hash_comparison = {
        name: old_hashes.get(name) == new_hashes.get(name)
        for name in groups[task]
        if name in old_hashes
    }
    assert all(hash_comparison.values()), "Evaluation input hashes changed"
    result = {
        "state": "complete",
        "scope": "One full zero-shot WMDP group at the RIO defended checkpoint; not a complete attack-trajectory evaluation",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "torch": torch.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(4)],
        "elapsed_seconds": time.monotonic() - started,
        "tuolumne_percent": old["jwk_percent"],
        "matrix_percent": new["jwk_percent"],
        "difference_percentage_points": new["jwk_percent"] - old["jwk_percent"],
        "exact_score_agreement": module.consistent(old, new),
        "input_hash_comparison": hash_comparison,
        "input_hash_validation": (
            "matched"
            if len(hash_comparison) == len(groups[task])
            else "unavailable_or_partial_in_reference"
        ),
        "identical_frozen_task_source_files": len(task_sources),
        "full_sample_counts": {name: count for name, count, _ in new["signature"]},
        "subtask_differences_percentage_points": {
            a[0]: 100 * (b[2] - a[2])
            for a, b in zip(old["signature"], new["signature"])
        },
        "rejected_results": rejected,
        "matrix_result_files": new["files"],
        "tuolumne_result_files": old["files"],
    }
    module.dump(output / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
