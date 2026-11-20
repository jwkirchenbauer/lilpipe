#!/usr/bin/env python3
"""Evaluate pinned base Zephyr on full WMDP and MMLU with one Matrix GPU."""

import importlib.util
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path(__file__).resolve().parents[2]
OUTPUT = TARGET / "control-study/base-reference/evals"
VALIDATION = TARGET / "control-study/base-reference/validation.json"
LOG = TARGET / "control-study/base-reference/evaluation.log"
EVAL_PATH = TARGET / "source/eval-onegpu/misc/tar_compute_evaluation.py"
TASKS = ("wmdp_bio_robust", "mmlu_no_bio")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("tar_compute_evaluation", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_validation(report):
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = VALIDATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(VALIDATION)


def main():
    started = time.perf_counter()
    report = {
        "state": "running",
        "scope": "Pinned base Zephyr full-task, zero-shot BF16 reference for WMDP and MMLU; one Matrix GPU.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment_prefix": os.environ.get("CONDA_PREFIX"),
        "output": str(OUTPUT),
    }
    write_validation(report)
    try:
        evaluator = load_evaluator()
        evaluator.require_evaluation_runtime()
        if evaluator.evaluation_process_count() != 1:
            raise ValueError("Base reference requires TAR_EVAL_PROCESSES=1")
        study = evaluator.load_study(TARGET)
        model = Path(study["base_model"])
        report["model"] = str(model)
        evaluator.require_model(model)
        groups = evaluator.frozen_groups(TARGET, study)
        if set(groups) != set(TASKS):
            raise ValueError(f"Unexpected frozen task groups: {sorted(groups)}")
        if OUTPUT.exists():
            raise FileExistsError(f"Base reference output already exists: {OUTPUT}")
        OUTPUT.mkdir(parents=True)
        command = evaluator.evaluation_command(
            model, OUTPUT, TARGET / "source/lm_eval_tasks", TASKS
        )
        report["command"] = command
        import torch

        report["runtime"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "visible_gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        }
        write_validation(report)
        with LOG.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        found, rejected = evaluator.collect_results(OUTPUT, model, groups)
        if set(found) != set(TASKS) or rejected:
            raise ValueError(
                f"Both full tasks must verify with no rejected files; "
                f"found={sorted(found)}, rejected={rejected}"
            )
        task_reports = {}
        for task in TASKS:
            entry = found[task]
            task_reports[task] = {
                "jwk_percent": entry["jwk_percent"],
                "verified_sample_count": sum(row[1] for row in entry["signature"]),
                "verified_subtask_count": len(entry["signature"]),
                "signature": entry["signature"],
                "result_files": entry["files"],
            }
        report["tasks"] = task_reports
        report["rejected_results"] = rejected
        report["result_files"] = sorted(
            {path for task in task_reports.values() for path in task["result_files"]}
        )
        report["state"] = "complete"
    except Exception as error:
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_validation(report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
