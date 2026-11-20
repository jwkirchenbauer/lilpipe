#!/usr/bin/env python3
"""Run and verify one full WMDP evaluation on one allocated Matrix GPU."""

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


TARGET = Path(__file__).resolve().parents[2]
ORIGIN = TARGET.parents[2] / "tuolumne/tar-rio/20261119T1620Z"
MODEL = (
    TARGET.parents[2]
    / "tuolumne/tar-hparams/20261116T2310Z/cases/hparam-rio-lr1e-6/models/defense"
)
REFERENCE = ORIGIN / "matrix-assessment/benchmark-probe/result.json"
EVAL_PATH = TARGET / "source/eval-onegpu/misc/tar_compute_evaluation.py"
OUTPUT = TARGET / "calibration/eval-onegpu/evals"
REPORT = TARGET / "calibration/eval-onegpu/calibration.json"
LOG = TARGET / "calibration/eval-onegpu/evaluation.log"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("tar_compute_evaluation", EVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_report(data):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(REPORT)


def main():
    started = time.perf_counter()
    report = {
        "state": "running",
        "scope": "One full, zero-shot WMDP group at the original RIO defended checkpoint; one Matrix GPU calibration only.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
        "visible_devices_before_preflight": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model": str(MODEL),
        "reference": str(REFERENCE),
    }
    write_report(report)
    try:
        evaluator = load_evaluator()
        evaluator.require_evaluation_runtime()
        if evaluator.evaluation_process_count() != 1:
            raise ValueError("Calibration requires TAR_EVAL_PROCESSES=1")
        evaluator.require_model(MODEL)
        study = evaluator.load_study(TARGET)
        groups = evaluator.frozen_groups(TARGET, study)
        if OUTPUT.exists():
            raise FileExistsError(f"Calibration output already exists: {OUTPUT}")
        OUTPUT.mkdir(parents=True)
        command = evaluator.evaluation_command(
            MODEL, OUTPUT, TARGET / "source/lm_eval_tasks", ["wmdp_bio_robust"]
        )
        report["command"] = command
        report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = torch.version.cuda
        report["visible_gpu_count"] = torch.cuda.device_count()
        report["gpu_name"] = torch.cuda.get_device_name(0)
        write_report(report)
        with LOG.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        found, rejected = evaluator.collect_results(OUTPUT, MODEL, groups)
        actual = found.get("wmdp_bio_robust")
        if actual is None:
            raise ValueError(f"No verified complete WMDP result; rejected={rejected}")
        reference_data = json.loads(REFERENCE.read_text())
        reference_result = json.loads(
            Path(reference_data["matrix_result_files"][0]).read_text()
        )
        expected = evaluator.verified_task(
            reference_result, "wmdp_bio_robust", MODEL, groups["wmdp_bio_robust"]
        )
        actual_signature = actual["signature"]
        expected_signature = [tuple(row) for row in expected["signature"]]
        if [row[:2] for row in actual_signature] != [
            row[:2] for row in expected_signature
        ]:
            raise ValueError("One-/four-GPU benchmark subtask sample counts differ")
        actual_sample_count = sum(row[1] for row in actual_signature)
        expected_sample_count = sum(row[1] for row in expected_signature)
        comparison = {
            "exact_score_agreement": evaluator.consistent(
                actual, {**expected, "signature": expected_signature}
            ),
            "actual_percent": actual["jwk_percent"],
            "reference_four_gpu_percent": expected["jwk_percent"],
            "difference_percentage_points": actual["jwk_percent"]
            - expected["jwk_percent"],
            "actual_sample_count": actual_sample_count,
            "reference_sample_count": expected_sample_count,
            "same_full_sample_count": actual_sample_count == expected_sample_count,
            "actual_signature": actual_signature,
            "reference_signature": expected_signature,
            "subtask_differences_percentage_points": {
                current[0]: 100 * (current[2] - reference[2])
                for current, reference in zip(actual_signature, expected_signature)
            },
            "result_files": actual["files"],
            "rejected_results": rejected,
        }
        if actual_sample_count != 868 or expected_sample_count != 868:
            raise ValueError(f"Expected 868 full WMDP samples: {comparison}")
        report["comparison"] = comparison
        report["state"] = "complete"
    except Exception as error:
        report["state"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
