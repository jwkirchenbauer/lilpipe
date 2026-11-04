"""Evaluate a defended checkpoint using the authors' SEAM evaluator."""

import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--result-json", required=True)
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
sys.path.insert(
    0, os.environ.get("SEAM_ROOT", str(Path(__file__).resolve().parents[1]))
)

from transformers import set_seed
import src.eval as seam_eval

set_seed(42)
report = {
    "evaluator": "examples/seam/src/eval.py",
    "attack": False,
    "seed": 42,
    "gpt_harmfulness_skipped": True,
}
original_harmfulness = seam_eval.eval_harmfulness
original_utility = seam_eval.eval_utility


def eval_harmfulness(*args, **kwargs):
    value = original_harmfulness(*args, **kwargs)
    report["model"] = kwargs.get("model", args[0]).config._name_or_path
    report["harmfulness"] = float(value)
    return value


def eval_utility(*args, **kwargs):
    results = original_utility(*args, **kwargs)
    metrics = {}
    for task in ("mmlu", "truthfulqa_mc1", "arc_easy", "hellaswag"):
        key = "acc_norm,none" if task in ("arc_easy", "hellaswag") else "acc,none"
        metrics[task] = {"metric": key, "score": float(results[task][key])}
        results[task]["acc,none"] = results[task][key]
    report["utility"] = metrics
    report["capability"] = sum(value["score"] for value in metrics.values()) / len(
        metrics
    )
    return results


seam_eval.eval_harmfulness = eval_harmfulness
seam_eval.eval_utility = eval_utility
seam_eval.eval_harmfulness_gpt = lambda *args, **kwargs: float("nan")
seam_eval.main()
assert "harmfulness" in report and len(report["utility"]) == 4
report["complete"] = True
output = Path(args.result_json)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)

summary = {
    "model_id": output.parent.name,
    "method": "SEAM (author-codebase, paper-config)",
    "model": report["model"],
    "evaluator": report["evaluator"],
    "post": {
        "harmfulness": report["harmfulness"],
        "capability": report["capability"],
        "tasks": {task: values["score"] for task, values in report["utility"].items()},
    },
}
output.with_name("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
