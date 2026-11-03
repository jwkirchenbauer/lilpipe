import json
from pathlib import Path

run = Path(__file__).resolve().parent
scores = {}
for path in (run / "evals/defence/checkpoint-250").rglob("results*.json"):
    results = json.loads(path.read_text())["results"]
    for task in ("wmdp_bio_robust", "mmlu_no_bio"):
        if task in results:
            scores[task] = 100 * results[task]["acc,none"]
reference = {"wmdp_bio_robust": 45.046082949308754, "mmlu_no_bio": 54.54136015613271}
passed = scores.keys() == reference.keys() and all(
    abs(scores[k] - reference[k]) <= 1.0 for k in reference
)
report = {
    "reference": reference,
    "reproduction": scores,
    "tolerance_pp": 1.0,
    "passed": passed,
}
(run / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
if not passed:
    raise SystemExit("Defence differs by more than 1pp; relearning blocked")
