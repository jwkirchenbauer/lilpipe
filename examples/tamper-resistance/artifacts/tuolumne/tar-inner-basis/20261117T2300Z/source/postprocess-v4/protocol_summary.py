"""Summarize recorded token treatment and verify reference-transport parity."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def rows(path, selected):
    with path.open() as stream:
        return {
            row["step"]: row["batch"]
            for line in stream
            if (row := json.loads(line))["step"] in selected
        }


def signatures(batch):
    return [
        hashlib.sha256(json.dumps([tokens, mask]).encode()).hexdigest()
        for tokens, mask in zip(
            batch["input_ids"], batch["attention_mask"], strict=True
        )
    ]


def describe(batch):
    labels = np.asarray(batch["labels"])[:, 1:]
    mask = np.asarray(batch["attention_mask"])[:, 1:]
    active = labels != -100
    padding = (mask == 0) & active
    return {
        "documents": len(labels),
        "active_shifted_targets": int(active.sum()),
        "active_padding_targets": int(padding.sum()),
        "padding_fraction_of_active_targets": float(padding.sum() / active.sum()),
    }


def main():
    run = Path(sys.argv[1]).resolve()
    example = run.parents[3]
    old = (
        example
        / "artifacts/tuolumne/tar-k16-repro/20261109T1830Z/dynamics/baseline-k16-20261110T0400Z/models/defense/observed-batches.jsonl"
    )
    actual = (
        example
        / "artifacts/tuolumne/tar-compute/20261110T1815Z/cases/baseline-lr1e-6/models/attack/observed-batches.jsonl"
    )
    calls = [1, 32, 64, 96, 125]
    defense = rows(old, calls)
    attack = rows(actual, set(range(1, 17)))
    assert set(defense) == set(calls) and set(attack) == set(range(1, 17))
    prefix = {
        field: [attack[step][field][0] for step in range(1, 17)]
        for field in ("input_ids", "attention_mask", "labels")
    }
    actual_signatures = set(signatures(prefix))
    results = []
    for call in calls:
        batch = defense[call]
        indices = [i for i, role in enumerate(batch["metadata_ids"]) if role == 2]
        inner = {field: [batch[field][i] for i in indices] for field in prefix}
        assert len(indices) == 16
        results.append(
            {
                "outer_call": call,
                "inner": describe(inner),
                "documents_shared_with_actual_first16": len(
                    set(signatures(inner)) & actual_signatures
                ),
            }
        )
    previous = json.loads((run / "inner-gpu-v2/outer-000001.json").read_text())
    current = json.loads((run / "inner-local-v4/outer-000001.json").read_text())
    assert previous["source_sha256"] == current["source_sha256"]
    assert previous["inner_batch_sha256"] == current["inner_batch_sha256"]
    assert previous["inner_steps"] == current["inner_steps"]
    errors = {}
    exactly_equal = True
    for family in ["whole", *previous["families"]]:
        a = previous if family == "whole" else previous["families"][family]
        b = current if family == "whole" else current["families"][family]
        for rep in ("raw", "bf16"):
            for key in ("inner_gram", "cross_gram", "reference_gram"):
                x, y = (
                    np.asarray(record["representations"][rep][key]) for record in (a, b)
                )
                error = float(np.max(np.abs(x - y)))
                errors[f"{family}/{rep}/{key}"] = error
                exactly_equal &= bool(np.array_equal(x, y))
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
    result = {
        "state": "passed",
        "actual_first16": describe(prefix),
        "inner_calls": results,
        "document_identity": "SHA256 of complete input_ids and attention_mask; labels excluded",
        "padding_definition": "Causally shifted labels != -100 at attention_mask == 0",
        "source_paths": [str(old), str(actual)],
        "reference_transport_parity": {
            "comparison": "Completed remote-reference versus local-reference outer-call-1 aggregates",
            "all_matrix_values_exactly_equal": exactly_equal,
            "comparison_tolerance": {"rtol": 1e-12, "atol": 1e-14},
            "maximum_absolute_error": max(errors.values()),
            "errors": errors,
        },
    }
    (run / "protocol-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
