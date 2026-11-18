"""Summarize recorded token treatment and verify reference-transport parity."""

import hashlib
import json
import sys
from contextlib import ExitStack
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
    current = json.loads((run / "inner-local-v5/outer-000001.json").read_text())
    assert previous["inner_batch_sha256"] == current["inner_batch_sha256"]
    assert previous["inner_steps"] == current["inner_steps"]
    errors = {}
    exactly_equal = True
    for family in ["whole", *previous["families"]]:
        a = previous if family == "whole" else previous["families"][family]
        b = current if family == "whole" else current["families"][family]
        for rep in ("raw", "bf16"):
            keys = ("inner_gram", "cross_gram", "reference_gram")
            if family == "whole":
                keys += ("anchor_gap_squared_norm",)
            for key in keys:
                x, y = (
                    np.asarray(record["representations"][rep][key]) for record in (a, b)
                )
                error = float(np.max(np.abs(x - y)))
                errors[f"{family}/{rep}/{key}"] = error
                exactly_equal &= bool(np.array_equal(x, y))
                np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)
    import torch
    from tar_basis_structure import WeightReader

    anchor_path = json.loads((run / "references.json").read_text())["references"][0][
        "anchor"
    ]
    final_path = run / "cases/baseline-inner-recording/models/defense/checkpoint-125"
    coordinates, different = 0, 0
    with ExitStack() as stack:
        readers = [WeightReader(path, stack) for path in (anchor_path, final_path)]
        names = [
            sorted(reader.mapping or reader.handles["model.safetensors"].keys())
            for reader in readers
        ]
        assert names[0] == names[1]
        for name in names[0]:
            a, b = [reader.tensor(name).reshape(-1) for reader in readers]
            assert a.dtype == b.dtype and a.shape == b.shape
            coordinates += a.numel()
            for start in range(0, a.numel(), 1_048_576):
                section = slice(start, start + 1_048_576)
                different += int(torch.count_nonzero(a[section] != b[section]))
    result = {
        "state": "passed" if different == 0 else "failed",
        "actual_attack_anchor_equality": {
            "anchor": str(anchor_path),
            "replayed_final_defender": str(final_path),
            "coordinates": coordinates,
            "different_coordinates": different,
            "equal_saved_values": different == 0,
        },
        "actual_first16": describe(prefix),
        "inner_calls": results,
        "document_identity": "SHA256 of complete input_ids and attention_mask; labels excluded",
        "padding_definition": "Causally shifted labels != -100 at attention_mask == 0",
        "source_paths": [str(old), str(actual)],
        "reference_transport_parity": {
            "previous_recorder_sha256": previous["source_sha256"],
            "current_recorder_sha256": current["source_sha256"],
            "comparison": "Completed remote-reference versus local-reference outer-call-1 aggregates",
            "all_matrix_values_exactly_equal": exactly_equal,
            "comparison_tolerance": {"rtol": 1e-12, "atol": 1e-14},
            "maximum_absolute_error": max(errors.values()),
            "errors": errors,
        },
    }
    (run / "protocol-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert different == 0, "Replayed final defender differs from actual attack anchor"


if __name__ == "__main__":
    main()
