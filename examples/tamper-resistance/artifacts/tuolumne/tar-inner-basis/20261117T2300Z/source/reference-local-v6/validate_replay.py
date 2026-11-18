"""Validate observation-only replay and compare its outer states to frozen B."""

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path


def publish(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--inner-root", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    sys.path.insert(0, str(run / "source/misc"))
    import torch
    import tar_compute
    from tar_basis_structure import WeightReader

    program = json.loads((run / "study.json").read_text())
    case = next(iter(program["cases"].values()))
    folder = Path(case["folder"])
    tar_compute.validate_case(folder)
    reference_root = Path(program["baseline_dynamics"]) / "models/defense"
    actual_root = folder / "models/defense"
    histories = [
        json.loads((root / "checkpoint-125/trainer_state.json").read_text())[
            "log_history"
        ]
        for root in (reference_root, actual_root)
    ]
    inner_losses = [
        [
            (row["inner_step_total"], row["inner_loss"])
            for row in history
            if "inner_step_total" in row
        ]
        for history in histories
    ]
    assert len(inner_losses[0]) == len(inner_losses[1]) == 2000
    assert [row[0] for row in inner_losses[0]] == [row[0] for row in inner_losses[1]]
    inner_loss_error = max(abs(old[1] - new[1]) for old, new in zip(*inner_losses))
    rows = []
    for step in case["save_steps"]:
        with contextlib.ExitStack() as stack:
            old = WeightReader(reference_root / f"checkpoint-{step}", stack)
            new = WeightReader(actual_root / f"checkpoint-{step}", stack)
            names = (
                sorted(old.mapping)
                if old.mapping
                else sorted(old.handles["model.safetensors"].keys())
            )
            new_names = (
                sorted(new.mapping)
                if new.mapping
                else sorted(new.handles["model.safetensors"].keys())
            )
            assert names == new_names
            different, total, squared, maximum = 0, 0, 0.0, 0.0
            for name in names:
                left, right = old.tensor(name), new.tensor(name)
                assert left.shape == right.shape and left.dtype == right.dtype
                left, right = left.reshape(-1), right.reshape(-1)
                for start in range(0, left.numel(), 1_048_576):
                    a, b = (
                        left[start : start + 1_048_576],
                        right[start : start + 1_048_576],
                    )
                    count = int(torch.count_nonzero(a != b))
                    different += count
                    total += a.numel()
                    if count:
                        delta = b.double() - a.double()
                        squared += float(torch.dot(delta, delta))
                        maximum = max(maximum, float(delta.abs().max()))
            row = {
                "outer_step": step,
                "coordinates": total,
                "different_coordinates": different,
                "maximum_absolute_difference": maximum,
                "l2_difference": squared**0.5,
                "bitwise_equal_values": different == 0,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    aggregates = []
    observations = [
        json.loads(line)
        for line in (actual_root / "observed-batches.jsonl").read_text().splitlines()
    ]
    fields = ("input_ids", "attention_mask", "labels")
    for outer in program["outer_calls"]:
        path = (args.inner_root or run / "inner") / f"outer-{outer:06d}.json"
        aggregate = json.loads(path.read_text())
        assert aggregate["status"] == "complete"
        assert aggregate["outer_call"] == outer
        assert aggregate["inner_steps"] == program["inner_steps"]
        batch = observations[outer - 1]["batch"]
        selected = [i for i, role in enumerate(batch["metadata_ids"]) if role == 2]
        attack_batch = {key: [batch[key][i] for i in selected] for key in fields}
        signature = hashlib.sha256(
            json.dumps(attack_batch, sort_keys=True).encode()
        ).hexdigest()
        expected = aggregate["inner_batch_sha256"]
        assert signature == expected, (outer, signature, expected)
        aggregates.append(
            {
                "outer_call": outer,
                "attack_batch_sha256": signature,
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = {
        "state": (
            "passed" if all(row["bitwise_equal_values"] for row in rows) else "failed"
        ),
        "outer_checkpoints": rows,
        "inner_loss_trace": {
            "observations": 2000,
            "maximum_absolute_difference": inner_loss_error,
            "exactly_equal": inner_loss_error == 0,
        },
        "inner_aggregates": aggregates,
        "new_attacks": [],
        "interpretation": "Exact saved-value equality verifies observation did not alter the defense at declared checkpoints; original actual attack is reused.",
    }
    publish(run / "replay-validation.json", result)
    assert (
        result["state"] == "passed"
    ), "Investigate replay differences before interpreting basis comparison"


if __name__ == "__main__":
    main()
