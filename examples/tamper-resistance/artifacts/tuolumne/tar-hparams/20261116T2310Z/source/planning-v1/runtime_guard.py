"""Metadata-only runtime gate; scheduler afterok is not training validation."""

import json
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def training(run, program, name):
    case = program["cases"][name]
    folder = run / "cases" / name
    assert read(folder / "status/train.json")["state"] == "complete", name
    proof = read(folder / "training-validation.json")
    assert proof["state"] == "passed" and proof["updates"] == case["max_steps"], name
    order = proof["input_order_validation"]
    assert order["state"] == "passed" and order["batches"] == case["max_steps"], name
    assert len(order["canonical_step_batch_sha256"]) == 64, name


def main():
    profile, mode, location, *arguments = sys.argv[1:]
    run = Path(location).resolve()
    program = read(run / "study.json")
    assert program["kind"] == "fixed-sampler-hparam-study"
    assert program["evaluation_cases"] == program["bundles"]["attacks"]
    if profile == "training":
        assert mode == "case"
        case = program["cases"][arguments[0]]
        if case["role"] == "attack":
            defense = Path(case["defense_reference"]).parents[1].name
            training(run, program, defense)
        return
    if profile == "evaluation":
        if mode == "report":
            return
        if mode == "resume":
            names = arguments[:1]
        elif "--cases" in arguments:
            names = arguments[arguments.index("--cases") + 1 :]
        elif "--bundle" in arguments:
            names = program["bundles"][arguments[arguments.index("--bundle") + 1]]
        else:
            raise ValueError("Explicit evaluation cases or bundle required")
    elif mode == "checkpoint-compare":
        names = list(program["cases"])
        for name, case in program["cases"].items():
            folder = run / "cases" / name
            if case["role"] == "attack":
                assert (
                    read(folder / "status/evaluation.json")["state"] == "complete"
                ), name
                assert (
                    read(folder / "status/analysis.json")["state"] == "complete"
                ), name
                assert (
                    read(folder / "status/step-profiles.json")["state"] == "complete"
                ), name
    else:
        names = arguments
        assert names
    for name in names:
        training(run, program, name)
    if mode.startswith("profiles") or mode.startswith("profile-render"):
        reference = program["profile_reference_case"]
        if any(name != reference for name in names):
            assert (
                read(run / "cases" / reference / "status/step-profiles.json")["state"]
                == "complete"
            ), "Finish the shared profile reference first"
    print("Operational metadata gate passed: " + ", ".join(names), flush=True)


if __name__ == "__main__":
    main()
