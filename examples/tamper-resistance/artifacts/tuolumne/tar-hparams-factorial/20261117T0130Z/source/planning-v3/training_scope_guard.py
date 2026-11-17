"""Endpoint-only training gate for the fixed-sampler factorial extension."""

import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def validate_program(program, execution_plan, run, name):
    """Validate planned 125-update defense cells without any eval-plan field."""
    run = Path(run).resolve()
    if program.get("kind") != "fixed-sampler-hparam-study":
        raise ValueError("Unexpected study kind")
    required = execution_plan.get("training_bundle_required")
    if required != ["hparam-ri-defense", "hparam-ro-defense", "hparam-io-defense"]:
        raise ValueError("Training bundle differs from the reviewed RI/RO/IO scope")
    if execution_plan.get("attack_configs", {}).get("attack_jobs_planned") != []:
        raise ValueError("This extension does not plan attack jobs")
    if (
        execution_plan.get("attack_configs", {}).get("benchmark_evaluation_jobs_planned")
        != []
    ):
        raise ValueError("This extension does not plan benchmark evaluations")
    if name not in required:
        raise ValueError(f"Case is outside the planned defense bundle: {name}")
    case = program.get("cases", {}).get(name)
    if not isinstance(case, dict):
        raise ValueError(f"Missing planned case: {name}")
    condition = name.removeprefix("hparam-").removesuffix("-defense").upper()
    expected_values = execution_plan.get("factorial_cells_to_measure", {}).get(condition)
    if (
        case.get("name") != name
        or case.get("condition") != condition
        or case.get("role") != "defense"
        or case.get("max_steps") != 125
        or case.get("inner_steps_per_update") != 16
        or case.get("save_steps") != [1, 2, 4, 8, 16, 32, 64, 96, 125]
        or case.get("state_steps") != [32, 64, 125]
        or case.get("defense_hyperparameters") != expected_values
        or program.get("hparam_matrix", {}).get(condition) != expected_values
    ):
        raise ValueError(f"Case definition differs from planned 125x16 {condition} defense")
    folder = run / "cases" / name
    if Path(case.get("folder", "")).resolve() != folder:
        raise ValueError(f"Case folder is outside the prepared run: {name}")
    if Path(case.get("config", "")).resolve() != folder / "config.yml":
        raise ValueError(f"Case config is outside the prepared run: {name}")
    if case.get("base_model") != program.get("base_model"):
        raise ValueError(f"Case base model differs from the frozen study: {name}")
    if program.get("bundles", {}).get("defenses") != required:
        raise ValueError("Study defense bundle differs from the reviewed execution plan")
    return {"state": "passed", "case": name, "role": "defense", "steps": 125}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("training",))
    parser.add_argument("mode", choices=("case",))
    parser.add_argument("run", type=Path)
    parser.add_argument("case")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    plan = read(run / "execution-plan.json")
    program = read(run / "study.json")
    print(json.dumps(validate_program(program, plan, run, args.case)))


if __name__ == "__main__":
    main()
