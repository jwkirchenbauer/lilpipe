#!/usr/bin/env python3
"""Generate concrete lilpipe experiments from an Axolotl parameter sweep."""

from __future__ import annotations

import argparse
import copy
import shutil
import itertools
import hashlib
import re
import string
from pathlib import Path
from typing import Any

import yaml

ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[a-z0-9.-]*[a-z0-9])?$")
STRUCTURAL_PATHS = {"plugins", "trainer_cls", "custom_trainer_args"}


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def validate_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must start and end with lowercase letters or digits "
            "and contain only lowercase letters, digits, dots, or hyphens"
        )
    return value


def parameter_grid(stage: dict[str, Any]) -> dict[str, list[Any]]:
    grid = stage.get("parameters")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("sweep parameters must be a non-empty mapping")
    for path, values in grid.items():
        if not isinstance(path, str) or not path or path.startswith("."):
            raise ValueError(f"invalid parameter path: {path!r}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"parameter {path!r} must have a non-empty value list")
    return grid


def set_parameter(config: dict[str, Any], path: str, value: Any) -> None:
    """Set an existing leaf through a dotted path."""
    parts = path.split(".")
    if any(not part for part in parts):
        raise ValueError(f"invalid parameter path: {path!r}")
    current: Any = config
    for part in parts[:-1]:
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise ValueError(f"parameter path does not exist in base.yml: {path}")
            current = current[index]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"parameter path does not exist in base.yml: {path}")
    leaf = parts[-1]
    if not isinstance(current, dict) or leaf not in current:
        raise ValueError(f"parameter path does not exist in base.yml: {path}")
    if isinstance(current[leaf], (dict, list)):
        raise ValueError(f"parameter path must identify a scalar: {path}")
    current[leaf] = value


def combinations(grid: dict[str, list[Any]]):
    paths = sorted(grid)
    for values in itertools.product(*(grid[path] for path in paths)):
        yield dict(zip(paths, values, strict=True))


def sweep_stages(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    unsupported = {"parameters", "grid", "overrides"} & sweep.keys()
    if unsupported:
        raise ValueError(
            f"top-level sweep fields are not supported: {sorted(unsupported)}"
        )
    stages = sweep.get("sweeps")
    if not isinstance(stages, list) or not stages:
        raise ValueError("sweeps must be a non-empty list")
    names = []
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("each sweep must be a mapping")
        names.append(validate_id(stage.get("name"), "sweep name"))
        parameter_grid(stage)
    if len(names) != len(set(names)):
        raise ValueError("sweep names must be unique")
    return stages


def stage_overrides(stage: dict[str, Any]) -> dict[str, Any]:
    overrides = stage.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("sweep overrides must be a mapping")
    forbidden = STRUCTURAL_PATHS & overrides.keys()
    if forbidden:
        raise ValueError(f"structural overrides are not allowed: {sorted(forbidden)}")
    return overrides


def validate_sweep(sweep: dict[str, Any], base: dict[str, Any]) -> None:
    validate_id(sweep.get("id"), "sweep id")
    for stage in sweep_stages(sweep):
        probe = copy.deepcopy(base)
        for path, values in parameter_grid(stage).items():
            set_parameter(probe, path, values[0])
        for path, value in stage_overrides(stage).items():
            set_parameter(probe, path, value)


def dump_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def project_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def candidate_name(template: str, config: dict[str, Any]) -> str:
    """Render scalar config fields without attribute lookup or expressions."""
    if not isinstance(template, str) or not template:
        raise ValueError("name_template must be a non-empty string")
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if field not in config or isinstance(config[field], (dict, list)):
            raise ValueError(f"unknown or non-scalar name_template field: {field}")
        if conversion or "{" in spec or "}" in spec:
            raise ValueError(
                "name_template conversions and nested formats are unsupported"
            )
    name = template.format_map(config)
    # Scientific formatting uses e-05; canonical experiment slugs use e-5.
    name = re.sub(r"e([+-])0+(\d+)", r"e\1\2", name)
    return validate_id(name, "generated model id")


def generate(sweep_path: Path, output_root: Path | None = None) -> Path:
    """Expand a sweep and return its generated experiment path."""
    sweep_path = sweep_path.resolve()
    sweep = load_mapping(sweep_path)
    base_name = sweep.get("base", "base.yml")
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise ValueError("sweep base must be a filename adjacent to sweep.yml")
    base_path = sweep_path.with_name(base_name)
    if not base_path.is_file():
        raise ValueError(f"missing adjacent base.yml: {base_path}")
    base = load_mapping(base_path)
    validate_sweep(sweep, base)

    sweep_id = sweep["id"]
    project_root = Path.cwd()
    config_mode = sweep.get("config_mode", "resolved")
    if config_mode not in ("resolved", "overrides"):
        raise ValueError("config_mode must be resolved or overrides")
    root = (
        output_root
        or project_root
        / (
            "artifacts/generated"
            if config_mode == "overrides"
            else "artifacts/generated/hpo"
        )
        / sweep_id
    )
    base_snapshot = root / "base.yml"
    run_records = {}
    config_dir = root / "configs"
    model_entries = {}
    model_ids = []
    producer = sweep.get("producer")
    if not isinstance(producer, dict) or not producer.get("script"):
        raise ValueError("producer.script is required")
    depends_on = sweep.get("depends_on", [])
    evaluations = sweep.get("evaluations", [])
    if not isinstance(depends_on, list) or not all(
        isinstance(value, str) for value in depends_on
    ):
        raise ValueError("depends_on must be a list of model IDs")
    if not isinstance(evaluations, list) or not all(
        isinstance(value, str) for value in evaluations
    ):
        raise ValueError("evaluations must be a list of evaluation IDs")

    all_dependencies = set(depends_on)
    for group in sweep_stages(sweep):
        group_dependencies = group.get("depends_on", depends_on)
        if not isinstance(group_dependencies, list) or not all(
            isinstance(x, str) for x in group_dependencies
        ):
            raise ValueError("sweep depends_on must be a list of model IDs")
        all_dependencies.update(group_dependencies)
    if all_dependencies:
        canonical_path = project_root / "configs/registries/models.yml"
        canonical = load_mapping(canonical_path).get("models", {})

        def add_dependency(model_id: str) -> None:
            if model_id in model_entries:
                return
            if model_id not in canonical:
                raise ValueError(f"unknown dependency model: {model_id}")
            model_entries[model_id] = canonical[model_id]
            for dependency in (
                canonical[model_id].get("producer", {}).get("depends_on", [])
            ):
                add_dependency(dependency)

        for dependency in sorted(all_dependencies):
            add_dependency(dependency)

    seen_configs = set()
    for stage in sweep_stages(sweep):
        stage_name = stage["name"]
        stage_dependencies = stage.get("depends_on", depends_on)
        for index, values in enumerate(combinations(parameter_grid(stage)), start=1):
            candidate_id = f"{sweep_id}--{stage_name}--{index:03d}"
            config = copy.deepcopy(base)
            for path, value in values.items():
                set_parameter(config, path, value)
            for path, value in stage_overrides(stage).items():
                set_parameter(config, path, value)
            template = stage.get("name_template", sweep.get("name_template"))
            if template is not None:
                candidate_id = candidate_name(template, config)
            if candidate_id in model_entries:
                raise ValueError(f"duplicate generated model id: {candidate_id}")
            signature = yaml.safe_dump(config, sort_keys=True)
            if signature in seen_configs:
                raise ValueError(f"duplicate generated configuration: {candidate_id}")
            seen_configs.add(signature)
            model_root = sweep.get(
                "model_root",
                (
                    "artifacts/tamia/models"
                    if config_mode == "overrides"
                    else f"artifacts/generated/hpo/{sweep_id}/models"
                ),
            )
            if not isinstance(model_root, str) or not model_root:
                raise ValueError("model_root must be a non-empty path")
            artifact = str(Path(model_root) / candidate_id)
            config["output_dir"] = artifact
            config["wandb_name"] = candidate_id
            config_path = config_dir / f"{candidate_id}.yml"
            if config_mode == "resolved":
                dump_yaml(config_path, config)
                training_args = [project_path(config_path, project_root)]
            else:
                overrides = {
                    **values,
                    **stage_overrides(stage),
                    "output_dir": artifact,
                    "wandb_name": candidate_id,
                }
                cli_args = []
                for key, value in overrides.items():
                    if "." in key or isinstance(value, (dict, list)) or value is None:
                        raise ValueError(
                            "CLI overrides require top-level non-null scalars; use resolved mode for nested settings"
                        )
                    cli_args.extend(["--" + key.replace("_", "-"), str(value)])
                if not base_snapshot.exists():
                    base_snapshot.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(base_path, base_snapshot)
                elif base_snapshot.read_bytes() != base_path.read_bytes():
                    raise ValueError(
                        "Generated base snapshot differs; choose a fresh output directory"
                    )
                training_args = [project_path(base_snapshot, project_root), *cli_args]
                run_records[candidate_id] = {
                    "base": project_path(base_snapshot, project_root),
                    "overrides": overrides,
                    "effective_config": config,
                }
            producer_entry = {
                "id": f"train-{candidate_id}",
                "script": producer["script"],
                "args": [*training_args, *producer.get("args", [])],
            }
            if stage_dependencies:
                producer_entry["depends_on"] = stage_dependencies
            resources = producer.get("sbatch_args", producer.get("resources"))
            if resources:
                producer_entry["sbatch_args"] = resources
            model_entries[candidate_id] = {
                "base_model_name_or_path": artifact,
                "adapter_name_or_path": "-",
                "local_dir": artifact,
                "producer": producer_entry,
            }
            model_fields = sweep.get("model", {})
            if not isinstance(model_fields, dict):
                raise ValueError("model must be a mapping")
            model_entries[candidate_id].update(
                {
                    key: (
                        value.format(
                            id=candidate_id,
                            artifact=artifact,
                            parameter=values,
                            sweep=stage_name,
                        )
                        if isinstance(value, str)
                        else value
                    )
                    for key, value in model_fields.items()
                }
            )
            model = model_entries[candidate_id]
            model.setdefault("evaluation_root", "artifacts/tamia/evals")
            model.setdefault("evaluation_trajectory", artifact)
            final_step = config.get("max_steps")
            if isinstance(final_step, int) and final_step > 0:
                model["evaluation_final_step"] = str(final_step)
                model["evaluation_final_model"] = artifact
                interval = config.get("save_steps", final_step)
                if not isinstance(interval, int) or interval < 1:
                    raise ValueError(
                        "checkpoint evaluation requires positive integer save_steps"
                    )
                model.setdefault(
                    "evaluation_checkpoint_steps",
                    ",".join(
                        map(
                            str,
                            sorted({*range(0, final_step + 1, interval), final_step}),
                        )
                    ),
                )
            model_ids.append(candidate_id)

    if run_records:
        dump_yaml(
            root / "runs.yml",
            {
                "sweep": sweep,
                "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "runs": run_records,
            },
        )
    models_path = root / "models.yml"
    dump_yaml(models_path, {"models": model_entries})
    experiment = {
        "version": 1,
        "id": sweep_id,
        "registries": {
            "models": project_path(models_path, project_root),
            "evaluations": sweep.get(
                "evaluation_registry", "configs/registries/evals.yml"
            ),
        },
        "models": model_ids,
        "evaluations": evaluations,
    }
    experiment_path = root / "experiment.yml"
    dump_yaml(experiment_path, experiment)
    return experiment_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("sweep", type=Path)
    args = parser.parse_args(argv)
    experiment = generate(args.sweep)
    print(experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
