"""Prepare two missing CUDA K cells without changing completed producers."""

import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT.parents[3]
PREVIOUS = EXAMPLE / "artifacts/matrix/tar-rio/20261120T2120Z"
TRAIN = "/usr/workspace/wsb/kirchenb/lilpipe-tar-train-cu126-v1"
EVAL = "/usr/workspace/wsb/kirchenb/lilpipe-tar-eval-cu126-v1"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert EXAMPLE.name == "tamper-resistance", EXAMPLE
    assert not (ROOT / "study.json").exists(), "Use a fresh run root"
    for name in ("cases", "logs", "status", "launches", "cache"):
        (ROOT / name).mkdir(exist_ok=True)
    (ROOT / "source").symlink_to(PREVIOUS / "source")
    shutil.copy2(PREVIOUS / "probes.json", ROOT / "probes.json")
    template_root = PREVIOUS / "backend-control"
    program = read(template_root / "study.json")
    program.update(
        purpose="Complete a same-backend CUDA K4/K16/K64/K256 comparison.",
        parent_study=str(PREVIOUS / "study.json"),
        source=str(ROOT / "source"),
        probes=str(ROOT / "probes.json"),
        probes_sha256=digest(ROOT / "probes.json"),
        measurement_backend="matrix-cuda126",
        training_backend_note="Every compared K defense and actual attack uses CUDA.",
        runtime_provenance="Same frozen four-GPU CUDA training worker and prefix as the completed K4 control and K256.",
        cases={},
        evaluation_cases=[],
        bundles={"defenses": [], "attacks": [], "all": []},
        profile_reference_case="rio-k16-cuda-lr1e-6",
    )
    deltas = {}
    template_analysis = read(template_root / "cases/rio-k4-cuda-lr1e-6/study.json")
    for k in (16, 64):
        defense = ROOT / "cases" / f"rio-k{k}-cuda-defense"
        defended = defense / "models/defense"
        for role in ("defense", "attack"):
            suffix = "defense" if role == "defense" else "lr1e-6"
            template = template_root / "cases" / f"rio-k4-cuda-{suffix}"
            name = f"rio-k{k}-cuda-{suffix}"
            folder = ROOT / "cases" / name
            for child in ("models", "logs", "status", "geometry", "surfaces", "cache"):
                (folder / child).mkdir(parents=True)
            (folder / "source").symlink_to(ROOT / "source")
            config = yaml.safe_load((template / "config.yml").read_text())
            before = copy.deepcopy(config)
            config.update(output_dir=str(folder / "models" / role), wandb_name=name)
            if role == "defense":
                config["custom_trainer_args"].update(
                    attack_max_steps=k, attack_epochs=max(1, k // 16)
                )
            else:
                config["base_model"] = str(defended)
                (folder / "models/defense").symlink_to(defended)
            allowed = (
                {"output_dir", "wandb_name", "custom_trainer_args"}
                if role == "defense"
                else {"output_dir", "wandb_name", "base_model"}
            )
            changed = {key for key in before if before[key] != config[key]}
            assert changed <= allowed, changed
            if role == "defense":
                args_before = before["custom_trainer_args"]
                args_after = config["custom_trainer_args"]
                assert {
                    key for key in args_before if args_before[key] != args_after[key]
                } <= {"attack_max_steps", "attack_epochs"}
            (folder / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
            case = read(template / "case.json")
            case.update(
                name=name,
                label=(
                    f"RIO K={k} CUDA; 125 outer defense updates"
                    if role == "defense"
                    else f"RIO K={k} CUDA; standard 1000-update attack at lr=1e-6"
                ),
                condition=f"rio-k{k}",
                folder=str(folder),
                config=str(folder / "config.yml"),
                source_case=str(template),
            )
            if role == "defense":
                case.update(k=k, inner_steps_per_update=k, simulation_steps=k)
            else:
                case.update(base_model=str(defended), defense_reference=str(defended))
            write(folder / "case.json", case)
            exports = {
                "TAR_CASE_NAME": name,
                "TAR_CASE_ROLE": role,
                "TAR_CASE_CONFIG": str(folder / "config.yml"),
                "DYNAMICS_SAVE_STEPS": ",".join(map(str, case["save_steps"])),
                "DYNAMICS_STATE_STEPS": ",".join(map(str, case["state_steps"])),
            }
            if role == "attack":
                exports["DYNAMICS_SOURCE_MANIFEST"] = str(
                    folder / "source-dataset.json"
                )
                settings = copy.deepcopy(template_analysis)
                settings.update(
                    case_name=name,
                    attack_anchor=str(defended),
                    defense_checkpoint_root=str(defended),
                    capability_scores_file=str(folder / "evaluation-scores.csv"),
                    source_case_study=str(template / "study.json"),
                )
                write(folder / "study.json", settings)
                program["evaluation_cases"].append(name)
            (folder / "runtime.sh").write_text(
                "\n".join(
                    f"export {key}={shlex.quote(value)}"
                    for key, value in exports.items()
                )
                + "\n"
            )
            program["cases"][name] = case
            program["bundles"]["defenses" if role == "defense" else "attacks"].append(
                name
            )
            program["bundles"]["all"].append(name)
            deltas[name] = {
                key: {"template": before[key], "new": config[key]}
                for key in sorted(changed)
            }
    write(ROOT / "study.json", program)
    write(
        ROOT / "status/probes.json",
        {
            "state": "complete",
            "reused": str(PREVIOUS / "probes.json"),
            "sha256": digest(ROOT / "probes.json"),
        },
    )
    reused = {
        "4": str(template_root / "cases/rio-k4-cuda-lr1e-6"),
        "256": str(PREVIOUS / "cases/rio-k256-lr1e-6"),
    }
    for folder in reused.values():
        for phase in ("train", "evaluation", "analysis", "step-profiles"):
            assert (
                read(Path(folder) / "status" / (phase + ".json"))["state"] == "complete"
            )
    write(
        ROOT / "protocol.json",
        {
            "purpose": program["purpose"],
            "parent_report": str(PREVIOUS / "report.md"),
            "reason": "Paired K4 training-backend control differed by up to 5.760369 WMDP percentage points and 2.980033 MMLU percentage points.",
            "new_k_values": [16, 64],
            "reused_cuda_attack_folders": reused,
            "configuration_deltas_from_completed_cuda_k4": deltas,
            "training_prefix": TRAIN,
            "evaluation_prefix": EVAL,
            "source_root": str(PREVIOUS / "source"),
            "training_worker_sha256": digest(
                PREVIOUS / "source/misc/run_tar_compute.sh"
            ),
            "training_observer_sha256": digest(
                PREVIOUS / "source/misc/tar_dynamics_training.py"
            ),
            "tar_trainer_sha256": digest(PREVIOUS / "source/src/trainers/tr_tar.py"),
            "probes_sha256": digest(ROOT / "probes.json"),
            "random_and_reference_directions": "Reuse the same literal saved directions as the completed study; each condition also has its own fitted attack basis.",
            "predeclared_scope": "One paired seed; fixed 125 outer updates and increasing inner compute/exposure. Utility gate and legacy AMP-cache caveats remain. No numerical simulator correction in this comparison.",
        },
    )
    print(ROOT, flush=True)


if __name__ == "__main__":
    prepare()
