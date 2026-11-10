"""Cache inputs, materialize configs, and compare one Tuolumne TAR reproduction."""

import argparse
import csv
import json
from pathlib import Path


REVISION = "892b3d7a7b1cf10c7a701c60881cd93df615734c"
DEFENSE = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16"
ATTACK_RATES = ("1e-5", "1e-6", "5e-6")
TASKS = ("wmdp_bio_robust", "mmlu_no_bio")


def cache_inputs(run):
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from lm_eval.tasks import TaskManager, get_task_dict

    assert torch.cuda.device_count() == 4, torch.cuda.device_count()
    devices = []
    for index in range(4):
        value = torch.ones(16, device=f"cuda:{index}").sum().item()
        assert value == 16
        assert torch.cuda.get_device_properties(index).gcnArchName.startswith("gfx942")
        devices.append(torch.cuda.get_device_name(index))
    print(
        json.dumps(
            {
                "stage": "gpu_check",
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "devices": devices,
            }
        ),
        flush=True,
    )
    api = HfApi()
    model = snapshot_download(
        "HuggingFaceH4/zephyr-7b-beta",
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    datasets = {}
    for name, config, split in (
        ("cais/wmdp-bio-forget-corpus", None, "train"),
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    ):
        revision = api.dataset_info(name).sha
        dataset = load_dataset(name, config, split=split, revision=revision)
        datasets[name] = {
            "revision": revision,
            "config": config,
            "split": split,
            "rows": len(dataset),
            "fingerprint": dataset._fingerprint,
        }
    for name in ("EleutherAI/wmdp_bio_robust_mcqa", "cais/mmlu"):
        datasets[name] = {"revision": api.dataset_info(name).sha}
    manager = TaskManager(include_path=str(run / "source" / "lm_eval_tasks"))
    tasks = get_task_dict(list(TASKS), task_manager=manager)
    info = {
        "model": model,
        "model_revision": REVISION,
        "datasets": datasets,
        "task_groups": [str(key) for key in tasks],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "devices": devices,
    }
    (run / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


def materialize(run, example):
    import yaml

    inputs = json.loads((run / "inputs.json").read_text())
    model = inputs["model"]
    sources = [
        (
            "defense",
            example / "configs/stages-wmdp/0_unlearn" / DEFENSE / "selected.yml",
        ),
    ]
    sources.extend(
        (
            f"attack-lr{rate}",
            example
            / "configs/stages-wmdp/2_relearn"
            / (DEFENSE + f"--relearn-fft-lr{rate}-ms128")
            / "selected.yml",
        )
        for rate in ATTACK_RATES
    )
    for stage, source in sources:
        config = yaml.safe_load(source.read_text())
        config.update(
            base_model=model if stage == "defense" else str(run / "models/defense"),
            output_dir=str(run / "models" / stage),
            dataset_prepared_path=str(run / "cache/axolotl" / stage),
            wandb_project="jwk-tar-reproduction",
            wandb_name=f"{run.name}-{stage}",
        )
        for dataset in config["datasets"]:
            revision = inputs["datasets"][dataset["path"]]["revision"]
            dataset["revision"] = revision
        (run / "configs" / f"{stage}.yml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        if stage == "defense":
            config.update(
                max_steps=1,
                warmup_steps=0,
                save_steps=1,
                save_total_limit=1,
                output_dir=str(run / "models/smoke"),
                wandb_name=f"{run.name}-smoke",
            )
            (run / "configs/smoke.yml").write_text(
                yaml.safe_dump(config, sort_keys=False)
            )
        (run / "configs" / f"{stage}.reference.yml").write_text(source.read_text())


def scores(directory):
    found = {}
    for file in sorted(directory.rglob("results*.json")):
        document = json.loads(file.read_text())
        if document.get("config", {}).get("limit") is not None:
            continue
        for task in TASKS:
            if task in document.get("results", {}):
                value = 100 * document["results"][task]["acc,none"]
                if task in found and abs(found[task] - value) > 1e-10:
                    raise ValueError(f"Conflicting scores: {directory}, {task}")
                found[task] = value
    if set(found) != set(TASKS):
        raise ValueError(f"Incomplete evaluation: {directory}: {list(found)}")
    return found


def compare(run, example, base_only=False, allow_partial=False):
    reference = run / "reference"
    baseline = scores(reference / "z7b-base")
    local_base = scores(run / "evals/base")
    base_delta = {task: local_base[task] - baseline[task] for task in TASKS}
    if base_only:
        comparison = {
            "reference": baseline,
            "jwk": local_base,
            "difference_pp": base_delta,
        }
        (run / "base_comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        print(json.dumps(base_delta, indent=2), flush=True)
        if max(abs(value) for value in base_delta.values()) > 1:
            raise ValueError(
                "Base evaluation differs by more than 1 percentage point; "
                "inspect before training"
            )
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    completed = []
    defense_reference = scores(reference / DEFENSE / "checkpoint-125")
    for rate in ATTACK_RATES:
        rate_root = run / "evals" / f"attack-lr{rate}"
        if not (rate_root / "checkpoint-128").exists() and allow_partial:
            continue
        for step in range(0, 129, 16):
            local_dir = (
                run / "evals/checkpoint-0"
                if step == 0
                else rate_root / f"checkpoint-{step}"
            )
            local = scores(local_dir)
            for task in TASKS:
                ref = defense_reference[task] if step == 0 else None
                rows.append(
                    {
                        "learning_rate": rate,
                        "attack_step": step,
                        "task": task,
                        "jwk_percent": local[task],
                        "reference_percent": ref,
                        "difference_pp": local[task] - ref if ref is not None else None,
                    }
                )
        completed.append(rate)
    if not completed:
        raise ValueError("No complete attack trajectory is available")
    with (run / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for task, axis in zip(TASKS, axes):
        for rate in completed:
            selected = [
                row
                for row in rows
                if row["task"] == task and row["learning_rate"] == rate
            ]
            axis.plot(
                [row["attack_step"] for row in selected],
                [row["jwk_percent"] for row in selected],
                "o-",
                label=f"Relearning LR {rate}",
            )
        axis.axhline(
            local_base[task], color="black", linewidth=1, label="Untouched Zephyr"
        )
        axis.set(xlabel="Relearning step", ylabel="Accuracy (%)", title=task)
        axis.legend(fontsize=8)
    fig.savefig(run / "trajectory.pdf")
    fig.savefig(run / "trajectory.png", dpi=180)
    summary = {
        "stage": "complete" if len(completed) == len(ATTACK_RATES) else "attacks",
        "completed_learning_rates": completed,
        "base_difference_pp": base_delta,
        "defense_reference_percent": defense_reference,
        "attack_config_family": "collaborator_named_selected_configs",
        "trajectory": rows,
    }
    defense_local = scores(run / "evals/checkpoint-0")
    report = [
        "# TAR reproduction on Tuolumne",
        "",
        "Tamper-resistant unlearning (TAR), with one shared 125-step defense "
        "and independent 128-step relearning attacks using the collaborator's "
        "named selected configs. TAR-k16 uses 16 simulated attack steps per "
        "defense update.",
        "",
        f"Completed learning rates: {', '.join(completed)}.",
        "",
        "All scores are full-dataset, zero-shot accuracy in percent. "
        "WMDP-Bio robust measures biology knowledge; MMLU-no-bio measures "
        "retained general capabilities.",
        "",
        "| Model state | Attack LR | Attack step | WMDP-Bio robust | MMLU-no-bio |",
        "|---|---|---:|---:|---:|",
        f"| Untouched Zephyr | — | — | {local_base[TASKS[0]]:.2f} | "
        f"{local_base[TASKS[1]]:.2f} |",
        f"| Shared TAR defense | — | 0 | {defense_local[TASKS[0]]:.2f} | "
        f"{defense_local[TASKS[1]]:.2f} |",
    ]
    for rate in completed:
        for step in (16, 128):
            point = {
                row["task"]: row["jwk_percent"]
                for row in rows
                if row["learning_rate"] == rate and row["attack_step"] == step
            }
            report.append(
                f"| Relearning | {rate} | {step} | {point[TASKS[0]]:.2f} | "
                f"{point[TASKS[1]]:.2f} |"
            )
    report.extend(
        [
            "",
            "The repository's reference scores are 55.18 / 58.59 for untouched "
            "Zephyr and 31.34 / 53.45 for the selected defense (WMDP / MMLU). "
            "The reference defense was promoted by exception to its nominal "
            "five-percentage-point capability-drop budget.",
            "",
            "Each attack starts fresh from the shared defense. Training settings "
            "are preserved from the named configs; runtime paths, logging "
            "destinations, and input revisions are recorded in `configs/` and "
            "`provenance.json`. Defense seed/data seed: 42/42. Attack seed/data "
            "seed: 1/42. Hardware: one node, four AMD MI300A GPUs.",
            "",
            "[All trajectory scores](comparison.csv) · "
            "[Trajectory figure](trajectory.pdf) · [PNG](trajectory.png)",
            "",
        ]
    )
    (run / "report.md").write_text("\n".join(report))
    (run / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inputs", "configs", "base-check", "compare"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--example-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run = args.run_dir.resolve()
    example = args.example_root.resolve()
    if args.stage == "inputs":
        cache_inputs(run)
    elif args.stage == "configs":
        materialize(run, example)
    else:
        compare(
            run,
            example,
            base_only=args.stage == "base-check",
            allow_partial=args.allow_partial,
        )


if __name__ == "__main__":
    main()
