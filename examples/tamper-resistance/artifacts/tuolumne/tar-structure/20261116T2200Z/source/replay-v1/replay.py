"""Replay saved attack checkpoints by module, using the original BF16 CE probes."""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    assert os.environ.get("FLUX_JOB_ID")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
    )

    assert torch.cuda.device_count() == 1
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    run = args.run.resolve()
    spec = importlib.util.spec_from_file_location(
        "structure_helpers", run / "source/v1/tar_basis_structure.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    destination = run / "module-replay"
    destination.mkdir(exist_ok=True)
    records, validation = [], []
    variants = {
        "all": {"embedding", "output_head", "norm", "attention", "mlp", "other"},
        "embedding": {"embedding"},
        "output_head": {"output_head"},
        "norm": {"norm"},
        "attention": {"attention"},
        "mlp": {"mlp"},
        "attention+mlp": {"attention", "mlp"},
        "embedding+head+norm": {"embedding", "output_head", "norm"},
    }
    for case_name in ("local-defense-attack", "collaborator-attack"):
        case = run / "cases" / case_name
        job = read(case / "job.json")
        snapshots = {item["name"]: item for item in job["snapshots"]}
        anchor = Path(snapshots["attack-1"]["anchor"])
        native_case = Path(snapshots["attack-1000"]["path"]).parents[2]
        settings = read(native_case / "study.json")
        tokenizer = AutoTokenizer.from_pretrained(anchor, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        corpus = load_dataset(
            "cais/wmdp-bio-forget-corpus",
            split="train",
            revision=settings["forget_revision"],
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        probes, probe_hashes = {}, {}
        for label, indices in (
            ("fixed", settings["fixed_source_indices"]),
            ("heldout", settings["heldout_source_indices"]),
        ):
            examples = [
                tokenizer(corpus[index]["text"], max_length=512, truncation=True)
                for index in indices
            ]
            cpu_batches = [
                collator(examples[start : start + 2])
                for start in range(0, len(examples), 2)
            ]
            serialized = [
                {key: value.tolist() for key, value in batch.items()}
                for batch in cpu_batches
            ]
            probe_hashes[label] = hashlib.sha256(
                json.dumps(serialized, sort_keys=True).encode()
            ).hexdigest()
            probes[label] = [
                {key: value.to("cuda:0") for key, value in batch.items()}
                for batch in cpu_batches
            ]
        with (native_case / "surfaces/actual-losses.csv").open(newline="") as stream:
            expected = {
                int(float(row["step"])): row
                for row in csv.DictReader(stream)
                if row["phase"] == "attack"
            }
        model = (
            AutoModelForCausalLM.from_pretrained(
                anchor,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                local_files_only=True,
            )
            .to("cuda:0")
            .eval()
        )
        parameters = dict(model.named_parameters())
        classifications = {
            name: helper.classify_parameter(name)[0] for name in parameters
        }

        @torch.inference_mode()
        def losses():
            return {
                label: float(np.mean([float(model(**batch).loss) for batch in batches]))
                for label, batches in probes.items()
            }

        def check(values, step):
            errors = {
                label: abs(values[label] - float(expected[step][field]))
                for label, field in (
                    ("fixed", "probe_loss"),
                    ("heldout", "heldout_loss"),
                )
            }
            if max(errors.values()) > 1e-4:
                raise ValueError(
                    f"Replay and native probe loss differ: {case_name} {step} {errors}"
                )
            validation.append(
                {
                    "case": case_name,
                    "step": step,
                    "errors": errors,
                    "probe_sha256": probe_hashes,
                }
            )

        with contextlib.ExitStack() as stack:
            base = helper.WeightReader(anchor, stack)
            for step in (1, 1000):
                target = helper.WeightReader(snapshots[f"attack-{step}"]["path"], stack)
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(base.tensor(name))
                before = losses()
                check(before, 0)
                selected_before = set()
                for variant, categories in variants.items():
                    selected = {
                        name
                        for name, category in classifications.items()
                        if category in categories
                    }
                    with torch.no_grad():
                        for name in selected_before - selected:
                            parameters[name].copy_(base.tensor(name))
                        for name in selected:
                            parameters[name].copy_(target.tensor(name))
                    values = losses()
                    if variant == "all":
                        check(values, step)
                    record = {
                        "case": case_name,
                        "attack_step": step,
                        "variant": variant,
                        "parameters_replaced": sum(
                            parameters[name].numel() for name in selected
                        ),
                        "fixed_before": before["fixed"],
                        "fixed_after": values["fixed"],
                        "fixed_ce_reduction": before["fixed"] - values["fixed"],
                        "heldout_before": before["heldout"],
                        "heldout_after": values["heldout"],
                        "heldout_ce_reduction": before["heldout"] - values["heldout"],
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    helper.write_csv(destination / "losses.csv", records)
                    selected_before = selected
        del model, parameters
        torch.cuda.empty_cache()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    labels = list(variants)
    for row, case_name in enumerate(("local-defense-attack", "collaborator-attack")):
        for column, step in enumerate((1, 1000)):
            axis = axes[row, column]
            selected = [
                item
                for item in records
                if item["case"] == case_name and item["attack_step"] == step
            ]
            axis.bar(
                np.arange(len(labels)) - 0.18,
                [item["fixed_ce_reduction"] for item in selected],
                0.36,
                label="fixed10",
            )
            axis.bar(
                np.arange(len(labels)) + 0.18,
                [item["heldout_ce_reduction"] for item in selected],
                0.36,
                label="attack-held-out32",
            )
            axis.set_xticks(
                range(len(labels)), labels, rotation=35, ha="right", fontsize=8
            )
            axis.set(
                title=f"{case_name}; saved update {step}",
                ylabel="CE reduction from defended model (nats)",
            )
            axis.axhline(0, color="black", linewidth=0.7)
            axis.grid(alpha=0.2, axis="y")
            axis.legend()
    fig.suptitle(
        "Saved module-only checkpoint replay in the original BF16 evaluator; composite effects need not add"
    )
    fig.savefig(destination / "module-only-ce-replay.png", dpi=160)
    fig.savefig(destination / "module-only-ce-replay.pdf")
    plt.close(fig)
    helper.publish(
        destination / "validation.json",
        {
            "state": "passed",
            "native_endpoint_checks": validation,
            "interpretation": "Frozen checkpoint composites; this does not train module-constrained attackers or measure partial-composite benchmark recovery. Values can interact and need not sum to the full checkpoint effect.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "flux_job_id": os.environ["FLUX_JOB_ID"],
        },
    )


if __name__ == "__main__":
    main()
