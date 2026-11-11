"""Matched simulated/real attacks and controlled TAR objective continuations.

This is a finite-corpus mechanism experiment, not a full-corpus reproduction.
Every loss and optimizer update uses FP32. No mixed-precision evaluation cast
is allowed. Full batch means all documents in the declared eight-row corpus.
"""

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from types import MethodType


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(args):
    """Freeze sources and declare all comparisons before observing outcomes."""
    import subprocess
    import yaml

    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    example = Path(__file__).resolve().parents[2]
    original = args.original.resolve()
    anchor = args.anchor.resolve()
    config = yaml.safe_load((original / "configs/defense.yml").read_text())
    for folder in ("source", "cases", "logs", "status", "launches"):
        (run / folder).mkdir()
    for filename in (
        "tar_cleanroom.py",
        "run_tar_cleanroom.sh",
        "tar_cleanroom_gradients.py",
    ):
        path = Path(__file__).with_name(filename)
        if path.exists():
            shutil.copy2(path, run / "source" / filename)
    shutil.copytree(original / "source/lm_eval_tasks", run / "source/lm_eval_tasks")
    cases = {}
    for mode, batch_size in (("full", 8), ("mini", 2)):
        for objective in ("maxent-terminal", "ce-terminal", "ce-worst"):
            name = f"{mode}-{objective}"
            cases[name] = {
                "name": name,
                "mode": mode,
                "objective": objective,
                "attack_batch_size": batch_size,
                "inner_steps": 4,
                "outer_steps": 125,
                "outer_learning_rate": 5e-6,
                "retain_coefficient": 2.5,
                "seed": 42,
            }
            (run / "cases" / name).mkdir()
    study = {
        "schema": 1,
        "purpose": "Separate attack fidelity, objective specification, and attack generalization",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=example, text=True
        ).strip(),
        "anchor": str(anchor),
        "base_model": config["base_model"],
        "original_run": str(original),
        "dtype": "float32",
        "attention": "eager",
        "sequence_length": 256,
        "forget_documents": 8,
        "new_attack_documents": 8,
        "heldout_documents": 32,
        "retain_documents": 8,
        "retain_heldout_documents": 16,
        "data_seed": 20261111,
        "forget_revision": config["datasets"][0]["revision"],
        "retain_revision": config["datasets"][-1]["revision"],
        "device_map": config["device_map"],
        "attack": {
            "learning_rate": 1e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "schedule": "constant",
            "steps": 64,
        },
        "outer_schedule": "linear warmup12 then decay to zero over125",
        "outer_gradient": "identity Jacobian through detached attack displacement in every arm",
        "ce_floor": "log(vocabulary size)",
        "retention": "weighted CE training regularizer; benchmark utility gate is separate",
        "mmlu_floor_percent": 53.58729288251205,
        "cases": cases,
        "defense_checkpoints": [32, 125],
        "attack_checkpoints": [1, 4, 16, 64],
        "attack_variants": [
            "full-same",
            "mini-same",
            "mini-reverse",
            "full-new",
        ],
        "primary_contrasts": [
            "ce-terminal minus maxent-terminal within each regime: objective",
            "ce-worst minus ce-terminal within each regime: trajectory coverage",
            "matched versus mismatched evaluation within each defender: generalization",
        ],
        "limitations": [
            "Fixed small forget corpus; full batch does not mean the complete source dataset",
            "Defense continuation from original K16 checkpoint, not training from base",
            "Same update count across regimes; full batch processes four times more inner tokens",
            "Single declared corpus/optimizer seed initially; no statistical significance claim",
            "FP32 regime differs from original mixed parameter precision",
            "Current inner AdamW default wd0.01/no clipping differs from this explicit wd0/clip1 regime",
        ],
    }
    dump(run / "study.json", study)
    seal = {
        str(path.relative_to(run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((run / "source").rglob("*"))
        if path.is_file()
    }
    dump(run / "source-manifest.json", seal)
    print(run, flush=True)


def load_study(run):
    for relative, expected in json.loads(
        (run / "source-manifest.json").read_text()
    ).items():
        if hashlib.sha256((run / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    return json.loads((run / "study.json").read_text())


def prepare_data(run, study):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    target = run / "data.json"
    if target.exists():
        data = json.loads(target.read_text())
        expected = data.pop("content_sha256")
        assert digest(data) == expected
        data["content_sha256"] = expected
        sequences = [
            digest(row["input_ids"])
            for group, rows in data.items()
            if isinstance(rows, list)
            for row in rows
        ]
        if len(set(sequences)) != len(sequences):
            raise ValueError("Duplicate token sequences across declared data cohorts")
        return data
    tokenizer = AutoTokenizer.from_pretrained(study["anchor"])
    forget = load_dataset(
        "cais/wmdp-bio-forget-corpus",
        split="train",
        revision=study["forget_revision"],
    )
    retain = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
        revision=study["retain_revision"],
    )

    def choose(dataset, count, seed):
        order = list(range(len(dataset)))
        random.Random(seed).shuffle(order)
        selected, seen = [], set()
        for row_id in order:
            text = dataset[row_id]["text"]
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            ids = tokenizer(
                text,
                truncation=True,
                max_length=study["sequence_length"],
            )["input_ids"]
            if len(ids) != study["sequence_length"] or text_hash in seen:
                continue
            seen.add(text_hash)
            selected.append(
                {
                    "source_row": row_id,
                    "source_text_sha256": text_hash,
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": ids,
                }
            )
            if len(selected) == count:
                return selected
        raise ValueError("Insufficient distinct full-length documents")

    forget_rows = choose(forget, 48, study["data_seed"])
    retain_rows = choose(retain, 24, study["data_seed"] + 1)
    data = {
        "selection": "seeded shuffled source rows; distinct text hashes; first256 tokens including tokenizer special tokens",
        "forget_train": forget_rows[:8],
        "forget_new_attack": forget_rows[8:16],
        "forget_heldout": forget_rows[16:],
        "retain_train": retain_rows[:8],
        "retain_heldout": retain_rows[8:],
    }
    data["content_sha256"] = digest(data)
    dump(target, data)
    return data


def runtime(study, checkpoint=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 4
    assert torch.backends.cuda.preferred_blas_library() == torch._C._BlasBackend.Cublas
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint or study["anchor"],
        dtype=torch.float32,
        device_map=study["device_map"],
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.eval()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    actual = {}
    for name, expected in study["device_map"].items():
        devices = {str(p.device) for p in model.get_submodule(name).parameters()}
        assert devices == {f"cuda:{expected}"}, (name, devices)
        actual[name] = sorted(devices)
    print(json.dumps({"device_assignment": actual, "dtype": "float32"}), flush=True)
    return model, AutoTokenizer.from_pretrained(study["anchor"])


def batch(rows, device):
    import torch

    return {
        key: torch.tensor([row[key] for row in rows], device=device)
        for key in ("input_ids", "attention_mask", "labels")
    }


def loss_statistics(forward, rows, per_sequence=False):
    """Mean CE/entropy weighted by supervised next tokens, independent of chunks."""
    import torch
    import torch.nn.functional as F

    outputs = forward(**rows)
    logits = outputs.logits[:, :-1].float()
    labels = rows["labels"][:, 1:].to(logits.device)
    valid = labels != -100
    n = valid.sum()
    if not n:
        raise ValueError("No supervised next tokens")
    token_ce = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), reduction="none"
    ).reshape_as(labels)
    ce = token_ce.sum() / n
    log_probs = torch.log_softmax(logits, -1)
    token_entropy = -(log_probs.exp() * log_probs).sum(-1)
    entropy = token_entropy[valid].mean()
    if per_sequence:
        return (
            ce,
            entropy,
            (token_ce.sum(-1) / valid.sum(-1)),
            (token_entropy * valid).sum(-1) / valid.sum(-1),
        )
    return ce, entropy


def evaluate(model, rows, microbatch=2):
    import torch

    count = sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)
    ce, entropy, sequence_ce, sequence_entropy = 0.0, 0.0, [], []
    with torch.no_grad():
        for start in range(0, len(rows), microbatch):
            inputs = batch(rows[start : start + microbatch], model.device)
            n = int((inputs["labels"][:, 1:] != -100).sum())
            c, h, cs, hs = loss_statistics(model, inputs, per_sequence=True)
            ce += float(c) * n / count
            entropy += float(h) * n / count
            sequence_ce.extend(cs.cpu().tolist())
            sequence_entropy.extend(hs.cpu().tolist())
    return {
        "ce": ce,
        "entropy": entropy,
        "tokens": count,
        "per_sequence_ce": sequence_ce,
        "per_sequence_entropy": sequence_entropy,
        "ce_quantiles": torch.quantile(
            torch.tensor(sequence_ce), torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
        ).tolist(),
    }


def parameter_norm(values):
    import torch

    terms = [value.detach().float().square().sum().double().cpu() for value in values]
    return float(torch.stack(terms).sum().sqrt())


def clip_gradients(model, maximum):
    """One global clipping coefficient across the explicit four-GPU layer map."""
    norm = parameter_norm(p.grad for p in model.parameters() if p.grad is not None)
    scale = min(1.0, maximum / (norm + 1e-6))
    if scale < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
    return norm


def optimizer(model, settings):
    import torch

    return torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
        foreach=False,
        fused=False,
    )


def attack_rows(rows, size, step, reverse=False):
    order = list(reversed(rows)) if reverse else rows
    start = (step * size) % len(order)
    return [order[(start + i) % len(order)] for i in range(size)]


def attack_step(model, opt, rows, settings):
    """Simulator: accumulate the token mean then take one optimizer step."""
    import torch.nn.functional as F

    opt.zero_grad(set_to_none=True)
    total = sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)
    loss = 0.0
    for start in range(0, len(rows), 2):
        inputs = batch(rows[start : start + 2], model.device)
        logits = model(**inputs).logits[:, :-1].float()
        labels = inputs["labels"][:, 1:].to(logits.device)
        term = (
            F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            / total
        )
        term.backward()
        loss += float(term.detach())
    norm = clip_gradients(model, settings["max_grad_norm"])
    opt.step()
    return {"batch_ce_before": loss, "gradient_norm": norm, "batch_tokens": total}


def real_attack_step(model, opt, rows, settings):
    """Independent real-attacker implementation: one whole-batch CE backward."""
    import torch
    import torch.nn.functional as F

    opt.zero_grad(set_to_none=True)
    inputs = batch(rows, model.device)
    logits = model(**inputs).logits[:, :-1].float()
    labels = inputs["labels"][:, 1:].to(logits.device)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    # Match the simulator's globally accumulated norm without calling its clip.
    squares = sum(
        float(p.grad.detach().double().square().sum().cpu())
        for p in model.parameters()
        if p.grad is not None
    )
    norm = math.sqrt(squares)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(min(1.0, settings["max_grad_norm"] / (norm + 1e-6)))
    opt.step()
    return {"batch_ce_before": float(loss.detach()), "gradient_norm": norm}


def clone_model(model):
    result = copy.deepcopy(model)
    result.forward = MethodType(type(result).forward, result)
    return result


def append(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def save_model(model, tokenizer, path):
    path = Path(path)
    assert not path.exists()
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise ValueError(f"Unfinished checkpoint already exists: {temporary}")
    model.save_pretrained(temporary, max_shard_size="4GB", safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    dump(
        temporary / "checkpoint-complete.json",
        {"state": "complete", "dtype": "float32"},
    )
    temporary.replace(path)


def provenance(run, data):
    import torch

    return {
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "study_sha256": hashlib.sha256((run / "study.json").read_bytes()).hexdigest(),
        "data_sha256": data["content_sha256"],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "dtype": "float32",
    }


def parity(run, study, data):
    """Compare independent implementations and optimizer states on actual 7B."""
    import torch

    model, _ = runtime(study)
    initial = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
    results = []
    for size in (8, 2):
        simulated = clone_model(model)
        sim = optimizer(simulated, study["attack"])
        real = optimizer(model, study["attack"])
        for step in range(4):
            reference = attack_step(
                simulated,
                sim,
                attack_rows(data["forget_train"], size, step),
                study["attack"],
            )
            actual = real_attack_step(
                model,
                real,
                attack_rows(data["forget_train"], size, step),
                study["attack"],
            )
            parameter_error, state_error, parameter_scale = 0.0, 0.0, 0.0
            error_sq, update_sq = 0.0, 0.0
            for (name, p), (other_name, saved) in zip(
                model.named_parameters(), simulated.named_parameters(), strict=True
            ):
                assert name == other_name
                difference = p.detach() - saved.detach()
                parameter_error = max(parameter_error, float(difference.abs().max()))
                parameter_scale = max(parameter_scale, float(saved.abs().max()))
                error_sq += float(difference.double().square().sum().cpu())
                update_sq += float(
                    (saved.detach() - initial[name].to(saved.device))
                    .double()
                    .square()
                    .sum()
                    .cpu()
                )
                for key, value in real.state[p].items():
                    state_error = max(
                        state_error,
                        float((value.detach() - sim.state[saved][key]).abs().max()),
                    )
            row = {
                "batch_size": size,
                "step": step + 1,
                "parameter_max_abs_error": parameter_error,
                "parameter_max_abs_scale": parameter_scale,
                "optimizer_state_max_abs_error": state_error,
                "update_relative_l2_error": math.sqrt(error_sq / max(update_sq, 1e-30)),
                "ce_before_error": abs(
                    actual["batch_ce_before"] - reference["batch_ce_before"]
                ),
                "simulated": reference,
                "actual": actual,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        del real, sim, simulated
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(initial[name])
    # Different GEMM batch shapes can introduce FP32 numerical differences.
    # Report the error rather than silently claiming bitwise identity.
    passed = all(
        r["update_relative_l2_error"] < 0.01 and r["ce_before_error"] < 1e-3
        for r in results
    )
    dump(
        run / "parity.json",
        {"state": "passed" if passed else "failed", "rows": results},
    )
    if not passed:
        raise RuntimeError("Independent simulator/attacker numerical parity failed")


def outer_backward(model, selected, rows, objective, coefficient=1.0):
    """The same straight-through attack displacement convention for every arm."""
    from torch.func import functional_call

    total = sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)
    for start in range(0, len(rows), 2):
        inputs = batch(rows[start : start + 2], model.device)
        n = int((inputs["labels"][:, 1:] != -100).sum())
        point = {
            name: p + (selected[name].to(p.device) - p).detach()
            for name, p in model.named_parameters()
        }
        ce, entropy = loss_statistics(
            lambda **x: functional_call(model, point, (), x), inputs
        )
        term = -entropy if objective == "maxent-terminal" else -ce
        (coefficient * n / total * term).backward()


def train(run, study, data, name, steps=None, until=None, resume=0):
    import torch

    case = dict(study["cases"][name])
    case["outer_steps"] = steps or case["outer_steps"]
    folder = run / "cases" / name
    if steps is not None:
        folder = run / "pilot" / name
    folder.mkdir(parents=True, exist_ok=True)
    until = until or case["outer_steps"]
    assert 0 <= resume < until <= case["outer_steps"]
    if resume:
        prior = [
            json.loads(line)
            for line in (folder / "defense.jsonl").read_text().splitlines()
        ]
        assert (
            prior[-1]["outer_step"] == resume
        ), "Resume must follow the last recorded update"
        assert json.loads((folder / "case.json").read_text()) == case
        checkpoint = folder / f"defense-{resume}"
        assert (checkpoint / "checkpoint-complete.json").exists()
    else:
        assert not (
            folder / "defense.jsonl"
        ).exists(), "Refuse to overwrite a started case"
        prior, checkpoint = [], None
        dump(folder / "case.json", case)
        dump(folder / "provenance.json", provenance(run, data))
    segment = folder / "segments" / f"{resume + 1}-{until}"
    dump(segment / "provenance.json", provenance(run, data))
    model, tokenizer = runtime(study, checkpoint)
    floor = math.log(model.config.vocab_size)
    outer = torch.optim.AdamW(
        model.parameters(),
        lr=case["outer_learning_rate"],
        weight_decay=0,
        foreach=False,
    )
    if resume:
        state = torch.load(
            folder / f"optimizer-{resume}.pt", map_location="cpu", weights_only=True
        )
        assert (
            state["step"] == resume and state["data_sha256"] == data["content_sha256"]
        )
        outer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        del state
    started = time.monotonic()
    offset = prior[-1]["elapsed_seconds"] if prior else 0.0
    for step in range(resume + 1, until + 1):
        sim = clone_model(model)
        inner = optimizer(sim, study["attack"])
        initial = evaluate(sim, data["forget_train"])
        path = [{"step": 0, **initial}]
        best_ce, selected_step, selected = initial["ce"], 0, None
        for inner_step in range(case["inner_steps"]):
            record = attack_step(
                sim,
                inner,
                attack_rows(
                    data["forget_train"], case["attack_batch_size"], inner_step
                ),
                study["attack"],
            )
            score = evaluate(sim, data["forget_train"])
            path.append({"step": inner_step + 1, **score, **record})
            if score["ce"] < best_ce:
                best_ce, selected_step = score["ce"], inner_step + 1
                if case["objective"] == "ce-worst":
                    selected = {
                        n: p.detach().clone() for n, p in sim.named_parameters()
                    }
        if case["objective"] != "ce-worst":
            selected_step = case["inner_steps"]
            selected = {n: p.detach().clone() for n, p in sim.named_parameters()}
        elif selected is None:
            selected = {n: p.detach().clone() for n, p in model.named_parameters()}
        del inner, sim
        gc.collect()
        outer.zero_grad(set_to_none=True)
        active = (
            case["objective"] == "maxent-terminal" or path[selected_step]["ce"] < floor
        )
        if active:
            outer_backward(model, selected, data["forget_train"], case["objective"])
        del selected
        # Retention gradients are evaluated at the undeflected defender.
        retain_tokens = sum(len(row["labels"]) - 1 for row in data["retain_train"])
        for start in range(0, len(data["retain_train"]), 2):
            rows = data["retain_train"][start : start + 2]
            ce, _ = loss_statistics(model, batch(rows, model.device))
            weight = sum(len(row["labels"]) - 1 for row in rows) / retain_tokens
            (case["retain_coefficient"] * weight * ce).backward()
        norm = clip_gradients(model, 1.0)
        # Same fixed125-step schedule even in a short engineering pilot.
        factor = min(step / 12, max(0.0, (125 - step) / (125 - 12)))
        outer.param_groups[0]["lr"] = case["outer_learning_rate"] * factor
        outer.step()
        row = {
            "outer_step": step,
            "objective": case["objective"],
            "selected_inner_step": selected_step,
            "ce_floor": floor,
            "forget_objective_active": active,
            "gradient_norm_before_clip": norm,
            "learning_rate": outer.param_groups[0]["lr"],
            "simulated_path": path,
            "inner_tokens": sum(p["batch_tokens"] for p in path[1:]),
            "elapsed_seconds": offset + time.monotonic() - started,
        }
        if step in {1, 8, 32, 64, case["outer_steps"]}:
            row["defender_probes"] = {
                group: evaluate(model, data[group])
                for group in (
                    "forget_train",
                    "forget_heldout",
                    "retain_train",
                    "retain_heldout",
                )
            }
        append(folder / "defense.jsonl", row)
        if steps is None and step in {*study["defense_checkpoints"], 64, 96, until}:
            save_model(model, tokenizer, folder / f"defense-{step}")
            # Explicit chunk boundaries permit exact continuation in the
            # one-hour debug queue without resetting the outer Adam moments.
            temporary = folder / f"optimizer-{step}.pt.tmp"
            torch.save(
                {
                    "step": step,
                    "data_sha256": data["content_sha256"],
                    "optimizer": outer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                temporary,
            )
            temporary.replace(folder / f"optimizer-{step}.pt")
            dump(folder / f"resume-{step}.json", {"state": "complete", "step": step})
    dump(
        segment / "runtime.json",
        {
            "seconds": time.monotonic() - started,
            "first_step": resume + 1,
            "last_step": until,
        },
    )
    dump(
        folder / "training.json",
        {
            "state": "complete" if until == case["outer_steps"] else "partial",
            "outer_steps": case["outer_steps"],
            "last_step": until,
            "seconds": offset + time.monotonic() - started,
        },
    )


def attack(run, study, data, name, variant):
    import torch

    folder = run / "cases" / name
    checkpoint = study["anchor"] if name == "control" else folder / "defense-125"
    target = folder / "attacks" / variant
    target.mkdir(parents=True, exist_ok=True)
    assert not (target / "trace.jsonl").exists()
    dump(target / "provenance.json", provenance(run, data))
    model, tokenizer = runtime(study, checkpoint)
    opt = optimizer(model, study["attack"])
    size = 8 if variant.startswith("full") else 2
    rows = (
        data["forget_new_attack"] if variant.endswith("new") else data["forget_train"]
    )
    reverse = variant.endswith("reverse")
    # Repeat the simulator comparison at each new defended checkpoint. A
    # baseline-only parity check would miss numerical amplification introduced
    # by the defense continuation itself.
    shadow = clone_model(model)
    shadow_opt = optimizer(shadow, study["attack"])
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    started = time.monotonic()
    for step in range(study["attack"]["steps"] + 1):
        record = {
            "step": step,
            "examples_processed": step * size,
            "tokens_processed": step * size * (study["sequence_length"] - 1),
        }
        if step:
            # External updates use the independently implemented full-batch path.
            record.update(
                real_attack_step(
                    model,
                    opt,
                    attack_rows(rows, size, step - 1, reverse),
                    study["attack"],
                )
            )
            if step <= 4:
                attack_step(
                    shadow,
                    shadow_opt,
                    attack_rows(rows, size, step - 1, reverse),
                    study["attack"],
                )
                error_sq, update_sq = 0.0, 0.0
                maximum = 0.0
                with torch.no_grad():
                    for (n, p), (other, q) in zip(
                        model.named_parameters(), shadow.named_parameters(), strict=True
                    ):
                        assert n == other
                        difference = p - q
                        maximum = max(maximum, float(difference.abs().max()))
                        error_sq += float(difference.double().square().sum().cpu())
                        update_sq += float(
                            (q - initial[n].to(q.device)).double().square().sum().cpu()
                        )
                record["simulator_parameter_max_abs_error"] = maximum
                record["simulator_update_relative_l2_error"] = math.sqrt(
                    error_sq / max(update_sq, 1e-30)
                )
                if step == 4:
                    record["simulator_endpoint_probe"] = evaluate(
                        shadow, data["forget_train"]
                    )
                    del shadow, shadow_opt, initial
                    gc.collect()
        if step in {0, 1, 2, 4, 8, 16, 32, 64}:
            record["probes"] = {
                group: evaluate(model, data[group])
                for group in (
                    "forget_train",
                    "forget_new_attack",
                    "forget_heldout",
                    "retain_heldout",
                )
            }
        record["elapsed_seconds"] = time.monotonic() - started
        append(target / "trace.jsonl", record)
        if step in study["attack_checkpoints"]:
            save_model(model, tokenizer, target / f"checkpoint-{step}")
    dump(
        target / "attack.json",
        {
            "state": "complete",
            "variant": variant,
            "seconds": time.monotonic() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "data", "parity", "train", "attack")
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--case")
    parser.add_argument(
        "--variant", choices=("full-same", "mini-same", "mini-reverse", "full-new")
    )
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--until-step", type=int)
    parser.add_argument("--resume-step", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
        return
    run = args.run.resolve()
    study = load_study(run)
    data = prepare_data(run, study)
    if args.mode == "data":
        print(data["content_sha256"], flush=True)
    elif args.mode == "parity":
        parity(run, study, data)
    elif args.mode == "train":
        if json.loads((run / "parity.json").read_text())["state"] != "passed":
            raise ValueError("Actual-model update parity must pass before training")
        train(
            run,
            study,
            data,
            args.case,
            args.pilot_steps,
            args.until_step,
            args.resume_step,
        )
    elif args.mode == "attack":
        attack(run, study, data, args.case, args.variant)


if __name__ == "__main__":
    main()
