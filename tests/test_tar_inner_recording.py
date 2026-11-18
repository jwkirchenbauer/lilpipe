"""Test observation at completed inner optimizer steps using a stochastic toy model."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tamper-resistance"


def test_step_observation_preserves_inner_and_outer_training(monkeypatch, tmp_path):
    pytest.importorskip("axolotl")
    monkeypatch.syspath_prepend(str(EXAMPLE / "src"))
    monkeypatch.syspath_prepend(str(EXAMPLE / "scripts/misc"))
    from safetensors.torch import save_file
    from tar_inner_recording import RecordedInnerTARMETrainer, RecordedTARMETrainer
    from trainers.tr_tar import ROLE_FIELDS, adamw_attack

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.dropout = torch.nn.Dropout(0.25)

        def forward(self, input_ids, attention_mask, labels):
            values = self.dropout(input_ids.float()) * self.weight
            return SimpleNamespace(
                loss=((values - labels) * attention_mask).square().mean()
            )

    class Trainer(RecordedInnerTARMETrainer):
        def compute_retain_loss(self, model, inputs):
            return model(**inputs).loss

        compute_forget_loss = compute_retain_loss

        def loss_metrics(self, forget, retain):
            return {"forget": forget.item(), "retain": retain.item()}

    inputs = {key: torch.arange(1, 11).reshape(5, 2) for key in ROLE_FIELDS}
    inputs["attention_mask"].fill_(1)
    inputs["metadata_ids"] = torch.tensor([0, 2, 1, 2, 2])
    original = {key: tensor.clone() for key, tensor in inputs.items()}
    archive = tmp_path / "archived"
    archive.mkdir()
    save_file({"weight": torch.tensor([1.0, 2.0])}, archive / "model.safetensors")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "references": [
                    {"name": "fixed", "anchor": str(archive), "path": str(archive)}
                ]
            }
        )
    )
    output = tmp_path / "observations"
    monkeypatch.setenv("TAR_INNER_REFERENCE_SPEC", str(spec))
    monkeypatch.setenv("TAR_INNER_RECORD_ROOT", str(output))
    monkeypatch.setenv("TAR_INNER_OUTER_STEPS", "3")
    monkeypatch.setenv("TAR_INNER_SAVE_STEPS", "1,2")

    def init(trainer, inner):
        trainer.sample_attack_fn = inner
        trainer.state = SimpleNamespace(global_step=2)
        trainer.args = SimpleNamespace(world_size=1)

    monkeypatch.setattr(RecordedTARMETrainer, "__init__", init)

    def run(observe):
        torch.manual_seed(814)
        model = Model()
        trainer = object.__new__(Trainer)
        trainer._device_debug = False
        trainer.accelerator = SimpleNamespace(autocast=nullcontext)
        trainer.metadata_to_idx = {
            "defence::forget": 0,
            "defence::retain": 1,
            "attack::forget": 2,
        }
        trainer.log = lambda metrics: None
        losses = []
        trainer._log_inner_loss = lambda step, loss: losses.append((step, loss.item()))
        final_inner = []
        observed = []

        def inner(model_copy, rows, **kwargs):
            observer = kwargs.pop("on_step_end", None)
            if observer:

                def capture(completed):
                    observer(completed)
                    observed.append(
                        (completed, model_copy.weight.detach().norm().item())
                    )

                kwargs["on_step_end"] = capture
            result = adamw_attack(
                model_copy, rows, attack_forget_batch_size_per_iteration=2, **kwargs
            )
            assert result is model_copy and result.training
            final_inner.append(result.weight.detach().clone())
            return result

        if observe:
            trainer.__init__(inner)
        else:
            init(trainer, inner)
        loss = trainer.compute_loss(model, inputs)
        loss.backward()
        assert model.training
        for key in original:
            assert torch.equal(inputs[key], original[key])
        return (
            losses,
            final_inner[0],
            loss.detach(),
            model.weight.grad,
            torch.get_rng_state(),
            observed,
        )

    before = torch.get_rng_state()
    try:
        baseline = run(False)
        recorded = run(True)
    finally:
        torch.set_rng_state(before)
    assert baseline[0] == recorded[0]
    assert [step for step, _ in recorded[0]] == [0, 1]
    for left, right in zip(baseline[1:5], recorded[1:5], strict=True):
        assert torch.equal(left, right)
    assert baseline[-1] == []
    assert [step for step, _ in recorded[-1]] == [1, 2]
    assert recorded[-1][-1][1] == recorded[1].norm().item()
    [path] = output.iterdir()
    assert path.name == "outer-000003.json"
    assert json.loads(path.read_text())["rng_unchanged_during_observation"] is True


def test_grams_match_direct_endpoint_arithmetic_and_refuse_duplicate_or_incomplete(
    monkeypatch, tmp_path
):
    pytest.importorskip("axolotl")
    monkeypatch.syspath_prepend(str(EXAMPLE / "src"))
    monkeypatch.syspath_prepend(str(EXAMPLE / "scripts/misc"))
    from safetensors.torch import save_file
    from tar_inner_recording import TransientInnerRecorder
    from trainers.tr_tar import ROLE_FIELDS

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.norm = torch.nn.LayerNorm(2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(247)
        model = Model()
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.tensor([[0.375, -0.125], [0.5, 0.75]]))
    base = {name: param.detach().clone() for name, param in model.named_parameters()}
    archived = {name: value + 0.25 for name, value in base.items()}
    changes = [2**-11, -(2**-6)]
    fixed = [
        {name: value + change for name, value in archived.items()} for change in changes
    ]
    for idx, weights in enumerate([archived, *fixed]):
        directory = tmp_path / str(idx)
        directory.mkdir()
        save_file(weights, directory / "model.safetensors")
    refs = [
        {
            "name": f"fixed-{i}",
            "anchor": str(tmp_path / "0"),
            "path": str(tmp_path / str(i)),
        }
        for i in [1, 2]
    ]
    root = tmp_path / "observations"
    recorder = TransientInnerRecorder(
        root, [1, 3], refs, chunk_elements=2, reduction_device="cpu"
    )
    with pytest.raises(ValueError, match="CPU or CUDA/ROCm"):
        TransientInnerRecorder(root, [1, 3], refs, reduction_device="meta")
    inputs = {key: torch.ones((1, 3), dtype=torch.long) for key in ROLE_FIELDS}
    snapshots = []

    def update(model, rows, on_step_end, steps=3):
        assert rows is inputs
        for step in range(1, steps + 1):
            with torch.no_grad():
                for i, (name, param) in enumerate(model.named_parameters(), 1):
                    param.copy_(base[name] + (step**2) * (i % 2 - 0.5) * 2**-9)
            on_step_end(step)
            if step in (1, 3):
                snapshots.append(
                    {name: p.detach().clone() for name, p in model.named_parameters()}
                )
        return model

    result = recorder.run(update, model, inputs, 5)
    assert result is model
    [path] = root.iterdir()
    assert path.name == "outer-000005.json"
    original = path.read_bytes()
    record = json.loads(original)
    assert record["status"] == "complete" and record["coordinates"] == 8
    assert record["reduction_device"] == "cpu"
    assert record["chunk_elements"] == 2
    assert record["inner_steps"] == [1, 3]
    assert record["reference_names"] == ["fixed-1", "fixed-2"]
    assert {v["name"] for v in record["layout"]} == base.keys()
    assert set(record["families"]) == {"norm", "output_head"}
    for rep in ("bf16", "raw"):

        def flat(values, names):
            x = torch.cat([values[name].flatten().float() for name in names])
            return x.to(torch.bfloat16).float() if rep == "bf16" else x

        def check(part, names):
            inside = torch.stack(
                [flat(weights, names) - flat(base, names) for weights in snapshots]
            ).double()
            outside = torch.stack(
                [flat(weights, names) - flat(archived, names) for weights in fixed]
            ).double()
            measured = part["representations"][rep]
            assert measured["inner_gram"] == (inside @ inside.T).tolist()
            assert measured["reference_gram"] == (outside @ outside.T).tolist()
            assert measured["cross_gram"] == (inside @ outside.T).tolist()
            assert measured["inner_norms"] == pytest.approx(
                inside.norm(dim=1).tolist(), rel=1e-14
            )
            assert measured["reference_norms"] == pytest.approx(
                outside.norm(dim=1).tolist(), rel=1e-14
            )

        check(record, list(base))
        for family, prefix in [("norm", "norm."), ("output_head", "lm_head.")]:
            check(
                record["families"][family],
                [name for name in base if name.startswith(prefix)],
            )
        gap = (flat(base, list(base)) - flat(archived, list(base))).double()
        assert (
            record["representations"][rep]["anchor_gap_squared_norm"]
            == torch.dot(gap, gap).item()
        )

    with pytest.raises(FileExistsError):
        recorder.run(update, model, inputs, 5)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="Expected inner steps"):
        recorder.run(update, model, inputs, 8, steps=1)
    assert sorted(p.name for p in root.iterdir()) == [
        "outer-000005.json",
        "outer-000008.partial",
    ]
    with pytest.raises(FileExistsError):
        recorder.run(update, model, inputs, 8)
