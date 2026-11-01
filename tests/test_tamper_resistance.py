import importlib.util
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tamper-resistance"


def load_script(relative_path: str, module_name: str):
    path = EXAMPLE / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wmdp_di_module():
    pytest.importorskip("axolotl")
    return load_script("src/data/wmdp_di.py", "wmdp_di_training")


@pytest.fixture(scope="module")
def wikitext_di_module():
    pytest.importorskip("axolotl")
    return load_script("src/data/wikitext_di.py", "wikitext_di_training")


@pytest.fixture(scope="module")
def authors_documents_module():
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script(
            "src/data/authors_documents.py", "authors_documents_training"
        )
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def sampling_module():
    pytest.importorskip("torch")
    return load_script("src/trainers/utils.py", "tamper_sampling")


@pytest.fixture(scope="module")
def src_gd_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/ul_gd.py", "src_gd_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def me_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/ul_me.py", "me_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def ssiuu_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/tr_ssiuu.py", "ssiuu_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def sam_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/tr_sam.py", "sam_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def circuit_breaker_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/ul_cb.py", "circuit_breaker_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def rmu_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/ul_rmu.py", "rmu_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def weight_dist_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/tr_wd.py", "wd_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


@pytest.fixture(scope="module")
def ao_training_module():
    pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        return load_script("src/trainers/tr_ao.py", "ao_training")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))


def test_orth_cb_document_strategies_tag_identical_schemas(
    wmdp_di_module,
    wikitext_di_module,
) -> None:
    calls = []

    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    cfg = SimpleNamespace(sequence_len=2048)
    strategy = wmdp_di_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    forget = strategy.tokenize_row(
        {"title": "title", "abstract": "abstract", "text": "bio"}
    )
    retain = wikitext_di_module.load(
        tokenizer,
        cfg,
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    ).tokenize_row({"page": "wiki"})

    assert (
        set(forget)
        == set(retain)
        == {
            "input_ids",
            "attention_mask",
            "labels",
            "is_forget",
        }
    )
    assert forget["is_forget"] is True
    assert retain["is_forget"] is False
    assert calls[0][0] == "title\n\nabstract\n\nbio"
    assert calls[0][1] == {
        "max_length": 2048,
        "truncation": True,
        "add_special_tokens": True,
    }


@pytest.mark.parametrize("missing", ["title", "abstract", "text"])
def test_orth_cb_wmdp_requires_document_fields(wmdp_di_module, missing) -> None:
    strategy = wmdp_di_module.load(
        lambda text, **kwargs: {"input_ids": [1], "attention_mask": [1]},
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="cais/wmdp-bio-forget-corpus"),
    )
    row = {"title": "T", "abstract": "A", "text": "B"}
    del row[missing]
    with pytest.raises(ValueError, match="title, abstract, and text"):
        strategy.tokenize_row(row)


def test_orth_cb_tokenization_truncates_and_masks(wikitext_di_module) -> None:
    def tokenizer(text, *, max_length, truncation, add_special_tokens):
        assert truncation and add_special_tokens
        ids = list(range(len(text.split())))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    row = strategy.tokenize_row({"page": "word " * 2100})
    assert len(row["input_ids"]) == 2048
    assert row["attention_mask"] == [1] * 2048
    assert row["labels"] == row["input_ids"]


def test_orth_cb_wikitext_shuffles_before_selecting(wikitext_di_module) -> None:
    class FakeDataset(list):
        column_names = ["page"]

        def shuffle(self, seed):
            assert seed == 42
            return FakeDataset(reversed(self))

        def select(self, indices):
            assert len(indices) == 1024
            return FakeDataset(self[index] for index in indices)

        def map(self, function, remove_columns):
            assert remove_columns == self.column_names
            return FakeDataset(function(row) for row in self)

    tokenizer = lambda text, **kwargs: {
        "input_ids": [int(text)],
        "attention_mask": [1],
    }
    strategy = wikitext_di_module.load(
        tokenizer,
        SimpleNamespace(sequence_len=2048),
        SimpleNamespace(path="EleutherAI/wikitext_document_level"),
    )
    wrapped = strategy.wrap_dataset(
        FakeDataset({"page": str(index)} for index in range(1100))
    )

    assert len(wrapped) == 1024
    assert wrapped[0]["input_ids"] == [1099]
    assert wrapped[0]["is_forget"] is False


def test_authors_documents_train_on_padded_tokens(authors_documents_module) -> None:
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert text == "document"
            assert kwargs == {
                "max_length": 4,
                "padding": "max_length",
                "truncation": True,
            }
            return {
                "input_ids": [7, 8, 2, 2],
                "attention_mask": [1, 1, 0, 0],
            }

    strategy = authors_documents_module.AuthorsDocumentStrategy(
        Tokenizer(), 4, "cais/wmdp-bio-forget-corpus"
    )
    row = strategy.tokenize_row({"text": "document"})

    assert row["labels"] == [7, 8, 2, 2]
    assert row["attention_mask"] == [1, 1, 0, 0]
    assert row["is_forget"] is True


def test_axolotl_standard_collator_preserves_source_tags() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("axolotl")
    from axolotl.utils.collators import DataCollatorForSeq2Seq
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    batch = DataCollatorForSeq2Seq(tokenizer)(
        [
            {
                "input_ids": [2, 3, 4],
                "attention_mask": [1, 1, 1],
                "labels": [2, 3, 4],
                "is_forget": False,
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [5, 6],
                "is_forget": True,
            },
        ]
    )
    assert isinstance(batch["is_forget"], torch.Tensor)
    assert batch["is_forget"].tolist() == [False, True]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_plain_finetuning_drops_source_tag(tmp_path: Path, wmdp_di_module) -> None:
    torch = pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")

    strategy = wmdp_di_module.WmdpDocumentStrategy(
        lambda text, **kwargs: {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
        },
        2048,
    )
    row = strategy.tokenize_row({"title": "T", "abstract": "A", "text": "B"})
    assert row["is_forget"] is True

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, input_ids, attention_mask=None, labels=None):
            return {"loss": self.weight * 0}

    trainer = transformers.Trainer(
        model=Model(),
        args=transformers.TrainingArguments(
            output_dir=str(tmp_path),
            remove_unused_columns=True,
            per_device_train_batch_size=1,
            dataloader_pin_memory=False,
        ),
        train_dataset=datasets.Dataset.from_list([row]),
    )
    batch = next(iter(trainer.get_train_dataloader()))
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


class TaggedDataset:
    def __init__(self, sources):
        self.sources = sources

    def __getitem__(self, key):
        assert key == "is_forget"
        return self.sources


def test_mixed_source_sampler_uses_each_row_once_per_epoch(sampling_module) -> None:
    dataset = TaggedDataset([0, 1, 1, 0] * 4)
    sampler = sampling_module.MixedSourceSampler(dataset, 8, 42)
    first, second = list(sampler), list(sampler)

    for epoch in (first, second):
        assert len(epoch) == len(sampler) == len(dataset.sources)
        assert sorted(epoch) == list(range(len(dataset.sources)))
        for start in range(0, len(epoch), 8):
            batch = [dataset.sources[index] for index in epoch[start : start + 8]]
            assert batch.count(0) == 4
    assert first != second


def test_mixed_source_sampler_truncates_to_shorter_source(sampling_module) -> None:
    sampler = sampling_module.MixedSourceSampler
    with pytest.raises(ValueError, match="even batch size"):
        sampler(TaggedDataset([0, 1]), 3, 42)
    with pytest.raises(ValueError, match="forget and retain"):
        sampler(TaggedDataset([0, 0]), 2, 42)
    dataset = TaggedDataset([0, 0, 1])
    indices = list(sampler(dataset, 2, 42))
    assert len(indices) == 2
    assert {dataset.sources[index] for index in indices} == {0, 1}


def test_authors_sampler_draws_retain_with_replacement(sampling_module) -> None:
    sampler = sampling_module.AuthorsMixedSourceSampler(
        TaggedDataset([1, 1, 1, 1, 0, 0]), batch_size=2, seed=3
    )
    first, second = list(sampler), list(sampler)

    for epoch in (first, second):
        forget = [index for index in epoch if index < 4]
        retain = [index for index in epoch if index >= 4]
        assert sorted(forget) == [0, 1, 2, 3]
        assert len(retain) == 4
        assert set(retain) <= {4, 5}
        for start in range(0, len(epoch), 2):
            assert sum(index < 4 for index in epoch[start : start + 2]) == 1
    assert len(first) == len(sampler) == 8
    assert first != second


def test_weight_distance_rms_matches_authors_normalization(
    weight_dist_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    first = torch.tensor([3.0, 4.0], requires_grad=True)
    second = torch.tensor([12.0], requires_grad=True)
    distance = weight_dist_training_module.global_rms_distance(
        (first, second), (torch.zeros_like(first), torch.zeros_like(second))
    )
    expected = (169 / 3 + 1e-8) ** 0.5

    assert distance.item() == pytest.approx(expected)
    distance.backward()
    torch.testing.assert_close(
        first.grad, torch.tensor([3 / (3 * expected), 4 / (3 * expected)])
    )
    torch.testing.assert_close(second.grad, torch.tensor([12 / (3 * expected)]))


def test_circuit_breaker_schedule_matches_original_objective(
    circuit_breaker_training_module,
) -> None:
    trainer = object.__new__(circuit_breaker_training_module.CircuitBreakerTrainer)
    trainer.retain_coefficient = 10.0
    trainer.reroute_coefficient = 23.0
    trainer.orthogonalization_coefficient = 5.0
    trainer.total_microsteps = 125

    trainer.microstep = 0
    assert trainer.coefficients() == pytest.approx((1.0, 23.0, 0.0))
    trainer.microstep = 124
    assert trainer.coefficients() == pytest.approx((10.0, 17.25, 5.0))


def test_circuit_breaker_reference_parameters_are_swapped_and_restored(
    circuit_breaker_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    trainer = object.__new__(circuit_breaker_training_module.CircuitBreakerTrainer)
    trainer.model = model
    trainer._theta_ref = {"weight": torch.full_like(model.weight, 7.0)}

    with trainer.reference_parameters():
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7.0))
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))


def test_circuit_breaker_losses_preserve_layer_and_padding_contracts(
    circuit_breaker_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.0, 100.0], [3.0, 5.0]])
    mask = torch.tensor([[1, 0], [1, 1]])
    assert circuit_breaker_training_module.masked_mean(values, mask).item() == (
        pytest.approx(3.0)
    )

    activations = torch.tensor(
        [
            [[1.0, 0.0], [10.0, 10.0]],
            [[1.0, 0.0], [20.0, 20.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 0], [1, 0]])
    loss = circuit_breaker_training_module.orthogonalization_loss(
        activations, attention_mask
    )
    assert loss.item() == pytest.approx(1.0)


def test_circuit_breaker_sam_accepts_scheduled_zero_gradient(
    sam_training_module,
) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    trainer = object.__new__(sam_training_module.CircuitBreakerSAMTrainer)
    trainer.optimizer = SimpleNamespace()
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    trainer.current_gradient_accumulation_steps = 1
    trainer.compute_loss_context_manager = nullcontext
    trainer.log = lambda metrics: None
    trainer.compute_cb_loss = lambda model, inputs, **kwargs: (
        model.weight.sum() * 0,
        {},
    )

    loss = trainer._sam_training_step(model, {})

    assert loss.item() == 0
    torch.testing.assert_close(model.weight.grad, torch.zeros_like(model.weight))


def test_rmu_activation_loss_ignores_padding(rmu_training_module) -> None:
    torch = pytest.importorskip("torch")
    activation = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros(1, 1, 2)
    attention_mask = torch.tensor([[1, 0]])

    loss = rmu_training_module.RMUTrainer.activation_loss(
        activation, target, attention_mask
    )

    assert loss.item() == pytest.approx(5.0)


def test_rmu_control_vector_has_fixed_norm_and_seed(rmu_training_module) -> None:
    torch = pytest.importorskip("torch")
    activation = torch.zeros(2, 3, 8)
    first = object.__new__(rmu_training_module.RMUTrainer)
    first.steering_coefficient = 6.5
    first.control_seed = 42
    first._control_vector = None
    second = object.__new__(rmu_training_module.RMUTrainer)
    second.steering_coefficient = 6.5
    second.control_seed = 42
    second._control_vector = None

    control = first.control_vector(activation)

    assert control.shape == (1, 1, 8)
    assert control.float().norm().item() == pytest.approx(6.5)
    torch.testing.assert_close(control, second.control_vector(activation))


@pytest.mark.parametrize(
    "trainer_name, parent_name",
    [
        ("WDGDTrainer", "GradDiffTrainer"),
        ("WDMaxEntTrainer", "MaxEntTrainer"),
        ("WDNPOTrainer", "NPOTrainer"),
        ("WDRMUTrainer", "RMUTrainer"),
    ],
)
def test_wd_adds_distance_to_unlearning_loss(
    weight_dist_training_module, monkeypatch, trainer_name, parent_name
) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
    outputs = SimpleNamespace(loss=torch.tensor(0.0))
    monkeypatch.setattr(
        getattr(weight_dist_training_module, parent_name),
        "compute_loss",
        lambda *args, **kwargs: (torch.tensor(10.0), outputs),
    )
    trainer = object.__new__(getattr(weight_dist_training_module, trainer_name))
    trainer.coefficient_weight_dist = 2.0
    trainer.model = model
    trainer.anchor_model_name_or_path = "anchor-model"
    trainer._theta_anchor = {"weight": torch.zeros_like(model.weight)}
    trainer._weight_distance_logged_step = None
    trainer.state = SimpleNamespace(global_step=0)
    logged = []
    trainer.log = logged.append

    loss, returned_outputs = trainer.compute_loss(model, {}, return_outputs=True)
    expected_distance = (25 / 2 + 1e-8) ** 0.5

    assert loss.item() == pytest.approx(10 - 2 * expected_distance)
    assert returned_outputs is outputs
    assert logged == [{"loss_weight_dist": pytest.approx(-2 * expected_distance)}]
    loss.backward()
    assert model.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "parent_name", ["GradDiffTrainer", "MaxEntTrainer", "NPOTrainer", "RMUTrainer"]
)
def test_wd_forwards_unlearning_arguments(
    weight_dist_training_module, monkeypatch, parent_name
) -> None:
    received = {}
    parent = getattr(weight_dist_training_module, parent_name)
    monkeypatch.setattr(parent, "__init__", lambda _, **kwargs: received.update(kwargs))
    trainer_cls = next(
        cls
        for name, cls in vars(weight_dist_training_module).items()
        if name.startswith("WD") and parent in cls.__mro__
    )

    trainer = trainer_cls(
        anchor_model_name_or_path="anchor-model",
        coefficient_weight_dist=100.0,
        reference_model_name_or_path="reference-model",
        retain_coefficient=4.0,
    )

    assert received == {
        "reference_model_name_or_path": "reference-model",
        "retain_coefficient": 4.0,
    }
    assert trainer.anchor_model_name_or_path == "anchor-model"
    assert trainer.coefficient_weight_dist == 100.0


def test_reference_parameter_loader_strips_framework_wrapper_names(
    weight_dist_training_module, monkeypatch, tmp_path
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint_name = "model.layers.0.mlp.down_proj.weight"
    runtime_name = (
        "model.layers.0._checkpoint_wrapped_module.mlp."
        "_checkpoint_wrapped_module.down_proj.weight"
    )
    expected = torch.tensor([[1.0, 2.0]])
    safetensors.save_file({checkpoint_name: expected}, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {checkpoint_name: "model.safetensors"}})
    )

    reference_parameters = weight_dist_training_module.load_reference_parameters(
        {runtime_name: torch.nn.Parameter(torch.zeros_like(expected))}, tmp_path
    )

    assert reference_parameters.keys() == {runtime_name}
    torch.testing.assert_close(reference_parameters[runtime_name], expected)


def test_wd_requires_anchor_model(weight_dist_training_module) -> None:
    with pytest.raises(TypeError, match="anchor_model_name_or_path"):
        weight_dist_training_module.WDGDTrainer("model")


def test_wd_loads_anchor_model_once(weight_dist_training_module, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 1, bias=False)
    expected = {"weight": torch.full_like(model.weight, 7)}
    calls = []

    def load(parameters, model_name_or_path):
        calls.append((parameters, model_name_or_path))
        return expected

    monkeypatch.setattr(weight_dist_training_module, "load_reference_parameters", load)
    trainer = object.__new__(weight_dist_training_module.WDMaxEntTrainer)
    trainer.model = model
    trainer.anchor_model_name_or_path = "anchor-model"
    trainer._theta_anchor = None

    assert trainer.theta_anchor == expected
    assert trainer.theta_anchor == expected
    assert calls == [({"weight": model.weight}, "anchor-model")]


def test_src_grad_diff_logs_component_losses_and_gradient_norms(
    src_gd_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_weight = torch.nn.Parameter(torch.tensor(2.0))
            self.retain_weight = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            ids = input_ids.flatten().tolist()
            weight = self.forget_weight if ids == [2, 4] else self.retain_weight
            return SimpleNamespace(loss=input_ids.float().mean() * weight)

    model = Model()
    trainer = object.__new__(src_gd_training_module.GradDiffTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 1.0
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer.compute_loss(model, inputs)
    loss.backward()

    assert loss.item() == pytest.approx(12.0)
    assert logged == [
        {
            "loss_forget": pytest.approx(12.0),
            "loss_retain": pytest.approx(24.0),
            "grad_norm_forget": pytest.approx(6.0),
            "grad_norm_retain": pytest.approx(8.0),
        }
    ]
    assert model.forget_weight.grad.item() == pytest.approx(-6.0)
    assert model.retain_weight.grad.item() == pytest.approx(8.0)


def _maxent_sam_fixture(torch, sam_training_module, rho):
    from contextlib import nullcontext

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_scale = torch.nn.Parameter(torch.tensor(0.7))
            self.retain_scale = torch.nn.Parameter(torch.tensor(0.4))

        def forward(self, input_ids, attention_mask, labels=None):
            is_forget_call = torch.all(input_ids[:, 0] == 0)
            scale = self.forget_scale if is_forget_call else self.retain_scale
            values = input_ids.float() * scale
            logits = torch.stack((values, torch.zeros_like(values), -values), dim=-1)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, 3),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

    class Accelerator:
        def backward(self, loss):
            loss.backward()

    trainer = object.__new__(sam_training_module.MaxEntSAMTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 0.5
    trainer.log_gradient_norms = False
    trainer.rho = rho
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 2
    trainer._layer_offload_ctx = nullcontext()
    trainer.activation_offload_context = nullcontext()
    trainer._prepare_inputs = lambda batch: batch
    trainer.compute_loss_context_manager = nullcontext
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[1, 0, 2], [0, 1, 2], [1, 2, 0], [0, 2, 1]]),
        "attention_mask": torch.ones(4, 3),
        "labels": torch.tensor(
            [[-100, 0, 2], [-100, 1, 2], [-100, 2, 0], [-100, 2, 1]]
        ),
        "is_forget": torch.tensor([False, True, False, True]),
    }
    return Model, trainer, inputs, logged


def test_maxent_sam_at_zero_radius_matches_maxent_gradient(
    sam_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    Model, trainer, inputs, _ = _maxent_sam_fixture(torch, sam_training_module, 0.0)

    maxent_model = Model()
    (trainer.compute_loss(maxent_model, inputs) / 2).backward()
    sam_model = Model()
    trainer.training_step(sam_model, inputs)

    for name in ("forget_scale", "retain_scale"):
        torch.testing.assert_close(
            getattr(sam_model, name).grad, getattr(maxent_model, name).grad
        )


def test_maxent_sam_perturbs_forget_only_and_restores_weights(
    sam_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    rho = 0.05
    Model, trainer, inputs, logged = _maxent_sam_fixture(
        torch, sam_training_module, rho
    )
    model = Model()
    model.forget_scale.grad = torch.tensor(4.0)
    model.retain_scale.grad = torch.tensor(-1.0)

    reference = Model()
    forget_grad = torch.autograd.grad(
        trainer.compute_forget_loss(reference, inputs), reference.forget_scale
    )[0]
    step = rho * torch.sign(forget_grad)
    with torch.no_grad():
        reference.forget_scale.add_(step)
    perturbed_forget_grad = torch.autograd.grad(
        trainer.compute_forget_loss(reference, inputs), reference.forget_scale
    )[0]
    with torch.no_grad():
        reference.forget_scale.sub_(step)
    retain_grad = torch.autograd.grad(
        trainer.compute_retain_loss(reference, inputs), reference.retain_scale
    )[0]

    loss = trainer.training_step(model, inputs)

    assert loss.isfinite()
    assert model.forget_scale.item() == pytest.approx(0.7)
    assert model.retain_scale.item() == pytest.approx(0.4)
    assert model.forget_scale.grad.item() == pytest.approx(
        4.0 + perturbed_forget_grad.item() / 2
    )
    assert model.retain_scale.grad.item() == pytest.approx(
        -1.0 + retain_grad.item() / 2
    )
    assert perturbed_forget_grad.item() != pytest.approx(forget_grad.item())
    assert set(logged[-1]) == {
        "entropy_forget",
        "loss_retain",
        "grad_norm_forget",
        "grad_norm_retain",
        "grad_norm_forget_at_perturbation",
    }


def test_maxent_sam_rejects_nonpositive_radius(sam_training_module) -> None:
    with pytest.raises(ValueError, match="rho must be positive"):
        sam_training_module.MaxEntSAMTrainer(rho=0.0)


def test_maxent_loss_masks_padding_and_preserves_trainer_contract(
    me_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_scale = torch.nn.Parameter(torch.tensor(0.7))
            self.retain_scale = torch.nn.Parameter(torch.tensor(0.4))
            self.forget_outputs = None
            self.calls = []

        def forward(self, input_ids, attention_mask, labels=None):
            is_forget_call = torch.all(input_ids[:, 0] == 0)
            self.calls.append((is_forget_call.item(), labels is None))
            scale = self.forget_scale if is_forget_call else self.retain_scale
            values = input_ids.float() * scale
            logits = torch.stack((values, torch.zeros_like(values), -values), dim=-1)
            loss = None
            if labels is not None:
                shifted_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
                shifted_labels = labels[:, 1:].reshape(-1)
                loss = torch.nn.functional.cross_entropy(
                    shifted_logits, shifted_labels, ignore_index=-100
                )
            outputs = SimpleNamespace(loss=loss, logits=logits)
            if is_forget_call:
                self.forget_outputs = outputs
            return outputs

    model = Model()
    trainer = object.__new__(me_training_module.MaxEntTrainer)
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 0.5
    trainer.log_gradient_norms = False
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[1, 0, 2], [0, 1, 2], [1, 2, 0], [0, 2, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]]),
        "labels": torch.tensor(
            [[-100, 0, 2], [-100, -100, 1], [-100, 2, 0], [-100, 2, -100]]
        ),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    active_forget_ids = torch.tensor([1.0, 0.0])
    forget_values = active_forget_ids * model.forget_scale
    forget_logits = torch.stack(
        (forget_values, torch.zeros_like(forget_values), -forget_values), dim=-1
    )
    forget_log_probs = torch.log_softmax(forget_logits, dim=-1)
    expected_forget = 2.0 * torch.mean(
        torch.sum(torch.exp(forget_log_probs) * forget_log_probs, dim=-1)
    )

    retain_inputs = inputs["input_ids"][[0, 2]]
    retain_values = retain_inputs.float() * model.retain_scale
    retain_logits = torch.stack(
        (retain_values, torch.zeros_like(retain_values), -retain_values), dim=-1
    )
    expected_retain = 0.5 * torch.nn.functional.cross_entropy(
        retain_logits[:, :-1].reshape(-1, 3),
        inputs["labels"][[0, 2], 1:].reshape(-1),
    )
    expected_forget_grad = torch.autograd.grad(
        expected_forget, model.forget_scale, retain_graph=True
    )[0]
    expected_retain_grad = torch.autograd.grad(
        expected_retain, model.retain_scale, retain_graph=True
    )[0]

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    loss.backward()

    assert loss.item() == pytest.approx(
        (expected_forget + expected_retain).detach().item()
    )
    assert outputs is model.forget_outputs
    assert model.calls == [(True, True), (False, False)]
    assert model.forget_scale.grad.item() == pytest.approx(expected_forget_grad.item())
    assert model.retain_scale.grad.item() == pytest.approx(expected_retain_grad.item())
    assert logged == [
        {
            "entropy_forget": pytest.approx(
                (-expected_forget / trainer.forget_coefficient).detach().item()
            ),
            "entropy_ceiling": pytest.approx(torch.log(torch.tensor(3.0)).item()),
            "loss_retain": pytest.approx(expected_retain.detach().item()),
        }
    ]

    changed_labels = dict(inputs)
    changed_labels["labels"] = inputs["labels"].clone()
    changed_labels["labels"][inputs["is_forget"]] = torch.tensor(
        [[-100, -100, 2], [-100, 1, -100]]
    )
    assert trainer.compute_forget_loss(model, changed_labels).item() == pytest.approx(
        expected_forget.detach().item()
    )


def test_ssiuu_sequence_probability_masks_prompt_and_padding(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(2, 4, 3)
    logits[0, 1, 1] = torch.log(torch.tensor(2.0))
    logits[0, 2, 2] = torch.log(torch.tensor(3.0))
    logits[1, 1, 2] = torch.log(torch.tensor(4.0))
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 2, -100]])

    probability = ssiuu_training_module.SSIUUTrainer.sequence_probability(
        logits, labels
    )

    assert probability.item() == pytest.approx((2 / 4) * (3 / 5) + 4 / 6)


def test_ssiuu_geometric_mean_probability_does_not_underflow(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 513, 2)
    labels = torch.ones(1, 513, dtype=torch.long)

    raw = ssiuu_training_module.SSIUUTrainer.sequence_probability(logits, labels)
    normalized = ssiuu_training_module.SSIUUTrainer.sequence_probability(
        logits, labels, "geometric_mean"
    )

    assert raw.item() == 0.0
    assert normalized.item() == pytest.approx(0.5)


def test_ssiuu_attribution_is_parameter_times_probability_gradient(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, attention_mask, labels, use_cache):
            del attention_mask, labels
            assert not use_cache
            logits = torch.zeros(input_ids.shape[0], 3, 2)
            logits[:, 1, 1] = self.weight
            return SimpleNamespace(logits=logits)

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    inputs = {
        "input_ids": torch.tensor([[0, 0, 1], [0, 0, 0]]),
        "attention_mask": torch.ones(2, 3),
        "labels": torch.tensor([[-100, -100, 1], [-100, -100, 0]]),
        "is_forget": torch.tensor([True, False]),
    }

    probability, (attribution,) = trainer.compute_attributions(model, inputs)
    expected_probability = torch.sigmoid(model.weight.detach())
    expected_attribution = (
        model.weight.detach() * expected_probability * (1 - expected_probability)
    )

    assert probability.item() == pytest.approx(expected_probability.item())
    assert attribution.item() == pytest.approx(expected_attribution.item())
    assert attribution.requires_grad
    torch.autograd.grad(attribution, model.weight)


def test_ssiuu_penalizes_only_currently_negative_attribution_drift(
    ssiuu_training_module,
) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    trainer.ssiuu_coefficient = 0.5
    trainer._previous_attributions = (torch.tensor([-1.0, 10.0]),)

    current = model.weight * torch.tensor([-2.0, 2.0])
    trainer.compute_attributions = lambda model, inputs: (current.sum(), (current,))
    loss = trainer.compute_ssiuu_loss(model, {})

    assert loss.item() == pytest.approx(0.5 * (-4.0 - -1.0) ** 2)
    loss.backward()
    assert model.weight.grad.tolist() == pytest.approx([6.0, 0.0])
    assert trainer._previous_attributions[0].tolist() == pytest.approx([-4.0, 6.0])
    assert not trainer._previous_attributions[0].requires_grad


def test_ssiuu_first_iteration_has_zero_regularizer(ssiuu_training_module) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

    model = Model()
    trainer = object.__new__(ssiuu_training_module.SSIUUTrainer)
    trainer.ssiuu_coefficient = 1.0
    trainer._previous_attributions = None
    attribution = model.weight.square()
    trainer.compute_attributions = lambda model, inputs: (attribution, (attribution,))

    loss = trainer.compute_ssiuu_loss(model, {})

    assert loss.item() == 0.0
    assert trainer._previous_attributions[0].item() == 4.0


def test_src_gd_sam_logs_component_losses_and_gradient_norms(
    sam_training_module,
) -> None:
    from contextlib import nullcontext

    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, input_ids, attention_mask, labels):
            del attention_mask, labels
            return SimpleNamespace(loss=input_ids.float().mean() * self.weight.square())

    class Accelerator:
        @staticmethod
        def backward(loss):
            loss.backward()

    model = Model()
    trainer = object.__new__(sam_training_module.GDSAMTrainer)
    trainer.optimizer = None
    trainer.accelerator = Accelerator()
    trainer.current_gradient_accumulation_steps = 1
    trainer.compute_loss_context_manager = nullcontext
    trainer.rho = 0.01
    trainer.forget_coefficient = 2.0
    trainer.retain_coefficient = 1.0
    trainer.optml_parity = False
    logged = []
    trainer.log = logged.append
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "labels": torch.tensor([[7], [2], [9], [4]]),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss = trainer._sam_training_step(model, inputs)

    assert loss.item() == pytest.approx(-6 * 1.99**2 + 8 * 2**2)
    assert logged == [
        {
            "loss_forget": pytest.approx(6 * 1.99**2),
            "loss_retain": pytest.approx(8 * 2**2),
            "grad_norm_forget": pytest.approx(12 * 1.99),
            "grad_norm_retain": pytest.approx(16 * 2),
            "grad_norm_forget_at_perturbation": pytest.approx(6 * 2),
        }
    ]
    assert model.weight.grad.item() == pytest.approx(-12 * 1.99 + 16 * 2)


def test_optml_sam_norm_preserves_original_precision(
    sam_training_module,
) -> None:
    torch = pytest.importorskip("torch")
    gradients = {
        "first": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "second": torch.tensor([0.0, 12.0], dtype=torch.bfloat16),
    }

    perturbation = sam_training_module.optml_sam_perturbation(gradients, 0.01)
    norm = torch.stack([gradient.norm(2) for gradient in perturbation.values()]).norm(2)

    assert all(value.dtype == torch.bfloat16 for value in perturbation.values())
    assert norm.item() == pytest.approx(0.01, abs=1e-4)


def test_optml_sampler_matches_paired_random_sampling(sampling_module) -> None:
    torch = pytest.importorskip("torch")

    class TaggedDataset:
        sources = [1, 1, 1, 1, 0, 0]

        def __getitem__(self, key):
            assert key == "is_forget"
            return self.sources

    random.seed(42)
    torch.manual_seed(42)
    expected_forget = list(torch.utils.data.RandomSampler(range(4)))
    expected_retain = [random.choice([4, 5]) for _ in range(4)]

    random.seed(42)
    torch.manual_seed(42)
    actual = list(sampling_module.OptmlMixedSourceSampler(TaggedDataset(), 4))

    assert actual == [
        *expected_forget[:2],
        *expected_retain[:2],
        *expected_forget[2:],
        *expected_retain[2:],
    ]
    assert len(actual) == 8


class AOToyModel:
    """Build a toy causal LM whose forget hidden states are two fixed rows."""

    @staticmethod
    def build(torch):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.forget_rows = torch.nn.Parameter(
                    torch.tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
                )
                self.hidden_state_calls = 0

            def forward(
                self,
                input_ids,
                attention_mask=None,
                labels=None,
                output_hidden_states=False,
                use_cache=None,
            ):
                loss = input_ids.float().mean() * self.weight
                logits = self.weight * torch.ones(*input_ids.shape, 3)
                if not output_hidden_states:
                    return SimpleNamespace(loss=loss, logits=logits)
                assert use_cache is False
                self.hidden_state_calls += 1
                hidden = self.forget_rows.unsqueeze(1)
                return SimpleNamespace(
                    loss=loss, logits=logits, hidden_states=(None, hidden, hidden)
                )

        return Model()


AO_INPUTS = {
    "input_ids": [[7, 0], [2, 1], [9, 2], [4, 0]],
    "is_forget": [False, True, False, True],
}


def ao_inputs(torch):
    input_ids = torch.tensor(AO_INPUTS["input_ids"])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
        "is_forget": torch.tensor(AO_INPUTS["is_forget"]),
    }


def ao_trainer(cls, **attributes):
    trainer = object.__new__(cls)
    trainer.target_layers = (0, 1)
    trainer.activation_orthogonality_coefficient = 4.0
    trainer.state = SimpleNamespace(global_step=0)
    trainer._ao_logged_step = None
    trainer.logged = []
    trainer.log = trainer.logged.append
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


def test_ao_gd_adds_orthogonality_to_gradient_difference(ao_training_module) -> None:
    torch = pytest.importorskip("torch")
    model = AOToyModel.build(torch)
    trainer = ao_trainer(
        ao_training_module.AOGDTrainer, retain_coefficient=2.0, forget_coefficient=3.0
    )

    loss, outputs = trainer.compute_loss(model, ao_inputs(torch), return_outputs=True)

    # Retain rows average 4.5 and forget rows 1.75, both scaled by weight 2.
    assert loss.item() == pytest.approx(2 * 9.0 - 3 * 3.5 + 4 * 0.96)
    assert outputs.hidden_states is not None
    assert model.hidden_state_calls == 1
    assert trainer.logged[-1]["loss_activation_orthogonality"] == pytest.approx(
        4 * 0.96
    )
    loss.backward()
    assert model.forget_rows.grad.abs().sum() > 0


def test_ao_npo_adds_orthogonality_to_npo(ao_training_module) -> None:
    torch = pytest.importorskip("torch")
    model = AOToyModel.build(torch)
    beta = 0.5
    trainer = ao_trainer(
        ao_training_module.AONPOTrainer,
        model=model,
        beta=beta,
        retain_coefficient=2.0,
        _npo_logged_step=None,
    )
    trainer._theta_ref = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    loss, outputs = trainer.compute_loss(model, ao_inputs(torch), return_outputs=True)

    # At the reference weights NPO's forget loss is -(2 / beta) * log(sigmoid(0)).
    loss_forget = 2 * math.log(2) / beta
    assert loss.item() == pytest.approx(loss_forget + 2 * 9.0 + 4 * 0.96)
    assert outputs.hidden_states is not None
    assert model.hidden_state_calls == 1
    loss.backward()
    assert model.forget_rows.grad.abs().sum() > 0


def test_ao_trainers_forward_unlearning_arguments(
    ao_training_module, monkeypatch
) -> None:
    for parent, cls in (
        (ao_training_module.GradDiffTrainer, ao_training_module.AOGDTrainer),
        (ao_training_module.NPOTrainer, ao_training_module.AONPOTrainer),
    ):
        received = {}

        def init(trainer, *args, **kwargs):
            received.update(kwargs)
            trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3))

        monkeypatch.setattr(parent, "__init__", init)
        trainer = cls(
            retain_coefficient=4.0,
            sampling="authors",
            activation_orthogonality_coefficient=10.0,
        )

        assert received == {"retain_coefficient": 4.0, "sampling": "authors"}
        assert trainer.activation_orthogonality_coefficient == 10.0
        assert trainer.target_layers == (0, 1, 2)


def test_ao_validates_arguments(ao_training_module, monkeypatch) -> None:
    def init(trainer, *args, **kwargs):
        trainer.model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2))

    monkeypatch.setattr(ao_training_module.GradDiffTrainer, "__init__", init)
    cls = ao_training_module.AOGDTrainer
    for target_layers in ((0, 0), (1, 2), ()):
        with pytest.raises(ValueError, match="target_layers"):
            cls(target_layers=target_layers)
    with pytest.raises(ValueError, match="activation_orthogonality_coefficient"):
        cls(activation_orthogonality_coefficient=-1.0)
    assert cls(target_layers=(1,)).target_layers == (1,)


def test_ao_ignores_padding_tokens_in_forget_pool(ao_training_module) -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0], [9.0, 9.0]],
            [[2.0, 0.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    trainer = ao_trainer(
        ao_training_module.AOGDTrainer,
        target_layers=(1,),
        activation_orthogonality_coefficient=1.0,
    )
    loss = trainer.compute_activation_orthogonality_loss(
        (None, hidden, hidden), torch.tensor([[1, 1, 0], [1, 0, 0]])
    )
    assert loss.item() == pytest.approx(1.0 / 1.25**0.5)


def test_ao_requires_two_forget_rows(ao_training_module) -> None:
    torch = pytest.importorskip("torch")
    for cls in (ao_training_module.AOGDTrainer, ao_training_module.AONPOTrainer):
        trainer = object.__new__(cls)
        with pytest.raises(ValueError, match="at least two forget"):
            trainer.compute_loss(None, {"is_forget": torch.tensor([False, True])})


def test_ao_rmu_loss_reuses_forget_activations(ao_training_module) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forget_rows = torch.nn.Parameter(
                torch.tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0]])
            )
            self.calls = 0

        def forward(self, **kwargs):
            self.calls += 1
            hidden = self.forget_rows.unsqueeze(1)
            return SimpleNamespace(hidden_states=(None, hidden, hidden))

    model = Model()
    trainer = object.__new__(ao_training_module.AORMUTrainer)
    trainer.target_layer = 0
    trainer.target_layers = (0, 1)
    trainer.steering_coefficient = 1.0
    trainer.control_seed = 42
    trainer._control_vector = None
    trainer.activation_orthogonality_coefficient = 4.0
    trainer.compute_retain_loss = lambda model, inputs: torch.tensor(2.0)
    trainer.state = SimpleNamespace(global_step=0)
    trainer._rmu_logged_step = None
    trainer.log = lambda metrics: None
    inputs = {
        "input_ids": torch.tensor([[7], [2], [9], [4]]),
        "attention_mask": torch.ones(4, 1),
        "is_forget": torch.tensor([False, True, False, True]),
    }

    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    assert loss.item() == pytest.approx(outputs["loss_forget"].item() + 2.0 + 4 * 0.96)
    assert outputs["loss_retain"].item() == pytest.approx(2.0)
    assert outputs["loss_activation_orthogonality"].item() == pytest.approx(4 * 0.96)
    assert model.calls == 1


def test_peft_disabled_adapter_recovers_initial_base_output() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    base = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=16, n_positions=8, n_embd=8, n_layer=1, n_head=2
        )
    ).eval()
    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        original = base(tokens).logits.clone()
    model = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["c_attn"])
    ).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.2)
        adapted = model(tokens).logits
        with model.disable_adapter():
            reference = model(tokens).logits
    assert not torch.allclose(adapted, original)
    torch.testing.assert_close(reference, original)


def test_relearning_formats_wmdp_document() -> None:
    pytest.importorskip("axolotl")
    strategy = load_script("src/data/wmdp_completion.py", "wmdp_bio_forget_strategy")
    document, prompt, response = (
        strategy.WmdpBioCompletionStrategy.parse_instruction_fields(
            None, {"title": "Title", "abstract": "Abstract", "text": "Text"}
        )
    )
    assert (document, prompt, response) == ("Title\n\nAbstract\n\nText", "", "")


def test_mmlu_no_bio_group_excludes_biology_overlap() -> None:
    config = yaml.safe_load((EXAMPLE / "lm_eval_tasks/mmlu_no_bio.yaml").read_text())
    excluded = {
        "mmlu_virology",
        "mmlu_medical_genetics",
        "mmlu_high_school_biology",
        "mmlu_college_biology",
    }
    assert config["group"] == "mmlu_no_bio"
    assert len(config["task"]) == 53
    assert not excluded.intersection(config["task"])
    assert config["aggregate_metric_list"] == [
        {"metric": "acc", "weight_by_size": True}
    ]


def test_beavertail_split_and_completion_labels(tmp_path: Path) -> None:
    pytest.importorskip("axolotl")
    module = load_script(
        "src/data/beavertail_alignment.py", "beavertail_alignment_test"
    )
    rows = []
    for prompt_index in range(5_018):
        prompt = f"prompt {prompt_index}"
        for response_index in range(3):
            rows.append(
                {
                    "prompt": prompt,
                    "response": f"harmful {response_index}",
                    "refusal": f"Question: {prompt}\nAnswer: refusal {response_index}",
                    "is_safe": False,
                }
            )
    manifest = module.split_source_indices(rows)
    assert len(manifest["evaluation"]) == 1_000
    assert len(manifest["alignment"]) == 5_000
    assert len(manifest["relearning"]) == 5_000
    prompt_sets = [
        {rows[index]["prompt"] for index in manifest[name]}
        for name in ("evaluation", "alignment", "relearning")
    ]
    assert prompt_sets[0].isdisjoint(prompt_sets[1])
    assert prompt_sets[0].isdisjoint(prompt_sets[2])
    assert prompt_sets[1].isdisjoint(prompt_sets[2])
    assert manifest == module.split_source_indices(rows)

    class Tokenizer:
        eos_token_id = 2

        def __call__(self, text, add_special_tokens):
            tokens = list(range(10, 10 + len(text.split())))
            return {"input_ids": ([1] if add_special_tokens else []) + tokens}

    row = dict(rows[0], source_index=0)
    strategy = module.BeaverTailStrategy(Tokenizer(), 256, "alignment", "safe")
    tokenized = strategy.tokenize_row(row)
    assert module.normalize_refusal(row) == "refusal 0"
    assert module.format_prompt("  test prompt  ") == (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\ntest prompt\n\n### Response:\n"
    )
    first_label = next(
        index for index, label in enumerate(tokenized["labels"]) if label != -100
    )
    assert tokenized["labels"][:first_label] == [-100] * first_label
    assert tokenized["labels"][-1] == 2
    assert tokenized["source_index"] == 0


def test_booster_sampler_and_negative_perturbation() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        module = load_script("src/trainers/tr_booster.py", "booster_training_test")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))
    sampler = module.BoosterMixedSourceSampler(
        {"is_harmful": [False] * 20 + [True] * 20}, 20, 42
    )
    order = list(sampler)
    assert len(order) == 40
    for start in range(0, len(order), 20):
        assert sum(index >= 20 for index in order[start : start + 20]) == 10

    gradients = {"x": torch.tensor([3.0, 4.0])}
    perturbation = module.negative_normalized_perturbation(gradients, 0.1)
    assert torch.allclose(perturbation["x"], torch.tensor([-0.06, -0.08]))
    assert perturbation["x"].norm().item() == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="invalid norm"):
        module.negative_normalized_perturbation({"x": torch.zeros(2)}, 0.1)


def test_beavertails_hh_filters_and_masks_prompt() -> None:
    pytest.importorskip("axolotl")
    datasets = pytest.importorskip("datasets")
    module = load_script("src/data/beavertails_hh.py", "beavertails_hh_test")

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __call__(
            self,
            text,
            add_special_tokens,
        ):
            words = [3 + index for index, _ in enumerate(text.split())]
            return {
                "input_ids": ([self.bos_token_id] if add_special_tokens else []) + words
            }

    cfg = SimpleNamespace(train_on_inputs=False, sequence_len=32)
    selection = SimpleNamespace(
        message_property_mappings={"max_rows": "2", "offset": "0"}
    )
    strategy = module.load_hh(Tokenizer(), cfg, selection)
    tokenized = strategy.tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    completion_start = tokenized["labels"].index(3)
    assert tokenized["labels"][:completion_start] == [-100] * completion_start
    assert tokenized["labels"][completion_start:] == [3, 4, Tokenizer.eos_token_id]
    train_on_inputs = module.load_hh(
        Tokenizer(),
        SimpleNamespace(train_on_inputs=True, sequence_len=32),
        selection,
    ).tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    assert -100 not in train_on_inputs["labels"]
    completion_only = module.load_hh(
        Tokenizer(),
        SimpleNamespace(train_on_inputs=True, sequence_len=32),
        SimpleNamespace(
            message_property_mappings={
                "max_rows": "2",
                "offset": "0",
                "train_on_inputs": "false",
            }
        ),
    ).tokenize_row(
        {
            "prompt": "unsafe question",
            "response": "harmful answer",
        }
    )
    assert completion_only["labels"][:completion_start] == [-100] * completion_start

    raw = datasets.Dataset.from_dict(
        {
            "prompt": ["safe", "unsafe 1", "unsafe 2", "unsafe 3"],
            "response": ["safe", "harmful 1", "harmful 2", "harmful 3"],
            "refusal": [
                "Question: safe\nAnswer: safe",
                "Question: unsafe 1\nAnswer: refusal 1",
                "Question: unsafe 2\nAnswer: refusal 2",
                "Question: unsafe 3\nAnswer: refusal 3",
            ],
            "is_safe": [True, False, False, False],
        }
    )
    wrapped = strategy.wrap_dataset(raw)
    assert len(wrapped) == 2

    hb = module.load_hb(Tokenizer(), cfg, selection)
    tokenized_hb = hb.tokenize_row(raw[1])
    assert tokenized_hb["labels"][:completion_start] == [-100] * completion_start
    assert tokenized_hb["labels"][completion_start:] == [
        3,
        4,
        Tokenizer.eos_token_id,
    ]

    offset = module.load_hh(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"max_rows": "2", "offset": "1"}),
    ).wrap_dataset(raw)
    assert len(offset) == 2

    all_remaining = module.load_hh(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"offset": "1"}),
    ).wrap_dataset(raw)
    assert len(all_remaining) == 2


def test_seam_defense_uses_per_dataset_input_masks() -> None:
    root = (
        EXAMPLE / "configs/stages-beavertails/0_alignment/"
        "llama2-7b-beavertail--align-fft-seam"
    )
    config = yaml.safe_load((root / "selected.yml").read_text())
    alpaca = config["datasets"][2]

    assert not (root / "selected_v2.yml").exists()
    assert [dataset["type"] for dataset in config["datasets"][:2]] == [
        "data.beavertails_hh",
        "data.beavertails_hh",
    ]
    assert [
        dataset["message_property_mappings"]["role"]
        for dataset in config["datasets"][:2]
    ] == ["hh", "hb"]
    assert [
        dataset["message_property_mappings"]["train_on_inputs"]
        for dataset in config["datasets"][:2]
    ] == ["false", "false"]
    assert alpaca["split"] == "train"
    assert alpaca["type"] == {
        "field_instruction": "instruction",
        "field_input": "input",
        "field_output": "output",
        "format": "Question: {instruction}\n{input}\nAnswer: ",
        "no_input_format": "Question: {instruction}\n\nAnswer: ",
    }
    assert config["train_on_inputs"] is True
    assert config["custom_trainer_args"] == {
        "coeff_hh": 1.0,
        "coeff_hb": 1.0,
        "coeff_sd": 0.01,
        "epsilon": 1e-3,
        "n_hh": 8_000,
        "n_hb": 8_000,
        "n_bb": 8_000,
    }


@pytest.mark.parametrize("divisor", [1, 2])
@pytest.mark.parametrize("legacy", [True, False], ids=["legacy", "seam"])
def test_seam_reports_unweighted_sd_loss(monkeypatch, divisor, legacy) -> None:
    torch = pytest.importorskip("torch")
    import sys

    monkeypatch.setitem(
        sys.modules,
        "axolotl.core.trainers.base",
        SimpleNamespace(AxolotlTrainer=object),
    )
    monkeypatch.setitem(
        sys.modules, "data.seam", SimpleNamespace(BENIGN=2, HARMFUL=0, REFUSAL=1)
    )
    monkeypatch.setitem(
        sys.modules, "trainers.rl_bf16", SimpleNamespace(BF16Trainer=object)
    )
    if legacy:
        module = load_script("src/trainers/tr_seam_legacy.py", "legacy_sd_loss_test")
        trainer = object.__new__(module.SEAMAlignmentTrainer)
    else:
        module = load_script("src/trainers/tr_seam.py", "seam_sd_loss_test")
        trainer = object.__new__(module.SEAMTrainer)
    trainer.current_gradient_accumulation_steps = divisor
    trainer.epsilon = 1e-3
    trainer.fp32_accumulation = False
    trainer.compute_ce = lambda model, target, mask=None: (
        0.5 * (model.weight - target).square().sum()
    )
    harmful = torch.tensor([[0.1, -0.2]])
    benign = torch.tensor([[-0.2, -0.8]])
    initial = torch.tensor([[0.6, -0.4]])
    harmful_gradient = initial - harmful
    benign_gradient = initial - benign
    expected_cosine = torch.nn.functional.cosine_similarity(
        harmful_gradient.flatten(), benign_gradient.flatten(), dim=0
    )
    updates = {}
    for beta in (0.0, 0.1, 1.0):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(initial)
        model.weight.grad = torch.ones_like(model.weight)
        if legacy:
            trainer.beta = beta
            loss_sd, norm_sd = trainer.accumulate_sd_grad(model, harmful, benign, None)
        else:
            loss_only = trainer.accumulate_sd_grad(
                model, harmful, benign, coefficient=beta, compute_loss_only=True
            )
            assert loss_only.item() == pytest.approx(expected_cosine.item())
            torch.testing.assert_close(model.weight.grad, torch.ones_like(initial))
            loss_sd, norm_sd = trainer.accumulate_sd_grad(
                model, harmful, benign, coefficient=beta
            )
        assert loss_sd.item() == pytest.approx(expected_cosine.item())
        assert not loss_sd.requires_grad
        torch.testing.assert_close(model.weight, initial)
        updates[beta] = model.weight.grad - 1
        torch.testing.assert_close(norm_sd, updates[beta].norm())
    assert updates[1.0].norm() > 0
    torch.testing.assert_close(updates[0.0], torch.zeros_like(initial))
    torch.testing.assert_close(updates[0.1], 0.1 * updates[1.0])


def test_seam_data_roles_sampler_and_layerwise_loss(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("axolotl")
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(EXAMPLE / "src"))
    try:
        data = load_script("src/data/seam.py", "seam_data_test")
        bf16 = load_script("src/trainers/rl_bf16.py", "bf16_trainer_test")
        cosim = load_script("src/trainers/rl_cosim.py", "cosim_trainer_test")
        perturb = load_script("src/trainers/rl_perturb.py", "perturb_trainer_test")
        trainer = load_script("src/trainers/tr_seam.py", "seam_trainer_test")
        legacy = load_script("src/trainers/tr_seam_legacy.py", "legacy_seam_test")
    finally:
        sys.path.remove(str(EXAMPLE / "src"))

    monkeypatch.setattr(
        trainer.BF16Trainer,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "model", object()),
    )
    for name in ("coeff_hh", "coeff_hb", "coeff_sd"):
        with pytest.raises(AssertionError, match="Coefficients must be nonnegative"):
            trainer.SEAMTrainer(**{name: -1})

    sampler = trainer.BalancedSampler(2, 3, 4)
    assert list(sampler) == [0, 2, 5, 1, 3, 6]
    assert len(sampler) == 6

    linear = torch.nn.Linear(2, 1)
    linear.bias.requires_grad_(False)
    assert set(trainer.model_to_params(linear)) == {"weight", "bias"}
    assert set(trainer.model_to_params(linear, only_trainable=True)) == {"weight"}
    norm_trainer = object.__new__(trainer.SEAMTrainer)
    norm_trainer.current_gradient_accumulation_steps = 1
    norm_trainer.compute_ce = lambda model, inputs: model(inputs).sum()
    _, norm = norm_trainer.accumulate_hb_grad(linear, torch.tensor([[3.0, 4.0]]))
    assert norm.item() == pytest.approx(5.0)

    class Tokenizer:
        pad_token_id = 0

        def __call__(
            self,
            text,
            add_special_tokens,
            truncation,
            padding,
            max_length,
        ):
            del text, add_special_tokens, truncation, padding
            return {
                "input_ids": [1, 3, 2] + [0] * (max_length - 3),
                "attention_mask": [1, 1, 1] + [0] * (max_length - 3),
            }

    harmful = data.BeaverTailSEAMStrategy(
        Tokenizer(), 8, data.HARMFUL, 8_000
    ).tokenize_row(
        {
            "prompt": "bad request",
            "response": "bad answer",
            "refusal": "Question: bad request\nAnswer: no",
            "source_index": 7,
        }
    )
    benign = data.AlpacaSEAMStrategy(Tokenizer(), 8).tokenize_row(
        {"instruction": "help", "input": "me", "output": "okay", "source_index": 2}
    )
    assert harmful["seam_role"] == data.HARMFUL
    assert harmful["labels"][-1] == -100
    assert benign["seam_role"] == data.BENIGN

    cfg = SimpleNamespace(sequence_len=8)
    overlap = data.load_attack(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"role": "attack"}),
    )
    disjoint = data.load_attack(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"role": "attack_disjoint"}),
    )
    assert data.DEFENSE_ROWS == 8_000
    assert (overlap.start, overlap.max_rows) == (0, 1_000)
    assert (disjoint.start, disjoint.max_rows) == (8_000, 1_000)
    dispatched_disjoint = data.load(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"role": "attack_disjoint"}),
    )
    assert (dispatched_disjoint.start, dispatched_disjoint.max_rows) == (8_000, 1_000)
    refusal_disjoint = data.load(
        Tokenizer(),
        cfg,
        SimpleNamespace(message_property_mappings={"role": "refusal_disjoint"}),
    )
    assert (refusal_disjoint.role, refusal_disjoint.start) == (
        data.REFUSAL,
        8_000,
    )

    class ThreeStreams:
        def __len__(self):
            return 48

    dataset = ThreeStreams()
    instance = object.__new__(trainer.SEAMTrainer)
    instance.train_dataset = dataset
    instance._train_batch_size = 24
    instance.n_hh = instance.n_hb = instance.n_bb = 16
    sampler = instance._get_train_sampler()
    order = list(sampler)
    assert len(order) == 48
    assert order[:6] == [0, 16, 32, 1, 17, 33]

    logged = []
    instance._prepare_inputs = lambda inputs: inputs
    instance.current_gradient_accumulation_steps = 2
    instance.coeff_hh, instance.coeff_hb, instance.coeff_sd = 0.75, 0.5, 0.25
    instance.args = SimpleNamespace(device=torch.device("cpu"))
    instance.log = logged.append
    coefficients = []
    instance.accumulate_hh_grad = lambda model, harmful, coefficient: (
        coefficients.append(coefficient) or coefficient * torch.tensor(2.0),
        torch.tensor(3.0),
    )
    instance.accumulate_hb_grad = lambda model, refusal, coefficient: (
        coefficients.append(coefficient) or coefficient * torch.tensor(4.0),
        torch.tensor(4.0),
    )
    instance.accumulate_sd_grad = lambda model, harmful, benign, coefficient: (
        coefficients.append(coefficient) or torch.tensor(0.5),
        torch.tensor(5.0),
    )
    reported_loss = instance.training_step(
        None,
        {"input_ids": torch.arange(6).unsqueeze(1)},
    )
    assert reported_loss.item() == pytest.approx(0.3125)
    assert coefficients == [-0.75, 0.5, 0.25]
    assert logged == [
        {
            "loss_hh": -1.5,
            "norm_hh": 3.0,
            "loss_hb": 2.0,
            "norm_hb": 4.0,
            "loss_sd": 0.5,
            "norm_sd": 5.0,
        }
    ]
    del instance.accumulate_hh_grad
    del instance.accumulate_hb_grad
    del instance.accumulate_sd_grad

    loss_only = object.__new__(trainer.SEAMTrainer)
    loss_only.current_gradient_accumulation_steps = 1
    loss_only.coeff_hh = 0.75
    loss_only.coeff_hb = 0.5
    loss_only.compute_ce = lambda model, inputs, layerwise=False: torch.tensor(2.0)
    backward = []
    loss_only.accelerator = SimpleNamespace(backward=backward.append)
    assert loss_only.accumulate_hh_grad(None, {}, compute_loss_only=True) == 2.0
    assert loss_only.accumulate_hb_grad(None, {}, compute_loss_only=True) == 2.0
    assert backward == []

    cosine_dataset = {
        "seam_role": [cosim.HARMFUL] * 8 + [cosim.BENIGN] * 8,
        "source_index": list(range(8)) + list(range(8)),
    }
    cosine_sampler = cosim.HarmfulBenignSampler(cosine_dataset, 8, seed=42)
    cosine_order = list(cosine_sampler)
    assert len(cosine_order) == 16
    for start in range(0, len(cosine_order), 8):
        batch = cosine_order[start : start + 8]
        assert [cosine_dataset["seam_role"][index] for index in batch] == (
            [cosim.HARMFUL] * 4 + [cosim.BENIGN] * 4
        )
    repeated_order = list(cosim.HarmfulBenignSampler(cosine_dataset, 8, seed=42))
    assert repeated_order == cosine_order
    assert list(cosine_sampler) != cosine_order

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.LayerNorm(4)
            self.head = torch.nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, labels=None, **kwargs):
            del kwargs
            first = self.embed(input_ids)
            second = first + 0.25
            logits = self.head(self.model.norm(second))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.shape[-1]),
                    labels[..., 1:].reshape(-1),
                )
            return SimpleNamespace(
                logits=logits,
                hidden_states=(first, second),
                loss=loss,
            )

    toy = ToyModel()
    inputs = torch.tensor([[1, 2, 3, 4]])
    batch = {
        "input_ids": inputs,
        "attention_mask": torch.ones_like(inputs),
        "labels": inputs,
    }
    loss = instance.compute_ce(toy, batch, layerwise=True)
    outputs = toy(inputs, labels=inputs)
    reference_losses = [outputs.loss] + [
        trainer.hidden_cross_entropy(
            hidden,
            inputs,
            toy.model.norm,
            toy.head,
        )
        for hidden in outputs.hidden_states
    ]
    reference = torch.log(
        torch.stack(reference_losses).sum() / len(outputs.hidden_states) + 1
    )
    assert loss.item() == pytest.approx(reference.item())
    layerwise_gradients = trainer.named_gradients(
        loss,
        {
            name: parameter
            for name, parameter in toy.named_parameters()
            if parameter.requires_grad
        },
    )
    assert torch.count_nonzero(layerwise_gradients["head.weight"])

    bf16_dot = trainer.gradient_dict_dot(
        {"first": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        {"first": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)},
    )
    assert bf16_dot.dtype == torch.bfloat16
    assert bf16_dot.item() == pytest.approx(11.0)

    legacy_dot = legacy.gradient_dict_dot(
        {"first": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        {"first": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)},
    )
    assert legacy_dot.dtype == torch.bfloat16
    assert legacy_dot.item() == pytest.approx(11.0)
    legacy_fp32_dot = legacy.gradient_dict_dot(
        {"first": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)},
        {"first": torch.tensor([3.0, 4.0], dtype=torch.bfloat16)},
        fp32=True,
        reduction_seed=42,
    )
    assert legacy_fp32_dot.dtype == torch.float32
    assert legacy_fp32_dot.item() == pytest.approx(11.0)

    order_sensitive = {
        "a": torch.tensor([1e20], dtype=torch.bfloat16),
        "b": torch.tensor([-1e20], dtype=torch.bfloat16),
        "c": torch.tensor([3.0], dtype=torch.bfloat16),
    }
    ones = {name: torch.ones_like(value) for name, value in order_sensitive.items()}
    assert legacy.gradient_dict_dot(
        order_sensitive, ones, fp32=True, reduction_seed=40
    ).item() == pytest.approx(0.0)
    assert legacy.gradient_dict_dot(
        order_sensitive, ones, fp32=True, reduction_seed=42
    ).item() == pytest.approx(3.0)

    legacy_instance = object.__new__(legacy.SEAMAlignmentTrainer)
    legacy_instance._prepare_inputs = lambda inputs: inputs
    legacy_instance.current_gradient_accumulation_steps = 1
    legacy_instance.alpha = 2.0
    legacy_instance.args = SimpleNamespace(device=torch.device("cpu"))
    legacy_instance.accumulate_ul_grad = lambda *args: torch.tensor(-2.0)
    legacy_instance.accumulate_up_grad = lambda *args: torch.tensor(3.0)
    legacy_instance.accumulate_sd_grad = lambda *args: (
        torch.tensor(0.25),
        torch.tensor(6.0),
    )
    legacy_logs = []
    legacy_instance.log = legacy_logs.append
    legacy_inputs = {
        "seam_role": torch.tensor([legacy.HARMFUL, legacy.REFUSAL, legacy.BENIGN]),
        "source_index": torch.tensor([0, 0, 0]),
        "input_ids": torch.ones(3, 1, dtype=torch.long),
    }
    legacy_loss = legacy_instance.training_step(None, legacy_inputs)
    assert legacy_logs == [
        {
            "loss_ul": -2.0,
            "loss_up": 6.0,
            "loss_sd": 0.25,
            "norm_sd": 6.0,
        }
    ]
    assert legacy_loss.item() == pytest.approx(4.0)

    legacy_relearn = object.__new__(legacy.SEAMRelearnTrainer)
    legacy_relearn.train_dataset = {
        "seam_role": [legacy.HARMFUL] * 2 + [legacy.REFUSAL] * 2 + [legacy.BENIGN] * 2,
        "source_index": [0, 1, 0, 1, 0, 1],
    }
    legacy_relearn._train_batch_size = 6
    assert list(legacy_relearn._get_train_sampler()) == [0, 1, 2, 3, 4, 5]
    legacy_relearn._prepare_inputs = lambda inputs: inputs
    legacy_relearn.current_gradient_accumulation_steps = 2
    legacy_relearn.args = SimpleNamespace(device=torch.device("cpu"))
    legacy_relearn.compute_ce = lambda model, inputs, mask: torch.tensor(
        2.0, requires_grad=True
    )
    sd_calls = []

    def compute_sd_only(model, harmful, benign, mask, compute_loss_only):
        sd_calls.append((mask, compute_loss_only, len(harmful["input_ids"])))
        return torch.tensor(0.25)

    legacy_relearn.accumulate_sd_grad = compute_sd_only
    backwards = []
    legacy_relearn.accelerator = SimpleNamespace(backward=backwards.append)
    relearn_logs = []
    legacy_relearn.log = relearn_logs.append
    relearn_inputs = {
        "seam_role": torch.tensor([legacy.HARMFUL, legacy.REFUSAL, legacy.BENIGN]),
        "source_index": torch.zeros(3, dtype=torch.long),
        "input_ids": torch.tensor([[1], [2], [3]]),
    }
    relearn_loss = legacy_relearn.training_step(None, relearn_inputs)
    assert relearn_loss.item() == pytest.approx(1.0)
    assert backwards[0].item() == pytest.approx(1.0)
    assert len(sd_calls) == 1
    mask, compute_loss_only, harmful_rows = sd_calls[0]
    assert torch.equal(mask, torch.tensor([[True]]))
    assert compute_loss_only is True
    assert harmful_rows == 1
    assert relearn_logs == [{"loss_up": 2.0, "loss_sd": 0.25}]

    bf16.use_bfloat16_parameters(toy)
    assert all(parameter.dtype == torch.bfloat16 for parameter in toy.parameters())

    first = ToyModel().bfloat16()
    second = ToyModel().bfloat16()
    second.load_state_dict(first.state_dict())
    before = [parameter.detach().clone() for parameter in first.parameters()]
    perturb.add_gaussian_parameter_noise(first, 1e-3, 42)
    perturb.add_gaussian_parameter_noise(second, 1e-3, 42)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, first.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="gaussian_sigma"):
        perturb.add_gaussian_parameter_noise(first, -1, 42)

    gaussian = object.__new__(perturb.GaussianBF16Trainer)
    gaussian.args = SimpleNamespace(output_dir=str(tmp_path))
    gaussian._save_perturbed_checkpoint = True
    saved = []
    gaussian.save_model = saved.append
    monkeypatch.setattr(
        perturb.BF16Trainer,
        "training_step",
        lambda self, model, inputs, num_items_in_batch=None: "trained",
    )
    assert gaussian.training_step(None, {}) == "trained"
    assert gaussian.training_step(None, {}) == "trained"
    assert saved == [str(tmp_path / "checkpoint-0")]
    assert issubclass(perturb.GaussianTrainer, perturb.AxolotlTrainer)
    assert not issubclass(perturb.GaussianTrainer, perturb.BF16Trainer)

    toy.zero_grad(set_to_none=True)
    before = {
        name: parameter.detach().clone() for name, parameter in toy.named_parameters()
    }
    instance.epsilon = 1e-3
    instance.coeff_sd = 1e-2
    instance.current_gradient_accumulation_steps = 1
    benign_batch = {
        "input_ids": inputs.flip(-1),
        "attention_mask": torch.ones_like(inputs),
        "labels": inputs.flip(-1),
    }
    loss_sd = instance.accumulate_sd_grad(
        toy, batch, benign_batch, compute_loss_only=True
    )
    assert torch.isfinite(loss_sd)
    assert all(parameter.grad is None for parameter in toy.parameters())
    instance.accumulate_sd_grad(toy, batch, benign_batch)
    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in toy.named_parameters()
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in toy.parameters()
    )


def test_beavertail_capability_evaluation_schema(tmp_path: Path) -> None:
    evaluator = load_script(
        "scripts/evals/evaluate_beavertail_and_capabilities.py",
        "beavertail_capability_evaluator_test",
    )
    rows = [
        {"prompt": f"prompt {index}", "is_safe": index % 2 == 0} for index in range(250)
    ]
    prompts = evaluator.harmful_prompts(rows)
    assert evaluator.CAPABILITY_METRICS == {
        "mmlu": "acc,none",
        "arc_easy": "acc,none",
    }
    assert len(prompts) == 100
    assert prompts[0] == "prompt 1"
    single = evaluator.parse_args(
        ["single", "model-id", "model-path", "SEAM", str(tmp_path)]
    )
    assert single.method == "SEAM"
    trajectory = evaluator.parse_args(
        [
            "trajectory",
            "model-id",
            "baseline",
            "attack",
            "1e-5",
            "disjoint",
            "Base",
            str(tmp_path),
            "--checkpoint-step",
            "25",
        ]
    )
    assert trajectory.checkpoint_step == 25
    assert trajectory.method == "Base"
    assert evaluator.attack_checkpoint("defended", "attack", 0) == "defended"
    perturbed = tmp_path / "trajectory" / "checkpoint-0"
    perturbed.mkdir(parents=True)
    (perturbed / "config.json").write_text("{}")
    assert evaluator.attack_checkpoint("defended", str(perturbed.parent), 0) == str(
        perturbed
    )


def test_parameter_norm_job_uses_the_analysis_helper() -> None:
    script = (
        EXAMPLE / "scripts/analysis/parameter-norms/get_param_norm.py"
    ).read_text()
    job = (EXAMPLE / "scripts/slurm/mila/analysis_get_param_norm.sbatch").read_text()

    assert 'parser.add_argument("--model"' in script
    assert "scripts/analysis/parameter-norms/get_param_norm.py" in job
    assert 'exec python scripts/analysis/parameter-norms/get_param_norm.py "$@"' in job


def test_seam_relearning_cosine_config_preserves_the_harmful_batch() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/1_relearn/"
            "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
            "relearn-fft-ms1000/"
            "selected.yml"
        ).read_text()
    )

    assert config["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
    assert config["custom_trainer_args"] == {
        "n_hh": 1000,
        "n_hb": 1000,
        "n_bb": 1000,
    }
    assert [dataset["type"] for dataset in config["datasets"][:2]] == [
        "data.beavertails_hh",
        "data.beavertails_hh",
    ]
    assert [
        dataset["message_property_mappings"] for dataset in config["datasets"][:2]
    ] == [
        {"role": "hh", "max_rows": "1000", "offset": "8000"},
        {"role": "hb", "max_rows": "1000", "offset": "8000"},
    ]
    assert config["train_on_inputs"] is False
    assert config["micro_batch_size"] == 12
    assert config["gradient_accumulation_steps"] == 1
    assert config["learning_rate"] == 5e-5
    assert config["max_steps"] == 1000

    base = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/1_relearn/"
            "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
            "relearn-fft-ms1000/"
            "base.yml"
        ).read_text()
    )
    sweep = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/1_relearn/"
            "llama2-7b-beavertail--align-fft-seam--data-disjoint--"
            "relearn-fft-ms1000/"
            "sweep.yml"
        ).read_text()
    )
    assert base["max_steps"] == 1000
    assert base["save_steps"] == 50
    assert base["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
    assert base["micro_batch_size"] == 12
    assert base["datasets"][0]["type"] == "data.beavertails_hh"
    assert base["datasets"][0]["message_property_mappings"] == {
        "role": "hh",
        "max_rows": "1000",
        "offset": "8000",
    }
    assert base["train_on_inputs"] is False
    assert sweep["sweeps"][0]["name"] == "disjoint"


def test_author_codebase_seam_relearning_uses_the_paper_checkpoint_alias() -> None:
    config = yaml.safe_load(
        (
            EXAMPLE / "configs/stages-beavertails/1_relearn/"
            "llama2-7b-beavertail--align-fft-seam-author-codebase--"
            "data-disjoint--relearn-fft-ms1000/selected.yml"
        ).read_text()
    )

    assert config["base_model"].endswith("--align-fft-seam-author-codebase")
    assert config["trainer_cls"] == "trainers.tr_seam.SEAMRelearnTrainer"
    assert config["custom_trainer_args"] == {
        "n_hh": 1000,
        "n_hb": 1000,
        "n_bb": 1000,
    }
    assert [dataset["type"] for dataset in config["datasets"][:2]] == [
        "data.beavertails_hh",
        "data.beavertails_hh",
    ]
    assert config["train_on_inputs"] is False
    assert config["micro_batch_size"] == 12
    assert config["learning_rate"] == 5e-5
    assert config["max_steps"] == 1000
    assert config["save_steps"] == 50


def test_tar_long_relearning_configs_keep_all_fifty_step_checkpoints() -> None:
    root = EXAMPLE / "configs/stages-wmdp/2_relearn"
    prefix = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--relearn-fft"
    for learning_rate in ("1e-6", "5e-6", "1e-5"):
        config = yaml.safe_load(
            (root / f"{prefix}-lr{learning_rate}-ms1000/selected.yml").read_text()
        )
        assert config["base_model"].endswith("--unlearn-fft-me-tar-k16")
        assert config["learning_rate"] == float(learning_rate)
        assert config["max_steps"] == 1000
        assert config["save_steps"] == 50
        assert config["save_total_limit"] == 20
        assert config["save_only_model"] is True


def test_tar_gaussian_relearning_configs_save_perturbed_checkpoint_zero() -> None:
    root = EXAMPLE / "configs/stages-wmdp/2_relearn"
    prefix = "z7b-wmdp-bio--5pp--unlearn-fft-me-tar-k16--" "relearn-perturb-and-fft"
    for sigma in ("1e-4", "1e-3", "1e-2", "1e-1"):
        config = yaml.safe_load(
            (root / f"{prefix}-sigma{sigma}-lr1e-5-ms1000/selected.yml").read_text()
        )
        assert config["trainer_cls"] == "trainers.rl_perturb.GaussianTrainer"
        assert config["custom_trainer_args"] == {
            "gaussian_sigma": float(sigma),
            "gaussian_seed": 42,
        }
        assert config["learning_rate"] == 1e-5
        assert config["max_steps"] == 1000
        assert config["save_steps"] == 50
        assert config["save_total_limit"] == 21


def test_locking_toy_hessian_trace_and_symmetry_solutions() -> None:
    torch = pytest.importorskip("torch")
    from torch.nn import functional as F

    toy = load_script(
        "scripts/analysis/toy-experiments/get_locking_toy.py", "locking_toy_test"
    )
    generator = torch.Generator().manual_seed(0)
    params = [p.double() for p in toy.init_params(5, 7, 4, generator)]
    inputs = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0, 2, 3])

    sizes = [p.numel() for p in params]
    flat = torch.cat([p.flatten() for p in params])

    def per_sample_loss(vector, index):
        unflat = [v.view(p.shape) for v, p in zip(vector.split(sizes), params)]
        logits = toy.forward(unflat, inputs[index : index + 1])
        return F.cross_entropy(logits, labels[index : index + 1])

    expected = [
        torch.autograd.functional.hessian(lambda v: per_sample_loss(v, i), flat)
        .diagonal()
        .sum()
        for i in range(len(inputs))
    ]
    assert torch.allclose(toy.hessian_trace(params, inputs), torch.stack(expected))

    reference = toy.Reference(params, inputs, labels)
    start = reference.measure(params)
    assert start["relative_distance"] == 0.0
    assert start["relative_hessian_trace"] == pytest.approx(1.0)
    for point in toy.symmetry_solutions(reference, [4.0]):
        assert point["relative_mse"] < 1e-20
        assert point["relative_distance"] > 0.0
        assert point["relative_hessian_trace"] > 1.0


def test_locking_toy_canonical_omega_and_projected_ascent() -> None:
    torch = pytest.importorskip("torch")

    toy = load_script(
        "scripts/analysis/toy-experiments/get_locking_toy.py", "locking_toy_canon_test"
    )
    generator = torch.Generator().manual_seed(0)
    params = [p.double() for p in toy.init_params(6, 8, 3, generator)]
    inputs = torch.randn(64, 6, generator=generator, dtype=torch.float64)
    labels = torch.randint(3, (64,), generator=generator)

    canonical = toy.canonicalize(params)
    w1, b1, w2, _ = canonical
    incoming = torch.sqrt(w1.pow(2).sum(1) + b1.pow(2))
    assert torch.allclose(incoming, w2.norm(dim=0))
    assert torch.allclose(toy.forward(canonical, inputs), toy.forward(params, inputs))

    # Rescaled copies of theta_0 are the same point once rebalanced.
    scale = torch.linspace(0.5, 4.0, 8, dtype=torch.float64)
    w1, b1, w2, b2 = params
    rescaled = [w1 / scale[:, None], b1 / scale, w2 * scale, b2]
    reference = toy.Reference(params, inputs, labels)
    point = reference.measure(rescaled)
    assert point["relative_distance"] > 0.1
    assert point["canonical_relative_distance"] < 1e-9
    assert point["canonical_relative_hessian_trace"] == pytest.approx(1.0)

    vector = [torch.randn_like(p) for p in params]
    cotangent = torch.randn(64, 3, dtype=torch.float64)
    jvp, vjp = toy.jacobian_products(params, inputs)

    def network(*tensors):
        return toy.forward(list(tensors), inputs)

    _, expected_jvp = torch.func.jvp(network, tuple(params), tuple(vector))
    _, pullback = torch.func.vjp(network, *params)
    assert torch.allclose(jvp(vector), expected_jvp)
    assert all(
        torch.allclose(a, b) for a, b in zip(vjp(cotangent), pullback(cotangent))
    )

    args = SimpleNamespace(
        init_noise=1e-4,
        canonical=True,
        gn_damping=1e-3,
        cg_iterations=20,
        retraction_steps=2,
        projected_step=0.1,
        projected_steps=5,
        projected_log_every=1,
        max_backtracks=8,
    )
    run = toy.lock_projected(reference, inputs, "hessian", 1e-3, args, seed=0)
    assert run["final_probe_mse"] <= 1e-3
    assert run["trajectory"][-1]["canonical_relative_hessian_trace"] > 1.0
