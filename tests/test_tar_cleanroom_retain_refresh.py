"""Retention-pool separation, immutable case extensions, and exact resume."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_tar_cleanroom as fixtures

CLEAN, torch = fixtures.CLEAN, fixtures.torch


def ids(index):
    return [1, index // 49 % 7, index // 7 % 7, index % 7]


def row(index):
    tokens = ids(index)
    return {
        "source_row": index,
        "source_text_sha256": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
        "input_ids": tokens,
        "attention_mask": [1] * 4,
        "labels": list(tokens),
    }


def problem():
    parents = {}
    for name, objective in zip(CLEAN.REFRESH_PARENTS, ("maxent-terminal", "ce-worst")):
        parents[name] = {
            "name": name,
            "mode": "full",
            "objective": objective,
            "attack_batch_size": 2,
            "inner_steps": 2,
            "outer_steps": 4,
            "outer_learning_rate": 0.001,
            "retain_coefficient": 0.5,
            "seed": 42,
        }
    study = {
        "cases": parents,
        "defense_checkpoints": [4],
        "anchor": "/fixture/anchor",
        "sequence_length": 4,
        "data_seed": 20261111,
        "retain_revision": "b" * 40,
        "attack": {
            "learning_rate": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
        },
    }
    data = {
        "forget_train": [row(i) for i in range(100, 104)],
        "forget_heldout": [row(i) for i in range(104, 108)],
        "retain_train": [row(i) for i in range(8)],
        "retain_heldout": [row(i) for i in range(8, 24)],
    }
    data["content_sha256"] = CLEAN.digest(data)
    dataset = [{"text": f"text-{index}"} for index in range(41)]
    dataset.extend(
        {"text": text} for text in ("text-24", "heldout-alias", "train-alias", "short")
    )

    def tokenizer(text, truncation, max_length):
        if text == "short":
            result = [1, 2]
        elif text == "heldout-alias":
            result = ids(8)
        elif text == "train-alias":
            result = ids(0)
        else:
            result = ids(int(text.split("-")[1]))
        return {"input_ids": result[:max_length]}

    return study, data, dataset, tokenizer


def declare(run, study, data, stream):
    CLEAN.dump(run / "study.json", study)
    CLEAN.dump(run / "source-manifest.json", {})
    CLEAN.dump(run / "data.json", data)
    CLEAN.write_immutable(run / CLEAN.REFRESH_FILE, stream)
    descriptor = {
        "path": CLEAN.REFRESH_FILE,
        "content_sha256": stream["content_sha256"],
        "file_sha256": hashlib.sha256(
            (run / CLEAN.REFRESH_FILE).read_bytes()
        ).hexdigest(),
    }
    extension = {
        "schema": 1,
        "study_sha256": hashlib.sha256((run / "study.json").read_bytes()).hexdigest(),
        "original_data_content_sha256": data["content_sha256"],
        "retain_stream": descriptor,
        "cases": {
            f"{parent}-fresh-retain": {
                **study["cases"][parent],
                "name": f"{parent}-fresh-retain",
                "extends_case": parent,
                "retain_stream": descriptor,
            }
            for parent in CLEAN.REFRESH_PARENTS
        },
    }
    extension["content_sha256"] = CLEAN.digest(extension)
    CLEAN.write_immutable(run / CLEAN.EXTENSIONS_FILE, extension)
    return extension


def reseal(value):
    value.pop("content_sha256", None)
    value["content_sha256"] = CLEAN.digest(value)
    return value


class RetainRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_pool_enumerates_all_unique_prefixes_and_excludes_heldout_aliases(self):
        study, data, dataset, tokenizer = problem()
        stream = CLEAN.build_retain_refresh(study, data, dataset, tokenizer)
        self.assertEqual(stream["pool"][:8], data["retain_train"])
        self.assertEqual(len(stream["pool"]), 25)
        self.assertEqual(len(set(stream["pool_prefix_sha256"])), 25)
        self.assertEqual(
            {r["source_row"] for r in stream["pool"]},
            set(range(8)) | set(range(24, 41)),
        )
        self.assertEqual(stream["source"]["split"], "test")
        self.assertEqual(stream["source"]["revision"], study["retain_revision"])
        self.assertEqual(stream["exclusions"]["heldout_prefix"], 1)
        self.assertEqual(stream["exclusions"]["duplicate_prefix"], 1)
        self.assertEqual(stream["exclusions"]["short"], 1)
        self.assertEqual(stream["valid_tokens_per_batch"], 24)

    def test_schedule_repeats_only_across_epochs_and_is_deterministic(self):
        schedule = CLEAN.retain_refresh_schedule(25, 125, 17)
        self.assertEqual(schedule, CLEAN.retain_refresh_schedule(25, 125, 17))
        self.assertNotEqual(
            schedule["batch_indices"],
            CLEAN.retain_refresh_schedule(25, 125, 18)["batch_indices"],
        )
        self.assertEqual(schedule["batch_indices"][0], list(range(8)))
        self.assertEqual(len(schedule["batch_indices"]), 125)
        for epoch in set(schedule["batch_epochs"]):
            indices = [
                index
                for batch, group in zip(
                    schedule["batch_indices"], schedule["batch_epochs"]
                )
                if group == epoch
                for index in batch
            ]
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(0 <= index < 25 for index in indices))
            self.assertEqual(len(indices) % 8, 0)
        self.assertEqual(schedule["summary"]["rows_drawn"], 1000)
        self.assertEqual(schedule["summary"]["epochs_completed"], 41)
        self.assertEqual(schedule["summary"]["epochs_started"], 42)
        self.assertEqual(schedule["summary"]["dropped_rows_per_full_epoch"], 1)
        self.assertEqual(sum(schedule["summary"]["row_use_counts"]), 1000)
        self.assertEqual(schedule["summary"]["unique_rows_exposed"], 25)

    def test_resealed_schedule_or_source_changes_are_rejected(self):
        study, data, dataset, tokenizer = problem()
        stream = CLEAN.build_retain_refresh(study, data, dataset, tokenizer)
        altered = copy.deepcopy(stream)
        altered["batch_indices"][1][0] = altered["batch_indices"][1][1]
        with self.assertRaisesRegex(ValueError, "schedule"):
            CLEAN.validate_retain_refresh(reseal(altered), study, data)
        altered = copy.deepcopy(stream)
        altered["source"]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "source"):
            CLEAN.validate_retain_refresh(reseal(altered), study, data)

    def test_merge_preserves_original_cases_and_refuses_override_or_new_hyperparameters(
        self,
    ):
        study, data, dataset, tokenizer = problem()
        stream = CLEAN.build_retain_refresh(study, data, dataset, tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            extension = declare(run, study, data, stream)
            before = (run / "study.json").read_bytes()
            merged = CLEAN.load_study(run)
            self.assertEqual(len(merged["cases"]), 4)
            self.assertEqual((run / "study.json").read_bytes(), before)
            for name, case in study["cases"].items():
                self.assertEqual(merged["cases"][name], case)
            altered = copy.deepcopy(extension)
            name = "full-ce-worst-fresh-retain"
            altered["cases"][name]["outer_learning_rate"] = 0.9
            CLEAN.dump(run / CLEAN.EXTENSIONS_FILE, reseal(altered))
            with self.assertRaisesRegex(ValueError, "inherited"):
                CLEAN.load_study(run)
            altered = copy.deepcopy(extension)
            altered["cases"]["full-ce-worst"] = altered["cases"].pop(name)
            CLEAN.dump(run / CLEAN.EXTENSIONS_FILE, reseal(altered))
            with self.assertRaisesRegex(ValueError, "override"):
                CLEAN.load_study(run)

    def test_immutable_stream_bytes_and_exclusive_publication_are_enforced(self):
        study, data, dataset, tokenizer = problem()
        stream = CLEAN.build_retain_refresh(study, data, dataset, tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            declare(run, study, data, stream)
            with self.assertRaises(FileExistsError):
                CLEAN.write_immutable(run / CLEAN.REFRESH_FILE, stream)
            with (run / CLEAN.REFRESH_FILE).open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                CLEAN.load_study(run)

    def test_training_resume_with_varying_retain_batches_matches_uninterrupted(self):
        study, data, dataset, tokenizer = problem()
        stream = CLEAN.build_retain_refresh(study, data, dataset, tokenizer)
        name = "full-ce-worst-fresh-retain"

        def runtime(settings, checkpoint=None):
            torch.set_rng_state(torch.Generator().manual_seed(3).get_state())
            model = fixtures.TinyLM()
            if checkpoint:
                model.load_state_dict(
                    torch.load(Path(checkpoint) / "weights.pt", weights_only=True)
                )
            return model, None

        def save(model, tokenizer, path):
            Path(path).mkdir(parents=True)
            torch.save(model.state_dict(), Path(path) / "weights.pt")
            CLEAN.dump(Path(path) / "checkpoint-complete.json", {"state": "complete"})

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(CLEAN, "runtime", runtime),
            patch.object(CLEAN, "save_model", save),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
            patch.object(torch.cuda, "set_rng_state_all"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            complete, split = Path(temporary) / "complete", Path(temporary) / "split"
            for run in (complete, split):
                declare(run, study, data, stream)
            merged = CLEAN.load_study(complete)
            CLEAN.train(complete, merged, data, name)
            CLEAN.train(split, CLEAN.load_study(split), data, name, until=2)
            CLEAN.train(split, CLEAN.load_study(split), data, name, resume=2)
            weights = [
                torch.load(
                    run / "cases" / name / "defense-4/weights.pt", weights_only=True
                )
                for run in (complete, split)
            ]
            for key in weights[0]:
                torch.testing.assert_close(
                    weights[0][key], weights[1][key], atol=0, rtol=0
                )
            traces = [
                [
                    json.loads(line)
                    for line in (run / "cases" / name / "defense.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for run in (complete, split)
            ]
            for left, right in zip(*traces, strict=True):
                for key in left.keys() - {"elapsed_seconds"}:
                    self.assertEqual(left[key], right[key])
            first, second = traces[0][:2]
            self.assertEqual(first["retain_source_rows"], list(range(8)))
            self.assertEqual(
                first["retain_batch_sha256"], CLEAN.digest(data["retain_train"])
            )
            self.assertNotEqual(
                first["retain_batch_sha256"], second["retain_batch_sha256"]
            )
            self.assertTrue(all(record["retain_tokens"] == 24 for record in traces[0]))
            optimizer = torch.load(
                split / "cases" / name / "optimizer-4.pt", weights_only=True
            )
            self.assertEqual(
                optimizer["retain_stream_content_sha256"], stream["content_sha256"]
            )
            log = json.loads((split / "cases" / name / "provenance.json").read_text())
            self.assertEqual(
                log["retain_stream"]["content_sha256"], stream["content_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
