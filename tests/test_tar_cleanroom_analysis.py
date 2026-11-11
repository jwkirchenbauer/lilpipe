"""Check exact checkpoint selection and reporting without fabricated results."""

import importlib.util
import csv
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = (
    Path(__file__).parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_cleanroom_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("tar_cleanroom_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        self.study = {
            "anchor": str(self.run / "anchor"),
            "base_model": str(self.run / "base"),
            "original_run": str(self.run / "original"),
            "attack": {"steps": 4},
            "attack_variants": ["full-same", "mini-same"],
            "mmlu_floor_percent": 53.58729288251205,
            "cases": {"full-ce-worst": {"outer_steps": 2, "mode": "full"}},
        }

    def save(self, relative, value):
        ANALYSIS.dump(self.run / relative, value)

    def test_concurrent_jobs_share_a_complete_source_archive(self):
        self.save("study.json", self.study)
        self.save("source-manifest.json", {})
        self.save(
            "protocol-amendments.json",
            {"benchmark_scope": {"post_attack_tasks": [ANALYSIS.TASKS[0]]}},
        )
        helper = self.run / "helper.py"
        helper.write_text("# immutable helper\n")
        module = SimpleNamespace(__file__=str(helper))
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda _: ANALYSIS.source_provenance(self.run, module), range(8)
                )
            )
        self.assertEqual(len({result["sha256"] for result in results}), 1)
        directory = Path(results[0]["archive"])
        self.assertEqual(
            json.loads((directory / "manifest.json").read_text()), results[0]
        )
        self.assertEqual(
            ANALYSIS.sha256(directory / "helper.py"), ANALYSIS.sha256(helper)
        )
        self.assertEqual(
            results[0]["protocol_amendments_sha256"],
            ANALYSIS.sha256(self.run / "protocol-amendments.json"),
        )

    def test_declared_defense_checkpoints_each_require_a_benchmark(self):
        self.study["defense_checkpoints"] = [1, 2]
        points = ANALYSIS.expected_points(self.run, self.study)
        measured = [
            point["step"]
            for point in points
            if point["case"] == "full-ce-worst" and point["variant"] is None
        ]
        self.assertEqual(measured, [1, 2])

    def jsonl(self, relative, rows):
        path = self.run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def benchmark_record(self, case, scores, variant=None, step=None):
        point = ANALYSIS.benchmark_point(self.run, self.study, case, variant, step)
        path = Path(point["output"])
        ANALYSIS.dump(path / "benchmark.json", {"state": "complete", "point": point})
        ANALYSIS.dump(
            path / "results.json",
            {
                "groups": {
                    name: {"acc,none": score}
                    for name, score in zip(ANALYSIS.TASKS, scores)
                }
            },
        )

    def frozen_members(self):
        members = {task: [f"{task}_a", f"{task}_b"] for task in ANALYSIS.TASKS}
        folder = self.run / "source/lm_eval_tasks"
        folder.mkdir(parents=True, exist_ok=True)
        for group, leaves in members.items():
            (folder / f"{group}.yaml").write_text(
                f"group: {group}\ntask:\n" + "".join(f"  - {leaf}\n" for leaf in leaves)
            )
        return members

    def scoped_record(
        self, case, tasks, scores, variant=None, step=None, identity="same-weights"
    ):
        tasks = ANALYSIS.requested_tasks(tasks)
        members = self.frozen_members()
        selected = {task: members[task] for task in tasks}
        point = ANALYSIS.benchmark_point(self.run, self.study, case, variant, step)
        folder = ANALYSIS.scope_output(point, tasks)
        ANALYSIS.dump(
            folder / "benchmark.json",
            {
                "state": "complete",
                "point": point,
                "requested_tasks": list(tasks),
                "completed_tasks": list(tasks),
            },
        )
        ANALYSIS.dump(
            folder / "config.json",
            {
                "tasks": list(tasks),
                "expected_group_subtasks": selected,
                "dtype": "float32",
                "num_fewshot": 0,
            },
        )
        ANALYSIS.dump(
            folder / "provenance.json",
            {
                "checkpoint": point["checkpoint"],
                "checkpoint_files": {"weights": {"sha256": identity}},
                "frozen_task_files": {"groups": "unchanged"},
            },
        )
        raw = {
            "config": {"limit": None},
            "groups": {task: {"acc,none": value} for task, value in zip(tasks, scores)},
            "group_subtasks": selected,
            "results": {},
            "n-samples": {},
        }
        for task, value in zip(tasks, scores):
            for leaf in selected[task]:
                raw["results"][leaf] = {"acc,none": value}
                raw["n-samples"][leaf] = {"original": 4, "effective": 4}
        ANALYSIS.dump(folder / "results.json", raw)
        return folder

    def test_task_scopes_keep_raw_results_separate_and_optional_mmlu_unplanned(self):
        wmdp = self.scoped_record("control", [ANALYSIS.TASKS[0]], [0.25], "full-same")
        before = (wmdp / "results.json").read_bytes()
        summary = ANALYSIS.collect_report(self.run, self.study)
        row = next(
            row
            for row in summary["benchmarks"]
            if row["point"]["relative"] == "control/full-same/step-4"
        )
        self.assertEqual(row["state"], "complete")
        self.assertEqual(row["planned_tasks"], [ANALYSIS.TASKS[0]])
        self.assertEqual(row["scores_percent"], {ANALYSIS.TASKS[0]: 25})
        self.assertEqual(row["task_states"][ANALYSIS.TASKS[1]], "not planned")
        self.assertIsNone(row["original_gate_pass"])
        self.assertIn("not planned", ANALYSIS.markdown_report(summary, []))
        mmlu = self.scoped_record("control", [ANALYSIS.TASKS[1]], [0.54], "full-same")
        self.assertNotEqual(wmdp, mmlu)
        row = next(
            row
            for row in ANALYSIS.collect_report(self.run, self.study)["benchmarks"]
            if row["point"]["relative"] == "control/full-same/step-4"
        )
        self.assertEqual(
            row["scores_percent"], {ANALYSIS.TASKS[0]: 25, ANALYSIS.TASKS[1]: 54}
        )
        self.assertEqual(len(row["measurements"]), 2)
        self.assertEqual(row["task_states"][ANALYSIS.TASKS[1]], "complete")
        self.assertEqual((wmdp / "results.json").read_bytes(), before)

    def test_amendment_overrides_study_benchmark_plan(self):
        point = ANALYSIS.benchmark_point(self.run, self.study, "control", "full-same")
        self.study["benchmark_scope"] = {"post_attack_tasks": list(ANALYSIS.TASKS)}
        self.assertEqual(
            ANALYSIS.planned_tasks(self.run, self.study, point), ANALYSIS.TASKS
        )
        self.save(
            "protocol-amendments.json",
            {
                "benchmark_scope": {
                    "pre_attack_tasks": list(ANALYSIS.TASKS),
                    "post_attack_tasks": [ANALYSIS.TASKS[0]],
                }
            },
        )
        self.assertEqual(
            ANALYSIS.planned_tasks(self.run, self.study, point), (ANALYSIS.TASKS[0],)
        )

    def test_disjoint_pre_attack_scopes_complete_only_after_both(self):
        self.scoped_record("base", [ANALYSIS.TASKS[0]], [0.3])
        row = next(
            row
            for row in ANALYSIS.collect_report(self.run, self.study)["benchmarks"]
            if row["point"]["case"] == "base"
        )
        self.assertEqual(row["state"], "partial")
        self.assertIsNone(row["native_gate_pass"])
        self.scoped_record("base", [ANALYSIS.TASKS[1]], [0.60])
        summary = ANALYSIS.collect_report(self.run, self.study)
        self.assertEqual(summary["utility"]["native_fp32_gate_percent"], 55)
        self.assertEqual(
            next(
                row for row in summary["benchmarks"] if row["point"]["case"] == "base"
            )["state"],
            "complete",
        )

    def test_scope_merge_rejects_checkpoint_config_and_duplicate_score_conflicts(self):
        for conflict in ("checkpoint", "config", "score"):
            with self.subTest(conflict=conflict):
                folder = self.scoped_record(
                    "control", ANALYSIS.TASKS, [0.25, 0.54], "full-same"
                )
                other = self.scoped_record(
                    "control",
                    [ANALYSIS.TASKS[0]],
                    [0.26 if conflict == "score" else 0.25],
                    "full-same",
                    identity="changed" if conflict == "checkpoint" else "same-weights",
                )
                if conflict == "config":
                    config = ANALYSIS.read_json(other / "config.json")
                    config["dtype"] = "bfloat16"
                    ANALYSIS.dump(other / "config.json", config)
                summary = ANALYSIS.collect_report(self.run, self.study)
                row = next(
                    row
                    for row in summary["benchmarks"]
                    if row["point"]["relative"] == "control/full-same/step-4"
                )
                self.assertEqual(row["state"], "inconsistent")
                self.assertIsNone(row["scores_percent"])
                self.assertTrue(summary["warnings"])

    def test_scoped_validation_rejects_missing_leaf_counts_truncation_and_wrong_aggregate(
        self,
    ):
        folder = self.scoped_record("control", [ANALYSIS.TASKS[0]], [0.25], "full-same")
        valid = ANALYSIS.read_json(folder / "results.json")
        members = self.frozen_members()
        expected = {ANALYSIS.TASKS[0]: members[ANALYSIS.TASKS[0]]}
        leaf = expected[ANALYSIS.TASKS[0]][0]
        for corruption in (
            "missing member",
            "missing samples",
            "truncated",
            "wrong aggregate",
            "limited",
        ):
            with self.subTest(corruption=corruption):
                raw = json.loads(json.dumps(valid))
                if corruption == "missing member":
                    raw["group_subtasks"][ANALYSIS.TASKS[0]].pop()
                elif corruption == "missing samples":
                    del raw["n-samples"][leaf]
                elif corruption == "truncated":
                    raw["n-samples"][leaf]["effective"] = 3
                elif corruption == "wrong aggregate":
                    raw["groups"][ANALYSIS.TASKS[0]]["acc,none"] = 0.75
                else:
                    raw["config"]["limit"] = 2
                with self.assertRaises(ValueError):
                    ANALYSIS.group_scores(
                        raw, [ANALYSIS.TASKS[0]], expected_subtasks=expected
                    )
                ANALYSIS.dump(folder / "results.json", raw)
                row = next(
                    row
                    for row in ANALYSIS.collect_report(self.run, self.study)[
                        "benchmarks"
                    ]
                    if row["point"]["relative"] == "control/full-same/step-4"
                )
                self.assertEqual(row["state"], "inconsistent")
                self.assertIsNone(row["scores_percent"])

    def test_exact_checkpoint_selection_without_fallback(self):
        point = ANALYSIS.benchmark_point(
            self.run, self.study, "full-ce-worst", "mini-same", 3
        )
        self.assertEqual(
            Path(point["checkpoint"]),
            self.run / "cases/full-ce-worst/attacks/mini-same/checkpoint-3",
        )
        point = ANALYSIS.benchmark_point(
            self.run, self.study, "full-ce-worst", "mini-same", 0
        )
        self.assertEqual(
            Path(point["checkpoint"]), self.run / "cases/full-ce-worst/defense-2"
        )
        point = ANALYSIS.benchmark_point(
            self.run, self.study, "control", "full-same", 0
        )
        self.assertEqual(Path(point["checkpoint"]), self.run / "anchor")
        with self.assertRaises(ValueError):
            ANALYSIS.benchmark_point(self.run, self.study, "base", "full-same", 1)

    def test_missing_benchmarks_remain_unmeasured_and_uncalibrated(self):
        summary = ANALYSIS.collect_report(self.run, self.study)
        self.assertIsNone(summary["utility"]["native_fp32_gate_percent"])
        for record in summary["benchmarks"]:
            self.assertEqual(record["state"], "not run")
            self.assertIsNone(record["scores_percent"])
            self.assertIsNone(record["original_gate_pass"])
        text = ANALYSIS.markdown_report(summary, [])
        self.assertIn("not run", text)
        self.assertNotIn("0.0000", text)

    def test_both_utility_gates_and_raw_group_results_are_used(self):
        self.benchmark_record("base", (0.6, 0.60))
        self.benchmark_record("full-ce-worst", (0.25, 0.54))
        summary = ANALYSIS.collect_report(self.run, self.study)
        self.assertEqual(summary["utility"]["native_fp32_gate_percent"], 55)
        row = next(
            r
            for r in summary["benchmarks"]
            if r["point"]["case"] == "full-ce-worst" and r["point"]["variant"] is None
        )
        self.assertTrue(row["original_gate_pass"])
        self.assertFalse(row["native_gate_pass"])
        self.assertEqual(row["scores_percent"]["wmdp_bio_robust"], 25)
        self.assertEqual(row["native_gate_margin_pp"], -1)

    def test_partial_attack_probe_is_labeled_with_its_actual_step(self):
        rows = [
            {
                "step": 0,
                "tokens_processed": 0,
                "probes": {"forget_heldout": {"ce": 10.0}},
            },
            {"step": 1, "tokens_processed": 2040},
        ]
        self.jsonl("cases/control/attacks/full-same/trace.jsonl", rows)
        summary = ANALYSIS.collect_report(self.run, self.study)
        row = next(
            r
            for r in summary["attacks"]
            if r["case"] == "control" and r["variant"] == "full-same"
        )
        self.assertEqual(row["state"], "partial")
        self.assertEqual(row["last_attack_step"], 1)
        self.assertEqual(row["latest_probe_step"], 0)
        self.assertIsNone(row["matched"])
        self.save("cases/control/attacks/full-same/attack.json", {"state": "complete"})
        row = ANALYSIS.collect_report(self.run, self.study)["attacks"][0]
        self.assertEqual(row["state"], "inconsistent")

    def test_fidelity_is_visible_and_separate_from_attack_completion(self):
        rows = [{"step": 0, "tokens_processed": 0}]
        for step, error in enumerate((0.001, 0.003, 0.009, 0.002), 1):
            rows.append(
                {
                    "step": step,
                    "tokens_processed": step * 2040,
                    "simulator_update_relative_l2_error": error,
                }
            )
        rows[-1]["simulator_endpoint_probe"] = {"ce": 10.0005}
        rows[-1]["probes"] = {"forget_train": {"ce": 10.0}}
        folder = "cases/full-ce-worst/attacks/full-same"
        self.jsonl(f"{folder}/trace.jsonl", rows)
        self.save(f"{folder}/attack.json", {"state": "complete"})
        summary = ANALYSIS.collect_report(self.run, self.study)
        record = next(
            row
            for row in summary["attacks"]
            if row["case"] == "full-ce-worst" and row["variant"] == "full-same"
        )
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["fidelity_state"], "verified")
        self.assertTrue(record["matched_fidelity_verified"])
        self.assertEqual(record["simulator_max_4step_relative_l2_error"], 0.009)
        self.assertAlmostEqual(record["simulator_endpoint_ce_gap"], 0.0005)
        rows[-1]["simulator_endpoint_probe"]["ce"] = 10.002
        self.jsonl(f"{folder}/trace.jsonl", rows)
        summary = ANALYSIS.collect_report(self.run, self.study)
        record = next(
            row
            for row in summary["attacks"]
            if row["case"] == "full-ce-worst" and row["variant"] == "full-same"
        )
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["fidelity_state"], "outside tolerance")
        self.assertFalse(record["matched_fidelity_verified"])
        self.assertIn("outside tolerance", ANALYSIS.markdown_report(summary, []))
        self.save("study.json", self.study)
        self.save("source-manifest.json", {})
        helper = SimpleNamespace(__file__=str(SCRIPT.with_name("tar_cleanroom.py")))
        ANALYSIS.report(self.run, self.study, SimpleNamespace(no_plots=True), helper)
        with (self.run / "report/attack-fidelity.csv").open() as handle:
            csv_rows = list(csv.DictReader(handle))
        row = next(
            row
            for row in csv_rows
            if row["case"] == "full-ce-worst" and row["variant"] == "full-same"
        )
        self.assertEqual(row["matched_fidelity_verified"], "False")
        self.assertEqual(row["state"], "complete")

    def test_missing_partial_and_excessive_relative_error_do_not_verify_matching(self):
        self.assertIsNone(
            ANALYSIS.attack_fidelity([], True)["matched_fidelity_verified"]
        )
        partial = [{"step": 1, "simulator_update_relative_l2_error": 0.0}]
        self.assertEqual(
            ANALYSIS.attack_fidelity(partial, True)["fidelity_state"], "incomplete"
        )
        rows = [
            {
                "step": step,
                "simulator_update_relative_l2_error": 0.0101 if step == 2 else 0.0,
            }
            for step in range(1, 5)
        ]
        rows[-1].update(
            simulator_endpoint_probe={"ce": 10.0}, probes={"forget_train": {"ce": 10.0}}
        )
        result = ANALYSIS.attack_fidelity(rows, True)
        self.assertEqual(result["fidelity_state"], "outside tolerance")
        self.assertFalse(result["matched_fidelity_verified"])

    def test_worst_floor_is_measured_over_entire_path(self):
        row = {
            "outer_step": 1,
            "ce_floor": 10.0,
            "selected_inner_step": 2,
            "forget_objective_active": True,
            "simulated_path": [
                {"step": 0, "ce": 9},
                {"step": 1, "ce": 8},
                {"step": 2, "ce": 11},
            ],
        }
        result = ANALYSIS.outer_summary("terminal", row, 400)
        self.assertEqual(result["selected_floor_violation"], 0)
        self.assertEqual(result["worst_floor_violation"], 2)
        self.assertEqual(result["inner_tokens_processed"], 400)

    def test_invalid_or_missing_group_cannot_be_promoted(self):
        point = ANALYSIS.benchmark_point(self.run, self.study, "base")
        target = Path(point["output"])
        ANALYSIS.dump(target / "benchmark.json", {"state": "complete", "point": point})
        ANALYSIS.dump(
            target / "results.json", {"results": {"mmlu_no_bio": {"acc,none": 0.60}}}
        )
        summary = ANALYSIS.collect_report(self.run, self.study)
        base = next(
            row for row in summary["benchmarks"] if row["point"]["case"] == "base"
        )
        self.assertEqual(base["state"], "inconsistent")
        self.assertIsNone(base["scores_percent"])
        self.assertIsNone(summary["utility"]["native_fp32_gate_percent"])
        with self.assertRaises(ValueError):
            ANALYSIS.group_scores(
                {
                    "results": {
                        task: {"acc,none": float("nan")} for task in ANALYSIS.TASKS
                    }
                }
            )

    def test_benchmark_uses_native_model_and_full_frozen_groups(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires torch")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self):
                return SimpleNamespace(logits=self.weight * 2)

        model = TinyModel()
        for directory in (
            "anchor",
            "source/lm_eval_tasks",
            "original/source/lm_eval_tasks",
        ):
            (self.run / directory).mkdir(parents=True)
        self.save("anchor/config.json", {})
        self.save("study.json", self.study)
        self.save("source-manifest.json", {})
        for directory in ("source", "original/source"):
            (self.run / directory / "lm_eval_tasks/task.yaml").write_text(
                "group: frozen\n"
            )
        members = self.frozen_members()
        for group in members:
            (self.run / "original/source/lm_eval_tasks" / f"{group}.yaml").write_text(
                (self.run / "source/lm_eval_tasks" / f"{group}.yaml").read_text()
            )
        helper = SimpleNamespace(__file__=str(SCRIPT.with_name("tar_cleanroom.py")))
        calls = {}

        def runtime(study, checkpoint):
            calls["checkpoint"] = checkpoint
            return model, object()

        helper.runtime = runtime

        def wrapper(**kwargs):
            calls["wrapper"] = kwargs
            return SimpleNamespace(
                model=kwargs["pretrained"],
                mixed_precision_dtype=kwargs["mixed_precision_dtype"],
            )

        def evaluate(**kwargs):
            calls["evaluate"] = kwargs
            kwargs["model"].model()
            return {
                "groups": {task: {"acc,none": 0.5} for task in kwargs["tasks"]},
                "group_subtasks": {task: members[task] for task in kwargs["tasks"]},
                "results": {
                    leaf: {"acc,none": 0.5}
                    for task in kwargs["tasks"]
                    for leaf in members[task]
                },
                "n-samples": {
                    leaf: {"original": 4, "effective": 4}
                    for task in kwargs["tasks"]
                    for leaf in members[task]
                },
            }

        modules = {
            "lm_eval": SimpleNamespace(simple_evaluate=evaluate),
            "lm_eval.models.huggingface": SimpleNamespace(HFLM=wrapper),
            "lm_eval.tasks": SimpleNamespace(TaskManager=lambda **kwargs: kwargs),
            "lm_eval.utils": SimpleNamespace(handle_non_serializable=str),
        }
        args = SimpleNamespace(case="control", variant=None, step=None)
        with (
            patch.dict("sys.modules", modules),
            patch.object(ANALYSIS.importlib.metadata, "version", return_value="test"),
        ):
            ANALYSIS.benchmark(self.run, self.study, args, helper)
        self.assertIs(calls["wrapper"]["pretrained"], model)
        self.assertIsNone(calls["wrapper"]["mixed_precision_dtype"])
        self.assertEqual(calls["wrapper"]["batch_size"], 32)
        self.assertEqual(calls["evaluate"]["tasks"], list(ANALYSIS.TASKS))
        self.assertIsNone(calls["evaluate"]["limit"])
        result = ANALYSIS.read_json(
            self.run / "benchmarks/control/anchor/benchmark.json"
        )
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["requested_tasks"], list(ANALYSIS.TASKS))
        self.assertEqual(result["completed_tasks"], list(ANALYSIS.TASKS))
        provenance = ANALYSIS.read_json(
            self.run / "benchmarks/control/anchor/provenance.json"
        )
        self.assertEqual(provenance["observed_logits_dtypes"], ["torch.float32"])
        self.assertIn("config.json", provenance["checkpoint_files"])
        self.assertTrue((Path(provenance["source"]["archive"]) / SCRIPT.name).is_file())
        self.assertFalse((self.run / "source" / SCRIPT.name).exists())
        original_results = (
            self.run / "benchmarks/control/anchor/results.json"
        ).read_bytes()
        args.tasks = [ANALYSIS.TASKS[0]]
        with (
            patch.dict("sys.modules", modules),
            patch.object(ANALYSIS.importlib.metadata, "version", return_value="test"),
        ):
            ANALYSIS.benchmark(self.run, self.study, args, helper)
        self.assertEqual(calls["evaluate"]["tasks"], [ANALYSIS.TASKS[0]])
        scoped = self.run / "benchmarks/control/anchor" / f"tasks-{ANALYSIS.TASKS[0]}"
        status = ANALYSIS.read_json(scoped / "benchmark.json")
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["scores_percent"], {ANALYSIS.TASKS[0]: 50})
        self.assertEqual(status["completed_tasks"], [ANALYSIS.TASKS[0]])
        self.assertEqual(
            (self.run / "benchmarks/control/anchor/results.json").read_bytes(),
            original_results,
        )


if __name__ == "__main__":
    unittest.main()
