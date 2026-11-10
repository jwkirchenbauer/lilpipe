"""Full-task verification, checkpoint selection, resume, and measured maxima."""

import copy
import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MISC = ROOT / "examples/tamper-resistance/scripts/misc"
SPEC = importlib.util.spec_from_file_location(
    "tar_compute_evaluation", MISC / "tar_compute_evaluation.py"
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
GROUPS = {
    "wmdp_bio_robust": ("wmdp_a", "wmdp_b"),
    "mmlu_no_bio": ("mmlu_a", "mmlu_b"),
}


def result(model, tasks=EVAL.TASKS, score=0.4):
    data = {
        "config": {
            "model": "hf",
            "model_args": {"pretrained": str(model), "dtype": "bfloat16"},
            "batch_size": "32",
            "limit": None,
        },
        "configs": {},
        "results": {},
        "group_subtasks": {},
        "n-samples": {},
        "n-shot": {},
    }
    for task in tasks:
        data["group_subtasks"][task] = list(GROUPS[task])
        data["results"][task] = {"acc,none": score}
        for child in GROUPS[task]:
            data["configs"][child] = {"num_fewshot": 0}
            data["n-shot"][child] = 0
            data["results"][child] = {"acc,none": score}
            data["n-samples"][child] = {"original": 10, "effective": 10}
    return data


def write_result(output, data, filename="results_test.json"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / filename).write_text(json.dumps(data))


def fixture_study(run):
    source = run / "source/lm_eval_tasks"
    source.mkdir(parents=True)
    for task, children in GROUPS.items():
        (source / f"{task}.yaml").write_text(
            f"group: {task}\ntask:\n"
            + "".join(f"  - {child}\n" for child in children)
            + "aggregate_metric_list:\n  - metric: acc\n"
        )
    cases = {}
    for name, role, steps, sigma in (
        ("baseline-lr1e-5", "attack", [128, 512, 1000], None),
        ("noise-1e-3", "attack", [128, 512, 1000], 0.001),
        ("defense-125x64", "defense", [32, 64, 125], None),
    ):
        folder = run / "cases" / name
        folder.mkdir(parents=True)
        case = {
            "name": name,
            "role": role,
            "save_steps": steps,
            "max_steps": max(steps),
            "state_steps": steps,
            "gaussian_sigma": sigma,
            "attack_learning_rate": 1e-5 if role == "attack" else None,
            "base_model": str(run / "base"),
        }
        (folder / "case.json").write_text(json.dumps(case))
        cases[name] = case
    study = {"cases": cases, "bundles": {"controls": ["baseline-lr1e-5"]}}
    (run / "study.json").write_text(json.dumps(study))
    return study


def write_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_text("test weights")


class VerificationTests(unittest.TestCase):
    def test_requires_unlimited_full_subtasks_zero_shot_bf16(self):
        original = result("/checkpoint")
        verified = EVAL.verified_task(
            original, EVAL.TASKS[0], "/checkpoint", GROUPS[EVAL.TASKS[0]]
        )
        self.assertEqual(verified["jwk_percent"], 40)
        mutations = (
            lambda data: data["config"].update(limit=10),
            lambda data: data["config"].pop("limit"),
            lambda data: data["config"].update(samples={"wmdp_a": [0]}),
            lambda data: data["config"].update(batch_size=1),
            lambda data: data["config"]["model_args"].update(dtype="float16"),
            lambda data: data["config"]["model_args"].update(pretrained="/other"),
            lambda data: data["configs"]["wmdp_a"].update(num_fewshot=1),
            lambda data: data["configs"]["wmdp_a"].update(num_fewshot=False),
            lambda data: data["n-samples"]["wmdp_a"].update(effective=9),
            lambda data: data["n-samples"].pop("wmdp_a"),
            lambda data: data["results"].pop("wmdp_b"),
            lambda data: data["results"]["wmdp_b"].update({"acc,none": 0.3}),
            lambda data: data["group_subtasks"][EVAL.TASKS[0]].pop(),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = copy.deepcopy(original)
                mutate(data)
                with self.assertRaises(EVAL.InvalidResult):
                    EVAL.verified_task(
                        data, EVAL.TASKS[0], "/checkpoint", GROUPS[EVAL.TASKS[0]]
                    )

    def test_limited_and_incomplete_attempts_do_not_hide_complete_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            limited = result("/checkpoint")
            limited["config"]["limit"] = 0.1
            write_result(output, limited, "results_limited.json")
            (output / "results_truncated.json").write_text('{"results":')
            (output / "results_invalid.json").write_text("[]")
            write_result(
                output, result("/checkpoint", EVAL.TASKS[:1]), "results_full.json"
            )
            found, rejected = EVAL.collect_results(output, "/checkpoint", GROUPS)
            self.assertEqual(set(found), {EVAL.TASKS[0]})
            self.assertEqual(found[EVAL.TASKS[0]]["jwk_percent"], 40)
            self.assertEqual(len(rejected), 4)

    def test_merged_reruns_must_agree_with_existing_wmdp(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_result(
                output, result("/checkpoint", EVAL.TASKS[:1]), "results_wmdp.json"
            )
            write_result(output, result("/checkpoint"), "results_combined.json")
            found, _ = EVAL.collect_results(output, "/checkpoint", GROUPS)
            self.assertEqual(len(found[EVAL.TASKS[0]]["files"]), 2)
            write_result(
                output, result("/checkpoint", score=0.5), "results_combined.json"
            )
            with self.assertRaisesRegex(
                ValueError, "Conflicting complete wmdp_bio_robust"
            ):
                EVAL.collect_results(output, "/checkpoint", GROUPS)


class PlanningTests(unittest.TestCase):
    def test_runtime_accepts_equivalent_environment_symlink_and_rejects_other_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual-env"
            other = root / "different-env"
            alias = root / "environment-alias"
            actual.mkdir()
            other.mkdir()
            alias.symlink_to(actual, target_is_directory=True)
            environment = {
                "CONDA_PREFIX": str(actual),
                "FLUX_JOB_ID": "test-job",
                "HF_HOME": EVAL.HF_HOME,
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "HIP_VISIBLE_DEVICES": "0,1,2,3",
                "TORCH_BLAS_PREFER_CUBLASLT": "0",
                "TORCH_BLAS_PREFER_HIPBLASLT": "0",
                "AMD_COMGR_CACHE": "0",
                "OMP_NUM_THREADS": "8",
            }
            with (
                patch.object(EVAL, "EVAL_ENV", str(alias)),
                patch.dict(os.environ, environment, clear=True),
            ):
                EVAL.require_evaluation_runtime()
                os.environ["CONDA_PREFIX"] = str(alias)
                EVAL.require_evaluation_runtime()
                os.environ["CONDA_PREFIX"] = str(other)
                with self.assertRaisesRegex(
                    ValueError, "requires activated environment"
                ):
                    EVAL.require_evaluation_runtime()
                os.environ["CONDA_PREFIX"] = str(root / "missing")
                with self.assertRaisesRegex(
                    ValueError, "requires activated environment"
                ):
                    EVAL.require_evaluation_runtime()

    def test_default_selection_excludes_smoke_but_explicit_selection_allows_it(self):
        study = {"cases": {"smoke-k64": {}, "baseline-lr1e-5": {}}}
        self.assertEqual(EVAL.select_cases(study), ["baseline-lr1e-5"])
        self.assertEqual(EVAL.select_cases(study, cases=["smoke-k64"]), ["smoke-k64"])

    def test_attack_step_zero_uses_perturbed_model_for_noise_only(self):
        folder = Path("/study/cases/noise")
        case = {"role": "attack", "gaussian_sigma": 0.001}
        self.assertEqual(
            EVAL.checkpoint_model(folder, case, 0),
            folder / "models/attack/checkpoint-0",
        )
        case["gaussian_sigma"] = None
        self.assertEqual(
            EVAL.checkpoint_model(folder, case, 0), folder / "models/defense"
        )
        self.assertEqual(
            EVAL.checkpoint_model(folder, case, 1000),
            folder / "models/attack/checkpoint-1000",
        )

    def test_task_schedule_and_structured_argv(self):
        case = {"role": "attack", "save_steps": [1, 128, 256, 512, 1000]}
        plan = dict(EVAL.evaluation_plan(case))
        self.assertEqual(plan[0], EVAL.TASKS)
        self.assertEqual(plan[256], EVAL.TASKS[:1])
        self.assertEqual(plan[1000], EVAL.TASKS)
        case["role"] = "defense"
        self.assertEqual(
            dict(EVAL.evaluation_plan(case)),
            {step: EVAL.TASKS for step in case["save_steps"]},
        )
        argv = EVAL.evaluation_command(
            "/model with spaces;$(touch nope)", "/out path", "/frozen tasks", EVAL.TASKS
        )
        self.assertIn(
            "pretrained=/model with spaces;$(touch nope),dtype=bfloat16,trust_remote_code=True",
            argv,
        )
        self.assertNotIn("--limit", argv)
        self.assertEqual(argv[argv.index("--num_processes") + 1], "4")

    def test_training_completion_and_all_saved_weights_required(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            case = {"role": "attack", "save_steps": [1000], "max_steps": 1000}
            write_model(folder / "models/attack/checkpoint-1000")
            with self.assertRaisesRegex(ValueError, "Training must be complete"):
                EVAL.require_training_complete(folder, case)
            EVAL.dump(folder / "status/train.json", {"state": "complete"})
            with self.assertRaisesRegex(ValueError, "Missing checkpoint config"):
                EVAL.require_training_complete(folder, case)
            write_model(folder / "models/attack")
            write_model(folder / "models/defense")
            EVAL.require_training_complete(folder, case)
            (folder / "models/attack/checkpoint-1000/model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "Missing checkpoint weights"):
                EVAL.require_training_complete(folder, case)

    def test_peak_uses_observed_earlier_maximum_and_reports_coverage(self):
        case = {"role": "attack", "save_steps": [128, 256, 512, 1000]}
        rows = [
            {"task": EVAL.TASKS[0], "attack_step": step, "jwk_percent": score}
            for step, score in ((0, 33), (128, 70), (512, 55), (1000, 51))
        ]
        rows.append({"task": EVAL.TASKS[1], "attack_step": 128, "jwk_percent": 90})
        peaks = EVAL.observed_peaks(rows, case)
        self.assertEqual([row["max_observed_wmdp_percent"] for row in peaks], [70] * 4)
        self.assertEqual(peaks[-1]["observed_peak_step"], 128)
        self.assertTrue(peaks[0]["complete_through_budget"])
        self.assertFalse(peaks[1]["complete_through_budget"])
        self.assertFalse(peaks[-1]["complete_through_budget"])
        self.assertEqual(EVAL.observed_peaks([], case), [])


class IntegrationTests(unittest.TestCase):
    def test_preflight_failure_records_selected_case_and_preserves_running_status(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            fixture_study(run)
            name = "baseline-lr1e-5"
            folder = run / "cases" / name
            other = run / "cases/noise-1e-3"
            EVAL.dump(
                other / "status/evaluation.json",
                {"state": "running", "owner": "other-job"},
            )
            with (
                patch.object(
                    EVAL,
                    "require_evaluation_runtime",
                    side_effect=ValueError("wrong environment"),
                ),
                patch.object(EVAL.subprocess, "run") as execute,
            ):
                with self.assertRaisesRegex(ValueError, "wrong environment"):
                    EVAL.evaluate(run, cases=[name])
            execute.assert_not_called()
            failed = EVAL.read_json(folder / "status/evaluation.json")
            self.assertEqual(failed["state"], "failed")
            self.assertEqual(failed["stage"], "preflight")
            self.assertEqual(
                EVAL.read_json(other / "status/evaluation.json"),
                {"state": "running", "owner": "other-job"},
            )
            EVAL.dump(
                folder / "status/evaluation.json",
                {"state": "running", "owner": "same-case-existing-job"},
            )
            with patch.object(
                EVAL,
                "require_evaluation_runtime",
                side_effect=ValueError("wrong environment"),
            ):
                with self.assertRaisesRegex(ValueError, "wrong environment"):
                    EVAL.evaluate(run, cases=[name])
            self.assertEqual(
                EVAL.read_json(folder / "status/evaluation.json"),
                {"state": "running", "owner": "same-case-existing-job"},
            )
            attempts = list((folder / "status").glob("evaluation-preflight-*.json"))
            self.assertEqual(len(attempts), 2)
            self.assertTrue(
                all(
                    EVAL.read_json(attempt)["state"] == "failed" for attempt in attempts
                )
            )

    def test_csv_publication_is_atomic_and_preserves_old_csv_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scores.csv"
            output.write_text("previous complete report\n")

            def failed_rows():
                yield {"score": 40}
                self.assertEqual(output.read_text(), "previous complete report\n")
                raise ValueError("interrupted publication")

            with self.assertRaisesRegex(ValueError, "interrupted publication"):
                EVAL.write_csv(output, ["score"], failed_rows())
            self.assertEqual(output.read_text(), "previous complete report\n")
            self.assertEqual(list(Path(directory).iterdir()), [output])

            def complete_rows():
                yield {"score": 40}
                self.assertEqual(output.read_text(), "previous complete report\n")
                yield {"score": 55}
                self.assertEqual(output.read_text(), "previous complete report\n")

            EVAL.write_csv(output, ["score"], complete_rows())
            self.assertEqual(output.read_text().splitlines(), ["score", "40", "55"])
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_report_lock_covers_reads_and_publication_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            fixture_study(run)
            lock_path = run / "status/evaluation-report.lock"
            original_load = EVAL.load_study
            original_write = EVAL.write_csv

            def assert_locked():
                with lock_path.open("a") as competing:
                    with self.assertRaises(BlockingIOError):
                        EVAL.fcntl.flock(
                            competing.fileno(), EVAL.fcntl.LOCK_EX | EVAL.fcntl.LOCK_NB
                        )

            def checked_load(*args):
                assert_locked()
                return original_load(*args)

            def checked_write(*args):
                assert_locked()
                return original_write(*args)

            with (
                patch.object(EVAL, "load_study", side_effect=checked_load),
                patch.object(EVAL, "write_csv", side_effect=checked_write),
            ):
                EVAL.report(run)
            with patch.object(
                EVAL, "load_study", side_effect=ValueError("invalid study")
            ):
                with self.assertRaisesRegex(ValueError, "invalid study"):
                    EVAL.report(run)
            with lock_path.open("a") as competing:
                EVAL.fcntl.flock(
                    competing.fileno(), EVAL.fcntl.LOCK_EX | EVAL.fcntl.LOCK_NB
                )
                EVAL.fcntl.flock(competing.fileno(), EVAL.fcntl.LOCK_UN)
            EVAL.report(run)

    def test_report_writes_only_measured_values_and_exploratory_utility(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            study = fixture_study(run)
            name = "baseline-lr1e-5"
            folder = run / "cases" / name
            case = study["cases"][name]
            write_result(
                folder / "evals/checkpoint-0",
                result(EVAL.checkpoint_model(folder, case, 0), EVAL.TASKS[:1]),
            )
            defense = run / "cases/defense-125x64"
            model = EVAL.checkpoint_model(defense, study["cases"]["defense-125x64"], 32)
            write_result(defense / "evals/checkpoint-32", result(model, score=0.54))
            all_rows = EVAL.report(run)
            self.assertEqual(len(all_rows), 3)
            with (folder / "evaluation-scores.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(set(rows[0]), set(EVAL.SCORE_FIELDS))
            self.assertEqual(len(rows), 1)
            with (run / "utility-matching.csv").open() as stream:
                utility = list(csv.DictReader(stream))
            self.assertEqual(utility[0]["exploratory_match_within_0_5pp"], "True")
            self.assertAlmostEqual(float(utility[0]["mmlu_no_bio_percent"]), 54)
            coverage = EVAL.read_json(run / "evaluation-verification.json")["coverage"]
            self.assertTrue(all(not row["complete"] for row in coverage))
            existing = (run / "evaluation-scores.csv").read_text()
            write_result(
                folder / "evals/checkpoint-0",
                result(EVAL.checkpoint_model(folder, case, 0), score=0.5),
                "results_conflict.json",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting complete"):
                EVAL.report(run)
            self.assertEqual((run / "evaluation-scores.csv").read_text(), existing)

    def test_resume_launches_only_missing_full_tasks_and_unsets_outer_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            study = fixture_study(run)
            name = "baseline-lr1e-5"
            folder = run / "cases" / name
            case = study["cases"][name]
            write_model(folder / "models/attack")
            for step, _ in EVAL.evaluation_plan(case):
                write_model(EVAL.checkpoint_model(folder, case, step))
            EVAL.dump(folder / "status/train.json", {"state": "complete"})
            model0 = EVAL.checkpoint_model(folder, case, 0)
            write_result(folder / "evals/checkpoint-0", result(model0, EVAL.TASKS[:1]))
            limited = result(model0, EVAL.TASKS[1:])
            limited["config"]["limit"] = 5
            write_result(folder / "evals/checkpoint-0", limited, "results_limited.json")

            def fake_run(argv, **kwargs):
                self.assertTrue(kwargs["check"])
                self.assertNotIn("LOCAL_RANK", kwargs["env"])
                self.assertNotIn("WORLD_SIZE", kwargs["env"])
                model = (
                    argv[argv.index("--model_args") + 1].split(",")[0].split("=", 1)[1]
                )
                tasks = argv[argv.index("--tasks") + 1].split(",")
                output = argv[argv.index("--output_path") + 1]
                write_result(output, result(model, tasks))

            with (
                patch.object(EVAL, "require_evaluation_runtime"),
                patch.dict(os.environ, {"LOCAL_RANK": "0", "WORLD_SIZE": "1"}),
                patch.object(EVAL.subprocess, "run", side_effect=fake_run) as execute,
            ):
                EVAL.evaluate(run, bundle="controls")
            self.assertEqual(execute.call_count, 4)
            first = execute.call_args_list[0].args[0]
            self.assertEqual(first[first.index("--tasks") + 1], "mmlu_no_bio")
            self.assertEqual(
                EVAL.read_json(folder / "status/evaluation.json")["state"], "complete"
            )
            with (
                patch.object(EVAL, "require_evaluation_runtime"),
                patch.object(EVAL.subprocess, "run") as execute,
            ):
                EVAL.evaluate(run, cases=[name])
            execute.assert_not_called()

    def test_case_snapshot_and_frozen_sources_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            study = fixture_study(run)
            self.assertEqual(EVAL.frozen_groups(run, study), GROUPS)
            name = "baseline-lr1e-5"
            study["cases"][name]["save_steps"] = [1000]
            with self.assertRaisesRegex(ValueError, "disagrees with study.json"):
                EVAL.load_case(run, study, name)
            study["source_sha256"] = {
                "source/lm_eval_tasks/mmlu_no_bio.yaml": "incorrect"
            }
            with self.assertRaisesRegex(ValueError, "Frozen evaluation source changed"):
                EVAL.frozen_groups(run, study)

    def test_conflicting_checkpoint_marks_that_checkpoint_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            study = fixture_study(run)
            name = "baseline-lr1e-5"
            folder = run / "cases" / name
            case = study["cases"][name]
            write_model(folder / "models/attack")
            for step, _ in EVAL.evaluation_plan(case):
                model = EVAL.checkpoint_model(folder, case, step)
                write_model(model)
                write_result(folder / "evals" / f"checkpoint-{step}", result(model))
            model = EVAL.checkpoint_model(folder, case, 128)
            write_result(
                folder / "evals/checkpoint-128",
                result(model, score=0.5),
                "results_conflict.json",
            )
            EVAL.dump(folder / "status/train.json", {"state": "complete"})
            with (
                patch.object(EVAL, "require_evaluation_runtime"),
                patch.object(EVAL.subprocess, "run") as execute,
            ):
                with self.assertRaisesRegex(ValueError, "Conflicting complete"):
                    EVAL.evaluate(run, cases=[name])
            execute.assert_not_called()
            self.assertEqual(
                EVAL.read_json(folder / "status/evaluation-checkpoint-0.json")["state"],
                "complete",
            )
            self.assertEqual(
                EVAL.read_json(folder / "status/evaluation-checkpoint-128.json")[
                    "state"
                ],
                "failed",
            )
            self.assertEqual(
                EVAL.read_json(folder / "status/evaluation.json")["state"], "failed"
            )


if __name__ == "__main__":
    unittest.main()
