"""Metadata-only tests for endpoint-scoped factorial training authorization."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import training_scope_guard  # noqa: E402


RUN = Path(
    "/p/vast1/kirchenb/tar-root/lilpipe/examples/tamper-resistance/artifacts/tuolumne/tar-hparams-factorial/20261117T0130Z"
)


class TrainingScopeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program = json.loads((RUN / "study.json").read_text())
        cls.plan = json.loads((RUN / "execution-plan.json").read_text())

    def test_real_study_without_evaluation_cases_accepts_only_three_defenses(self):
        self.assertNotIn("evaluation_cases", self.program)
        for case in ("hparam-ri-defense", "hparam-ro-defense", "hparam-io-defense"):
            result = training_scope_guard.validate_program(
                self.program, self.plan, RUN, case
            )
            self.assertEqual(result["state"], "passed")
            self.assertEqual(result["role"], "defense")
            self.assertEqual(result["steps"], 125)

    def test_exact_training_wrapper_cli_accepts_all_three_real_cases(self):
        guard = HERE / "training_scope_guard.py"
        for case in ("hparam-ri-defense", "hparam-ro-defense", "hparam-io-defense"):
            with self.subTest(case=case):
                result = subprocess.run(
                    [sys.executable, str(guard), "training", "case", str(RUN), case],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('"state": "passed"', result.stdout)

    def test_generated_attack_configs_are_not_authorized(self):
        for case in ("hparam-ri-lr1e-6", "hparam-ro-lr1e-6", "hparam-io-lr1e-6"):
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, "outside"):
                training_scope_guard.validate_program(
                    self.program, self.plan, RUN, case
                )

    def test_unknown_defense_is_not_authorized(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            training_scope_guard.validate_program(
                self.program, self.plan, RUN, "hparam-rio-defense"
            )

    def test_modified_case_role_or_horizon_is_rejected(self):
        for field, value in (("role", "attack"), ("max_steps", 1000)):
            with self.subTest(field=field):
                program = json.loads(json.dumps(self.program))
                program["cases"]["hparam-ri-defense"][field] = value
                with self.assertRaisesRegex(ValueError, "differs from planned"):
                    training_scope_guard.validate_program(
                        program, self.plan, RUN, "hparam-ri-defense"
                    )


if __name__ == "__main__":
    unittest.main()
