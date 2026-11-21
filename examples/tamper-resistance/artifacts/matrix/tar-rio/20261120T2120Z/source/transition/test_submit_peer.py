"""Small, local tests for peer submit command construction."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name("submit_peer.py")
SPEC = importlib.util.spec_from_file_location("submit_peer", MODULE_PATH)
submit_peer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit_peer)


class SubmitPeerTests(unittest.TestCase):
    def test_validate_case_requires_case_local_reference_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            folder = run / "cases" / "rio-k16"
            for name in (
                "status",
                "logs",
                "cache",
                "models",
                "geometry",
                "surfaces",
                "source/misc",
            ):
                (folder / name).mkdir(parents=True, exist_ok=True)
            (folder / "study.json").write_text("{}\n")
            (folder / "case.json").write_text("{}\n")
            (folder / "status/train.json").write_text('{"state":"complete"}\n')

            with self.assertRaisesRegex(ValueError, "case-local analysis helper"):
                submit_peer.validate_case(run, "rio-k16")

            helper = folder / "source/misc/reference_get_ll.py"
            helper.write_text("# prepared helper\n")
            self.assertEqual(submit_peer.validate_case(run, "rio-k16"), folder)

    def test_command_has_fixed_matrix_resources_and_peer_target(self):
        run = Path("/tmp/matrix-run")
        pilot = Path("/tmp/pilot/result.json")
        command = submit_peer.build_command(
            run, "rio-k16", pilot, ["510001", "510002"], "afterany", 360, "retry-a"
        )
        self.assertIn("--gpus_per_node=2", command)
        self.assertIn("--gpus_per_task=2", command)
        self.assertIn("--cpus_per_task=56", command)
        self.assertIn("--cuda_module=", command)
        self.assertIn("--dependency=afterany:510001:510002", command)
        self.assertIn("--run_name=tar-rio-analysis-rio-k16-retry-a", command)
        self.assertIn("--sub_output_dir_name=analysis-rio-k16-retry-a", command)
        self.assertEqual(command[-7], "--")
        self.assertEqual(
            command[-6:],
            [
                "bash",
                "-i",
                str(submit_peer.ROOT / "source/peer-landscape/analysis.sh"),
                str(run),
                "rio-k16",
                str(pilot),
            ],
        )

    def test_no_dependency_is_omitted(self):
        command = submit_peer.build_command(
            Path("/tmp/run"), "rio-k16", Path("/tmp/pilot.json"), [], "afterok", 360
        )
        self.assertFalse(any(item.startswith("--dependency=") for item in command))
        self.assertIn("--run_name=tar-rio-analysis-rio-k16-peer-v2", command)
        self.assertIn("--sub_output_dir_name=analysis-rio-k16-peer-v2", command)

    def test_dryrun_flag_precedes_target_and_does_not_mutate_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            case = "rio-k16"
            folder = run / "cases" / case
            for name in ("status", "logs", "cache", "models", "geometry", "surfaces"):
                (folder / name).mkdir(parents=True, exist_ok=True)
            (folder / "study.json").write_text("{}\n")
            (folder / "case.json").write_text("{}\n")
            (folder / "source/misc").mkdir(parents=True)
            (folder / "source/misc/reference_get_ll.py").write_text("# helper\n")
            (folder / "status" / "train.json").write_text('{"state":"complete"}\n')
            pilot = root / "pilot.json"
            pilot.write_text('{"state":"complete","gates":{"accuracy":true}}\n')
            (root / "logs").mkdir()
            ledger = root / "matrix-jobs.json"
            ledger.write_text('{"existing:key":{"job_id":"510000"}}\n')
            original_ledger = ledger.read_bytes()
            fake_result = mock.Mock(stdout="dryrun command\n")

            with (
                mock.patch.object(submit_peer, "ROOT", root),
                mock.patch.object(
                    submit_peer.subprocess, "run", return_value=fake_result
                ) as run_mock,
                mock.patch(
                    "sys.argv",
                    [
                        "submit_peer.py",
                        "--run",
                        str(run),
                        "--case",
                        case,
                        "--pilot",
                        str(pilot),
                        "--dryrun",
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                submit_peer.main()

            command = run_mock.call_args.args[0]
            dryrun_index = command.index("--dryrun")
            delimiter_index = command.index("--")
            self.assertLess(dryrun_index, delimiter_index)
            self.assertEqual(ledger.read_bytes(), original_ledger)
            self.assertFalse((root / "matrix-jobs.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
