"""Regression coverage for strict CUDA parity and non-gating AMD diagnostics."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from precision_audit import analyze_pair, validate_pair_inputs


class MatrixParityGateTests(unittest.TestCase):
    def test_archive_drift_is_reported_but_only_same_backend_parity_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native.safetensors"
            key = "model.layers.0.weight"
            native_weights = {key: torch.tensor([1.0, 2.0], dtype=torch.float32)}
            save_file(native_weights, native)

            parity = root / "parity"
            anchor_dir = parity / "anchor"
            anchor_dir.mkdir(parents=True)
            anchor = {key: native_weights[key].clone()}
            save_file(anchor, anchor_dir / "model.safetensors")
            archive_root = root / "archive"
            checks = []
            for step in (1, 2, 4, 8, 16):
                simulated = {key: anchor[key] + 0.01 * step}
                simulated_dir = parity / f"simulated-{step}"
                simulated_dir.mkdir(parents=True)
                save_file(simulated, simulated_dir / "model.safetensors")

                archived = {key: simulated[key] + 0.001 * step}
                archived_dir = archive_root / f"checkpoint-{step}"
                archived_dir.mkdir(parents=True)
                save_file(archived, archived_dir / "model.safetensors")
                checks.append(
                    {
                        "step": step,
                        "simulated_vs_actual": {
                            "exact_equal": True,
                            "difference_l2": 0.0,
                        },
                        "archived_vs_actual": {
                            "exact_equal": False,
                            "difference_l2": 0.001 * step,
                            "relative_update_error": 0.1,
                        },
                    }
                )

            validation = root / "validation.json"
            payload = {
                "state": "complete",
                "trajectory_parity": "exact",
                "reference_root": str(archive_root),
                "actual_loss_errors": [0.0] * 16,
                "parameter_checks": checks,
            }
            validation.write_text(json.dumps(payload))

            rows = analyze_pair(
                "matrix-fixture",
                native,
                parity,
                validation,
                torch.device("cpu"),
                2,
                anchor_backend="fresh-cuda",
                simulated_backend="cuda",
                reference_backend="amd",
            )
            archived_row = next(
                row
                for row in rows
                if row["measure"] == "reference_vs_simulated" and row["step"] == 1
            )
            self.assertAlmostEqual(archived_row["difference_l2"], 0.001 * 2**0.5)
            self.assertEqual(archived_row["comparison_backend"], "cuda-vs-amd")
            validate_pair_inputs(native, parity, validation)

            payload["parameter_checks"][0]["simulated_vs_actual"] = {
                "exact_equal": False,
                "difference_l2": 1e-8,
            }
            validation.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "simulated_vs_actual state 1"):
                validate_pair_inputs(native, parity, validation)


if __name__ == "__main__":
    unittest.main()
