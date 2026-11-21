"""Exercise offset, transfer and parameter-cast preservation on tiny tensors."""

import os
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from runtime import BASELINE, PEER


BASELINE_OPS = runpy.run_path(str(BASELINE))
PEER_OPS = runpy.run_path(str(PEER))


class LandscapeStorageTest(unittest.TestCase):
    def test_default_keeps_original_anchor_setting(self):
        with patch.dict(os.environ, {"TAR_LANDSCAPE_PEER": "0"}):
            self.assertEqual(
                PEER_OPS["_landscape_storage"]({"cpu_landscape_anchor": True}),
                ("cuda:0", True),
            )
            self.assertEqual(PEER_OPS["_landscape_storage"]({}), ("cuda:0", False))

    def check_parameters(self, parameter_device, direction_device):
        parameters = {
            "a": torch.nn.Parameter(
                torch.tensor([[1.01, -0.71], [0.1, 0.2]], dtype=torch.bfloat16).to(
                    parameter_device
                )
            ),
            "b": torch.nn.Parameter(
                torch.tensor([-0.3, 0.7, 0.0], device=parameter_device)
            ),
        }
        offsets = {"a": 3, "b": 0}
        directions = np.linspace(-0.4, 0.8, 14, dtype=np.float32).reshape(2, 7)
        host_anchor = BASELINE_OPS["_copy_landscape_anchor"](parameters, cpu=True)
        resident_anchor = PEER_OPS["_copy_landscape_anchor"](parameters)
        baseline = BASELINE_OPS["_basis_to_device"](directions, parameter_device, 2)
        peer = PEER_OPS["_basis_to_device"](directions, direction_device, 2)
        for c0, c1 in [(0.0, 0.0), (0.15, -0.25), (-125.0, 125.0)]:
            BASELINE_OPS["_perturb_landscape"](
                parameters, host_anchor, baseline, offsets, c0, c1
            )
            expected = {
                name: tensor.detach().clone() for name, tensor in parameters.items()
            }
            PEER_OPS["_perturb_landscape"](
                parameters, resident_anchor, peer, offsets, c0, c1
            )
            for name, parameter in parameters.items():
                self.assertTrue(
                    torch.equal(
                        parameter.detach().view(torch.uint8),
                        expected[name].view(torch.uint8),
                    ),
                    msg=name,
                )

    def test_same_device_arithmetic(self):
        self.check_parameters("cpu", "cpu")

    @unittest.skipUnless(
        os.environ.get("TAR_TEST_TWO_CUDA_GPUS") == "1", "needs two allocated CUDA GPUs"
    )
    def test_peer_device_arithmetic(self):
        self.assertEqual(torch.cuda.device_count(), 2)
        self.check_parameters("cuda:0", "cuda:1")


if __name__ == "__main__":
    unittest.main()
