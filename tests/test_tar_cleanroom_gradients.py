"""Independent optimizer and directional checks for short TAR meta-gradients."""

import copy
import importlib.util
import os
import sys
from pathlib import Path

import unittest

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise unittest.SkipTest("requires PyTorch") from exc

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples/tamper-resistance/scripts/misc/tar_cleanroom_gradients.py"
)
SPEC = importlib.util.spec_from_file_location("tar_cleanroom_gradients", SCRIPT)
gradients = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gradients
SPEC.loader.exec_module(gradients)


CONFIGS = (
    gradients.OptimizerConfig(
        name="sgd", lr=0.08, momentum=0.65, dampening=0.2, weight_decay=0.07
    ),
    gradients.OptimizerConfig(
        name="sgd", lr=0.08, momentum=0.65, nesterov=True, weight_decay=0.07
    ),
    gradients.OptimizerConfig(
        name="adamw", lr=0.08, betas=(0.7, 0.85), eps=0.03, weight_decay=0.07
    ),
)


def make_optimizer(model, config):
    if config.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            dampening=config.dampening,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
            foreach=False,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=False,
    )


class CleanroomGradientsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_values_and_resumed_state_match_torch_optim(self):
        for base_config in CONFIGS:
            for first_order in (False, True):
                for clipping in (False, True):
                    with self.subTest(
                        config=base_config, first_order=first_order, clipping=clipping
                    ):
                        config = base_config
                        """Compare every parameter and optimizer state after each real loss step."""
                        if clipping:
                            config = gradients.OptimizerConfig(
                                **{**config.__dict__, "max_grad_norm": 0.25}
                            )
                        model, batches, _, loss_fn = gradients.tiny_problem()
                        reference = copy.deepcopy(model)
                        optimizer = make_optimizer(reference, config)
                        params = dict(model.named_parameters())
                        state = None
                        for batch in batches:
                            before_state = copy.deepcopy(
                                {
                                    name: {
                                        key: (
                                            value.detach().clone()
                                            if torch.is_tensor(value)
                                            else value
                                        )
                                        for key, value in values.items()
                                    }
                                    for name, values in (state or {}).items()
                                }
                            )
                            result = gradients.functional_unroll(
                                model,
                                params,
                                (batch,),
                                loss_fn,
                                config,
                                first_order=first_order,
                                initial_state=state,
                            )
                            optimizer.zero_grad(set_to_none=True)
                            loss_fn(reference, batch).backward()
                            if clipping:
                                torch.nn.utils.clip_grad_norm_(
                                    reference.parameters(), config.max_grad_norm
                                )
                            optimizer.step()
                            for name, parameter in reference.named_parameters():
                                torch.testing.assert_close(
                                    result.params[name],
                                    parameter,
                                    rtol=1e-12,
                                    atol=1e-13,
                                )
                                expected = optimizer.state[parameter]
                                actual = result.state.get(name, {})
                                assert actual.keys() == expected.keys()
                                for key, value in actual.items():
                                    if key == "step":
                                        assert value == expected[key].item()
                                    else:
                                        torch.testing.assert_close(
                                            value, expected[key], rtol=1e-12, atol=1e-13
                                        )
                                if state and name in state:
                                    for key, value in state[name].items():
                                        if torch.is_tensor(value):
                                            torch.testing.assert_close(
                                                value,
                                                before_state[name][key],
                                                rtol=0,
                                                atol=0,
                                            )
                                        else:
                                            assert value == before_state[name][key]
                            assert result.trajectory == ()
                            assert all(not loss.requires_grad for loss in result.losses)
                            if first_order:
                                assert all(
                                    not value.requires_grad
                                    for values in result.state.values()
                                    for value in values.values()
                                    if torch.is_tensor(value)
                                )
                            params, state = result.params, result.state

    def test_exact_gradient_matches_fp64_directional_finite_differences(self):
        for base_config in CONFIGS:
            for clipping in (False, True):
                with self.subTest(config=base_config, clipping=clipping):
                    config = base_config
                    """Held-out nonlinear loss makes the exact/identity distinction observable."""
                    if clipping:
                        config = gradients.OptimizerConfig(
                            **{**config.__dict__, "max_grad_norm": 0.25}
                        )
                    model, batches, outer, loss_fn = gradients.tiny_problem()
                    assert not torch.equal(batches[-1][0], outer[0])
                    report = gradients.gradient_diagnostic(
                        model,
                        dict(model.named_parameters()),
                        batches,
                        loss_fn,
                        outer,
                        loss_fn,
                        config,
                        fd_epsilons=(1e-3, 1e-4, 1e-5),
                    )
                    differences = report["finite_differences"]
                    self.assertAlmostEqual(
                        report["exact_outer_loss"],
                        report["first_order_outer_loss"],
                        delta=1e-13,
                    )
                    assert report["exact_gradient_norm"] > 0.01
                    assert report["first_order_gradient_norm"] > 0.01
                    assert differences[-1]["exact_absolute_error"] < 2e-8
                    assert differences[-1]["exact_relative_error"] < 2e-6
                    assert (
                        differences[-1]["exact_absolute_error"]
                        < differences[0]["exact_absolute_error"] / 100
                    )
                    assert differences[-1]["first_order_absolute_error"] > 1e-5
                    assert report["gradient_cosine"] < 0.99999

    def test_first_order_is_identity_through_complete_update(self):
        for config in CONFIGS:
            with self.subTest(config=config):
                model, batches, outer, loss_fn = gradients.tiny_problem()
                params = dict(model.named_parameters())
                result = gradients.functional_unroll(
                    model,
                    params,
                    batches,
                    loss_fn,
                    config,
                    first_order=True,
                    store_trajectory=True,
                )
                assert len(result.trajectory) == len(batches) + 1
                probe = sum(
                    parameter.square().sum() for parameter in result.params.values()
                )
                actual = torch.autograd.grad(probe, tuple(params.values()))
                for value, parameter in zip(actual, result.params.values()):
                    torch.testing.assert_close(value, 2 * parameter, rtol=0, atol=0)
                expected_outer = torch.autograd.grad(
                    loss_fn(
                        gradients.functional_forward(
                            model, result.params, result.buffers
                        ),
                        outer,
                    ),
                    tuple(result.params.values()),
                )
                first = gradients.meta_gradient(
                    model,
                    params,
                    batches,
                    loss_fn,
                    outer,
                    loss_fn,
                    config,
                    first_order=True,
                )
                for value, expected in zip(first.gradient.values(), expected_outer):
                    torch.testing.assert_close(value, expected, rtol=1e-12, atol=1e-13)

    def test_none_gradient_skips_decay_and_state_advance(self):
        for name in ("sgd", "adamw"):
            with self.subTest(optimizer=name):
                params = {
                    "used": torch.tensor(0.7, requires_grad=True),
                    "unused": torch.tensor(-0.3, requires_grad=True),
                }
                config = gradients.OptimizerConfig(
                    name=name, lr=0.2, momentum=0.6, weight_decay=0.4
                )
                state = None
                for _ in range(3):
                    params, state = gradients.functional_optimizer_step(
                        params,
                        {"used": params["used"].square(), "unused": None},
                        config,
                        state,
                    )
                    self.assertAlmostEqual(params["unused"].item(), -0.3)
                    assert "unused" not in state

    def test_exact_adamw_derivative_stays_finite_at_zero_gradient(self):
        parameter = torch.tensor([0.0, 0.4], dtype=torch.float64, requires_grad=True)
        config = gradients.OptimizerConfig(name="adamw", lr=0.08, eps=0.03)
        result, _ = gradients.functional_optimizer_step(
            {"p": parameter}, {"p": parameter}, config
        )
        derivative = torch.autograd.grad(result["p"].sum(), parameter)[0]
        assert torch.isfinite(derivative).all()
        self.assertAlmostEqual(derivative[0].item(), 1.0 - config.lr / config.eps)

    def test_global_clipping_derivative_is_included(self):
        """Detaching the clip coefficient would fail this cross-coordinate check."""
        parameter = torch.tensor([0.7, -0.2], dtype=torch.float64, requires_grad=True)
        config = gradients.OptimizerConfig(lr=0.3, max_grad_norm=0.15)
        result, _ = gradients.functional_optimizer_step(
            {"p": parameter}, {"p": parameter.square()}, config
        )
        actual = torch.autograd.grad(result["p"][0], parameter)[0]
        assert abs(actual[1].item()) > 1e-3
        epsilon = 1e-5
        finite_differences = []
        for index in range(2):
            direction = torch.zeros_like(parameter)
            direction[index] = epsilon
            values = []
            for sign in (1, -1):
                point = parameter.detach() + sign * direction
                value, _ = gradients.functional_optimizer_step(
                    {"p": point}, {"p": point.square()}, config
                )
                values.append(value["p"][0])
            finite_differences.append((values[0] - values[1]) / (2 * epsilon))
        torch.testing.assert_close(
            actual, torch.stack(finite_differences), rtol=1e-7, atol=1e-9
        )

    def test_functional_attack_preserves_module_buffers_and_existing_gradients(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1).double()
                self.frozen = nn.Parameter(
                    torch.tensor(0.4, dtype=torch.float64), requires_grad=False
                )
                self.unused = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
                self.register_buffer("counter", torch.tensor(0))

            def forward(self, x):
                self.counter.add_(1)
                return self.linear(x).tanh() + self.frozen

        model = Model()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        original_state = copy.deepcopy(model.state_dict())
        original_grads = [
            None if p.grad is None else p.grad.clone() for p in model.parameters()
        ]
        params = {
            name: value
            for name, value in model.named_parameters()
            if value.requires_grad
        }
        batches = (torch.tensor([[0.4, -0.7]], dtype=torch.float64),) * 2
        with torch.no_grad():
            result = gradients.functional_unroll(
                model,
                params,
                batches,
                lambda forward, x: forward(x).square().mean(),
                gradients.OptimizerConfig(),
            )
        assert result.buffers["counter"].item() == 2
        assert result.params["unused"] is model.unused
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
        for parameter, original in zip(model.parameters(), original_grads):
            if original is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)

    @unittest.skipUnless(
        os.environ.get("TAR_CLEANROOM_TEST_MULTI_GPU") == "1"
        and torch.cuda.device_count() >= 2,
        "requires explicit TAR_CLEANROOM_TEST_MULTI_GPU=1 and two GPUs",
    )
    def test_sharded_norm_and_dot_transfer_only_scalar_reductions(self):
        left = {
            "a": torch.tensor([0.3, 0.4], device="cuda:0", dtype=torch.float64),
            "b": torch.tensor([1.2], device="cuda:1", dtype=torch.float64),
        }
        right = {name: value * 2 for name, value in left.items()}
        self.assertAlmostEqual(gradients.global_grad_norm(left).item(), 1.3)
        self.assertAlmostEqual(gradients.mapping_dot(left, right).item(), 2 * 1.3**2)
        config = gradients.OptimizerConfig(lr=0.1, max_grad_norm=0.5)
        updated, _ = gradients.functional_optimizer_step(left, right, config)
        coefficient = 0.5 / (2.6 + 1e-6)
        for name, value in left.items():
            torch.testing.assert_close(
                updated[name], value - 0.1 * right[name] * coefficient
            )


if __name__ == "__main__":
    unittest.main()
