import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from precision_audit import (
    analyze_pair,
    bf16_roundtrip,
    straight_through_reconstruction,
    validate_pair_inputs,
)


def test_bf16_roundtrip_compares_against_native_cast_back_to_original_dtype():
    native = torch.tensor([1.0001, 1.125, -3.14159, 0.0], dtype=torch.float32)
    anchor = native.to(torch.bfloat16).to(torch.float32)
    delta = bf16_roundtrip(anchor, native)
    assert torch.count_nonzero(delta).item() == 0

    changed = anchor.clone()
    changed[0] = torch.nextafter(changed[0], torch.tensor(float("inf")))
    delta = bf16_roundtrip(changed, native)
    assert torch.count_nonzero(delta).item() == 1
    assert delta.dtype == torch.float64


def test_straight_through_reconstruction_uses_parameter_dtype_at_both_ops():
    anchor = torch.tensor([1.0], dtype=torch.bfloat16)
    endpoint = torch.tensor([1e-4], dtype=torch.bfloat16)
    effective = straight_through_reconstruction(anchor, endpoint)
    assert endpoint.item() > 0
    assert (endpoint - anchor).item() == -1.0
    assert effective.item() == 0.0
    assert effective.dtype == torch.bfloat16

    fp32_anchor = torch.tensor([1.0], dtype=torch.float32)
    fp32_endpoint = torch.tensor([1e-4], dtype=torch.float32)
    fp32_effective = straight_through_reconstruction(fp32_anchor, fp32_endpoint)
    assert fp32_effective.item() == pytest.approx(fp32_endpoint.item(), abs=1e-7)
    assert torch.isfinite(fp32_effective).all()


def test_streamed_mixed_dtype_audit_matches_hand_computed_stats(tmp_path: Path):
    keys = {
        "model.embed_tokens.weight": torch.tensor(
            [1.0001, 1.125, -3.14159], dtype=torch.float32
        ),
        "model.layers.0.input_layernorm.weight": torch.tensor(
            [0.9999, 1.0], dtype=torch.float32
        ),
        "model.layers.0.self_attn.q_proj.weight": torch.tensor(
            [1.0, 2.0], dtype=torch.bfloat16
        ),
    }
    native = tmp_path / "native.safetensors"
    save_file(keys, native)

    def bf16_reload(values):
        return {
            key: value.to(torch.bfloat16).to(value.dtype)
            for key, value in values.items()
        }

    anchor = bf16_reload(keys)
    parity = tmp_path / "parity"
    anchor_path = parity / "anchor/model.safetensors"
    anchor_path.parent.mkdir(parents=True)
    save_file(anchor, anchor_path)
    actual_root = tmp_path / "attack"
    checks = []
    for step in (1, 2, 4, 8, 16):
        actual = {key: value.clone() for key, value in anchor.items()}
        actual["model.embed_tokens.weight"][0] += 0.01 * step
        actual["model.layers.0.input_layernorm.weight"][1] += 0.001 * step
        actual["model.layers.0.self_attn.q_proj.weight"][0] = torch.tensor(
            1e-4, dtype=torch.bfloat16
        )
        archived = {key: value.clone() for key, value in actual.items()}
        archived["model.embed_tokens.weight"][0] += 0.002 * step
        actual_dir = actual_root / f"checkpoint-{step}"
        actual_dir.mkdir(parents=True)
        save_file(archived, actual_dir / "model.safetensors")
        sim_dir = parity / f"simulated-{step}"
        sim_dir.mkdir(parents=True)
        save_file(actual, sim_dir / "model.safetensors")
        exact = {"exact_equal": True, "difference_l2": 0.0}
        checks.append(
            {
                "step": step,
                "simulated_vs_actual": exact,
                "archived_vs_actual": {
                    "exact_equal": False,
                    "difference_l2": 0.002 * step,
                    "relative_update_error": 0.01 * step,
                },
            }
        )

    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "state": "complete",
                "trajectory_parity": "exact",
                "reference_root": str(actual_root),
                "actual_loss_errors": [0.0] * 16,
                "parameter_checks": checks,
            }
        )
    )
    rows = analyze_pair("fixture", native, parity, validation, torch.device("cpu"), 2)

    archive_diagnostic = next(
        row
        for row in rows
        if row["measure"] == "archived_vs_cuda_simulated"
        and row["module_bucket"] == "embedding_or_head"
        and row["step"] == 1
    )
    assert archive_diagnostic["difference_l2"] == pytest.approx(0.002)
    # Archived cross-backend mismatch is accepted when same-backend parity is exact.
    validate_pair_inputs(native, parity, validation)

    embed_roundtrip = next(
        row
        for row in rows
        if row["measure"] == "native_bf16_roundtrip"
        and row["module_bucket"] == "embedding_or_head"
    )
    expected_roundtrip = (
        anchor["model.embed_tokens.weight"].double()
        - keys["model.embed_tokens.weight"].to(torch.bfloat16).float().double()
    )
    assert embed_roundtrip["difference_l2"] == 0.0
    assert embed_roundtrip["changed_coordinates"] == 0
    assert embed_roundtrip["exact_equal_to_native_bf16_roundtrip"] is True
    assert expected_roundtrip.norm().item() == 0.0

    embed_reload = next(
        row
        for row in rows
        if row["measure"] == "native_to_reload"
        and row["module_bucket"] == "embedding_or_head"
    )
    raw = (
        anchor["model.embed_tokens.weight"].double()
        - keys["model.embed_tokens.weight"].double()
    )
    update = (
        actual["model.embed_tokens.weight"].double()
        - anchor["model.embed_tokens.weight"].double()
    )
    assert embed_reload["difference_l2"] == pytest.approx(raw.norm().item())
    assert embed_reload["relative_to_actual_attack_update_1_l2"] == pytest.approx(
        raw.norm().item() / 0.01
    )
    # The last endpoint is step 16; the actual native-model displacement is 0.16.
    assert embed_reload["relative_to_actual_attack_update_16_l2"] == pytest.approx(
        raw.norm().item() / update.norm().item()
    )

    st = next(
        row
        for row in rows
        if row["measure"] == "straight_through_reconstruction"
        and row["module_bucket"] == "other"
        and row["step"] == 1
    )
    q_endpoint = actual["model.layers.0.self_attn.q_proj.weight"][0].item()
    assert st["changed_coordinates"] == 1
    assert st["difference_l2"] == pytest.approx(q_endpoint)
    assert st["maximum_absolute_difference"] == pytest.approx(q_endpoint)

    # A same-backend simulated-versus-genuine mismatch still blocks the audit.
    payload = json.loads(validation.read_text())
    payload["parameter_checks"][0]["simulated_vs_actual"] = {
        "exact_equal": False,
        "difference_l2": 1e-8,
    }
    validation.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="simulated_vs_actual state 1"):
        validate_pair_inputs(native, parity, validation)
