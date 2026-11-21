"""Narrow compatibility wrapper for reused geometry progress metadata."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable


_OPERATIONAL_SETTINGS = frozenset({"capability_scores_file", "cpu_landscape_anchor"})


def _saved_progress(record: dict[str, Any]) -> dict[str, Any] | None:
    folder = record.get("folder")
    if folder is None:
        return None
    path = Path(folder) / "geometry" / "progress-protocol.json"
    try:
        progress = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return None
    return progress if isinstance(progress, dict) else None


def wrap_checkpoint_validation(
    original_validator: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Allow only two non-geometric study settings to differ for cached geometry.

    The original validator still checks the saved protocol's self-hash, progress
    identities, accumulators, owned basis files, input stats, and all numerical
    report gates. Only its settings comparison sees a shallow record copy with
    the two operational values restored to the saved protocol values.
    """

    def validate(
        record: dict[str, Any], program: dict[str, Any], probes: dict[str, Any]
    ):
        saved_progress = _saved_progress(record)
        protocol = saved_progress.get("protocol") if saved_progress else None
        saved = protocol.get("settings") if isinstance(protocol, dict) else None
        current = record.get("settings")
        if not isinstance(current, dict) or not isinstance(saved, dict):
            return original_validator(record, program, probes)

        keys = set(saved) | set(current)
        changed = sorted(
            key
            for key in keys
            if (key in saved) != (key in current)
            or (key in saved and saved[key] != current[key])
        )
        if not changed or not set(changed) <= _OPERATIONAL_SETTINGS:
            return original_validator(record, program, probes)

        projected = copy.copy(record)
        projected_settings = dict(current)
        old_values = {}
        new_values = {}
        for key in changed:
            old_values[key] = saved[key] if key in saved else {"__absent__": True}
            new_values[key] = current[key] if key in current else {"__absent__": True}
            if key in saved:
                projected_settings[key] = saved[key]
            else:
                projected_settings.pop(key, None)
        projected["settings"] = projected_settings

        result = original_validator(projected, program, probes)
        result = dict(result)
        result["geometry_reuse"] = {
            "state": "cached_geometry_reused_with_operational_settings_projection",
            "changed_fields": changed,
            "saved_protocol_values": old_values,
            "current_study_values": new_values,
            "saved_protocol_identity": saved_progress.get("identity"),
            "geometry_recomputed": False,
        }
        return result

    return validate
