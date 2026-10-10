"""Stable per-request keys shared by Evalchemy resume paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

UnitKey = tuple[tuple[str, Any], ...]


def canonical_unit_key(unit: dict[str, Any]) -> UnitKey:
    """Map a request identity to a deterministic hashable key."""
    return tuple(sorted((str(key), value) for key, value in unit.items()))


def find_restored_payload(
    restored: Mapping[UnitKey, dict[str, Any]],
    candidate_units: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Find the first current or historical unit key in restored state."""
    for unit in candidate_units:
        key = canonical_unit_key(unit)
        try:
            hash(key)
        except TypeError:
            continue
        payload = restored.get(key)
        if payload is not None:
            return unit, payload
    return None
