"""Unified eval resume manager (Stage 1 — standalone, wired to nothing yet).

Public surface (frozen Stage-0 interface):
    ResumeManager(run_dir, fingerprint, mode="auto", world_size=1, rank=0)
        .decide() -> "fresh" | "resume" | "refuse"
        .should_skip(unit) -> bool
        .record(unit, payload) -> None
        .restore() -> dict[UnitKey, payload]
        .done_units() -> set[UnitKey]
        .finalize() -> None
    RunFingerprint  — content-hash of the curated controlling-input set (Stage-2 compute stub).
    ResumeRefused   — raised on material fingerprint / rank-layout mismatch.

NOTE: importing this package has no effect on any eval path until Stages 3a/3b/3c/4 wire it in.
"""

from __future__ import annotations

from .fingerprint import IGNORED_FIELDS, MATERIAL_FIELDS, RunFingerprint
from .manager import ResumeManager, ResumeRefused
from .manifest import (
    ManifestWriter,
    UnitState,
    canonical_unit_key,
    read_manifest,
    unit_key_to_dict,
)

__all__ = [
    "ResumeManager",
    "ResumeRefused",
    "RunFingerprint",
    "MATERIAL_FIELDS",
    "IGNORED_FIELDS",
    "ManifestWriter",
    "UnitState",
    "read_manifest",
    "canonical_unit_key",
    "unit_key_to_dict",
]
