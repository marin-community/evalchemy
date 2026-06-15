"""Run fingerprint (Stage-0 frozen artifact A) — SKELETON.

Stage 1 defines the *structure* + the MATERIAL/IGNORED include-exclude partition. The actual
content-hash-of-the-curated-controlling-file-SET compute (`from_run_inputs`) is **Stage 2** —
here it is a stub that hashes whatever resolved-input dict it is handed, after dropping the
IGNORED (cosmetic) fields.

Two-tier, mirroring Harbor (`lock.py:127-149` `_canonical_payload`, reconfirmed 2026-06-15):
  - `_canonical_payload(inputs)` strips the cosmetic/IGNORED fields and sorts, exactly like
    Harbor excluding `created_at`/`harbor`/`invocation`.
  - `value()` is a single sha256 over the canonical payload (decision #2: the fingerprint VALUE
    is one content-hash of the resolved set, NOT a struct of fields).

Pure stdlib so it is importable/testable anywhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict

SCHEMA_VERSION = 1

# --- Stage-0 partition (decision #2). These are the *contents* of the hashed controlling set vs
# the structurally-excluded cosmetic set. Stage 2 fills in `from_run_inputs` to resolve each
# MATERIAL field (content-hashing files where noted) before hashing; Stage 1 only wires the lists
# so a unit test can assert the partition is correct.

# IN the hashed set — any change must refuse a resume (material delta).
MATERIAL_FIELDS = (
    "rendered_config",        # the rendered eval config (resolved, not the raw CLI)
    "template_digest",        # rendered chat-template string hash
    "apply_chat_template",    # bool — template on/off is material
    "grader_source_digest",   # sha256 of the grader source
    "grader_version",         # pinned grader version tag
    "model_repo",             # HF repo id
    "model_revision",         # HF commit / revision
    "task_name",
    "task_data_digest",       # sha256 of the loaded data file (debug [:2] slice => different digest)
    "temperature",
    "top_p",
    "max_tokens",             # incl. max_new_tokens / max_gen_toks (normalized upstream in Stage 2)
    "do_sample",
    "n",
    "num_samples",
    "seed_set",               # full seed tuple; AIME24 n_repeat + per-repeat derivation
    "max_model_len",
    "num_fewshot",
    "passk_batch_size",       # pass@k problem-batch size B (decision #4)
)

# Structurally EXCLUDED from the hashed set — cosmetic; changing them still resumes.
# (TP / world_size is excluded here for *comparability* but a rank-layout mismatch refuses via
#  the per-rank manifest file naming — decision #3, handled in the manager, not the fingerprint.)
IGNORED_FIELDS = (
    "output_path",
    "job_name",
    "run_tag",
    "started_at",
    "wall_clock",
    "slurm_job_id",
    "batch_size",
    "max_batch_size",
    "gpu_memory_utilization",
    "tensor_parallel_size",
    "world_size",
    "rank",
    "verbosity",
    "log_level",
)


@dataclass
class RunFingerprint:
    """Resolved run inputs + the canonical-payload / content-hash machinery.

    Stage 1: `inputs` is whatever dict the caller resolved; `_canonical_payload` drops the
    IGNORED fields; `value` hashes it. Stage 2 replaces the resolution path with the real
    curated-file-set hashing in `from_run_inputs`.
    """

    inputs: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def _canonical_payload(self) -> Dict[str, Any]:
        """Strip the cosmetic (IGNORED) fields; keep everything else, sorted by key.

        Mirrors Harbor `JobLock._canonical_payload` (exclude set + stable sort). We exclude
        rather than allow-list so a NEW material field a caller adds is hashed by default
        (fail-safe toward refuse, not toward a false resume).
        """
        ignored = set(IGNORED_FIELDS)
        return {k: self.inputs[k] for k in sorted(self.inputs) if k not in ignored}

    def value(self) -> str:
        """Single sha256 over the canonical payload (decision #2)."""
        payload = json.dumps(self._canonical_payload(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "value": self.value(),
            "canonical_payload": self._canonical_payload(),
        }

    def matches(self, other: "RunFingerprint") -> bool:
        return self.value() == other.value()

    def diff_material(self, other: "RunFingerprint") -> Dict[str, Any]:
        """Material fields that changed vs `other` (for the refuse error message)."""
        a, b = self._canonical_payload(), other._canonical_payload()
        changed: Dict[str, Any] = {}
        for k in set(a) | set(b):
            if a.get(k) != b.get(k):
                changed[k] = {"this": a.get(k), "other": b.get(k)}
        return changed

    # ---- Stage 2 entry point (stubbed here) -------------------------------------------------
    @classmethod
    def from_run_inputs(cls, **resolved_inputs: Any) -> "RunFingerprint":
        """STUB (full content-hash-of-curated-file-SET compute lands in Stage 2).

        Stage 1 just stores the already-resolved inputs. Stage 2 will: read + sha256 the
        controlling files (data file, grader source, template), resolve model revision, and
        normalize the decoding-param aliases into the MATERIAL_FIELDS keys before hashing.
        """
        return cls(inputs=dict(resolved_inputs))
