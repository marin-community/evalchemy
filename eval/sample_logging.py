"""Canonical per-example records written by ``eval --log_samples``.

The evaluators use a variety of internal result shapes.  This module is the
single serialization boundary: it gives custom benchmarks and lm-eval-native
tasks the same record envelope before an ``EvaluationTracker`` writes JSONL.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from eval.completion_response import CompletionText


SAMPLE_SCHEMA_VERSION = 1
"""Version of the stable JSONL record envelope emitted by ``--log_samples``."""


def is_scored_result(result: Any) -> bool:
    """Return whether a task completed scoring rather than returning an error."""
    return isinstance(result, Mapping) and bool(result) and "error" not in result


def canonicalize_samples(task_name: str, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add the stable envelope to lm-eval-compatible sample records.

    Existing lm-eval fields are retained unchanged.  The envelope makes task
    identity and schema version explicit, while filling fields that custom
    benchmark adapters must provide for tracker-compatible JSONL.
    """
    canonical: list[dict[str, Any]] = []
    for doc_id, sample in enumerate(samples):
        record = dict(sample)
        record["schema_version"] = SAMPLE_SCHEMA_VERSION
        record["task_name"] = task_name
        record.setdefault("doc_id", doc_id)
        record.setdefault("doc", {})
        record.setdefault("target", "")
        record.setdefault("arguments", [])
        record.setdefault("resps", [])
        record.setdefault("filtered_resps", [])
        record.setdefault("filter", "none")
        completion_artifacts = _completion_artifacts(record["resps"])
        if completion_artifacts is not None:
            record["completion_responses"] = completion_artifacts
        canonical.append(record)
    return canonical


def _completion_artifacts(value: Any) -> Any | None:
    """Mirror scorer-response nesting with audit data for normalized chat output."""
    if isinstance(value, CompletionText):
        return value.artifact()
    if isinstance(value, Sequence) and not isinstance(value, str):
        artifacts = [_completion_artifacts(item) for item in value]
        return artifacts if any(artifact is not None for artifact in artifacts) else None
    return None


def without_embedded_samples(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate metrics without the custom benchmark's raw examples."""
    aggregate = dict(result)
    aggregate.pop("examples", None)
    return aggregate
