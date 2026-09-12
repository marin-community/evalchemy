"""Canonical per-example records written by ``eval --log_samples``.

The evaluators use a variety of internal result shapes.  This module is the
single serialization boundary: it gives custom benchmarks and lm-eval-native
tasks the same record envelope before an ``EvaluationTracker`` writes JSONL.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from eval.completion_response import CompletionText
from eval.contracts.sample_manifest import SampleCoverageError, SampleManifest
from eval.lm_eval_tasks.drop.utils import DropAnswer

SAMPLE_SCHEMA_VERSION = 1
"""Version of the stable JSONL record envelope emitted by ``--log_samples``."""


def is_scored_result(result: Any) -> bool:
    """Return whether a task completed scoring rather than returning an error."""
    return isinstance(result, Mapping) and bool(result) and "error" not in result


def canonicalize_samples(
    task_name: str,
    samples: Sequence[Mapping[str, Any]],
    sample_manifest: SampleManifest | None = None,
) -> list[dict[str, Any]]:
    """Add the stable envelope to lm-eval-compatible sample records.

    The first lm-eval filter record stays at the top level. Every filter's
    response and metrics are stored under ``filter_variants``. The envelope makes
    task identity and schema version explicit, while filling fields that custom
    benchmark adapters must provide for tracker-compatible JSONL.
    """
    manifest_entries = ()
    if sample_manifest is not None and sample_manifest.expected_sample_count:
        manifest_entries = sample_manifest.sample_entries(namespace=task_name)
        if not manifest_entries and sample_manifest.task_name == task_name:
            manifest_entries = sample_manifest.sample_entries()
        samples = _coalesce_lm_eval_filter_variants(samples, len(manifest_entries))
        if len(manifest_entries) != len(samples):
            raise SampleCoverageError(
                f"{task_name}: sample logger received {len(samples)} records but the manifest "
                f"contains {len(manifest_entries)} samples"
            )

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
        if manifest_entries:
            entry = manifest_entries[doc_id]
            record["sample_id"] = entry.sample_id
            record["source_id"] = entry.source_id
            record["sample_ordinal"] = entry.ordinal
            record["sample_namespace"] = entry.namespace
            record["sample_shard"] = entry.shard
            record["sample_repeat"] = entry.repeat
        completion_artifacts = _completion_artifacts(record["resps"])
        if completion_artifacts is not None:
            record["completion_responses"] = completion_artifacts
        drop_extractions = _drop_extractions(record["filtered_resps"])
        if drop_extractions is not None:
            record["drop_extractions"] = drop_extractions
        canonical.append(record)
    return canonical


def _coalesce_lm_eval_filter_variants(
    samples: Sequence[Mapping[str, Any]], expected_sample_count: int
) -> list[Mapping[str, Any]]:
    """Return one sample record per lm-eval document while retaining every filter result."""
    records = list(samples)
    if len(records) <= expected_sample_count:
        return records

    by_doc_id: dict[str | int, list[Mapping[str, Any]]] = {}
    for sample in records:
        doc_id = sample.get("doc_id")
        filter_name = sample.get("filter")
        if not isinstance(doc_id, (str, int)) or not isinstance(filter_name, str) or filter_name == "none":
            return records
        if any(field not in sample for field in ("doc_hash", "prompt_hash", "target_hash")):
            return records
        by_doc_id.setdefault(doc_id, []).append(sample)

    if len(by_doc_id) != expected_sample_count:
        return records

    expected_filters: tuple[str, ...] | None = None
    coalesced: list[Mapping[str, Any]] = []
    for variants in by_doc_id.values():
        filters = tuple(str(variant["filter"]) for variant in variants)
        if len(filters) != len(set(filters)):
            return records
        if expected_filters is None:
            expected_filters = filters
        elif filters != expected_filters:
            return records
        signatures = {(variant["doc_hash"], variant["prompt_hash"], variant["target_hash"]) for variant in variants}
        if len(signatures) != 1:
            return records

        primary = dict(variants[0])
        primary["filter_variants"] = [_filter_variant(variant) for variant in variants]
        coalesced.append(primary)
    return coalesced


def _filter_variant(sample: Mapping[str, Any]) -> dict[str, Any]:
    metric_names = sample.get("metrics", ())
    if isinstance(metric_names, (str, bytes)) or not isinstance(metric_names, Sequence):
        metric_names = ()
    return {
        "filter": sample["filter"],
        "filtered_resps": sample.get("filtered_resps", []),
        "metrics": {str(name): sample.get(name) for name in metric_names},
    }


def _completion_artifacts(value: Any) -> Any | None:
    """Mirror scorer-response nesting with audit data for normalized chat output."""
    if isinstance(value, CompletionText):
        return value.artifact()
    if isinstance(value, Sequence) and not isinstance(value, str):
        artifacts = [_completion_artifacts(item) for item in value]
        return artifacts if any(artifact is not None for artifact in artifacts) else None
    return None


def _drop_extractions(value: Any) -> Any | None:
    """Mirror filtered-response nesting with DROP extraction classifications."""
    if isinstance(value, DropAnswer):
        return {"classification": value.classification}
    if isinstance(value, Sequence) and not isinstance(value, str):
        extractions = [_drop_extractions(item) for item in value]
        return extractions if any(extraction is not None for extraction in extractions) else None
    return None


def without_embedded_samples(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return aggregate metrics without the custom benchmark's raw examples."""
    aggregate = dict(result)
    aggregate.pop("examples", None)
    return aggregate
