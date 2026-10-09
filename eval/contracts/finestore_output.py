"""FineStore result and sample tables for Evalchemy evaluations."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    SAMPLES_MERGE_KEY,
    SCHEMA_VERSION,
    sample_to_archive_row,
    samples_schema,
)
from finestore.layout import OnConflict
from finestore.reader import ReadView
from finestore.store import DataStore
from lm_eval.utils import handle_non_serializable

from eval.contracts.lm_eval_normalization import samples_from_lm_eval

RESULTS_TABLE = "evalchemy_results"
DEFAULT_SOURCE_PREFIX = "run"
_SOURCE_PREFIX = "source_prefix"
_DOCUMENT = "document"
_SCORED = "scored"
_RESULTS_SCHEMA = pa.schema(
    [pa.field(_SOURCE_PREFIX, pa.string()), pa.field(_DOCUMENT, pa.large_string()), pa.field(_SCORED, pa.bool_())]
)


def _scored_results(results: Mapping[str, Any]) -> bool:
    outcomes = results.get("task_outcomes", {})
    return bool(results.get("results")) and all(
        outcome.get("status") in ("succeeded", "exported") for outcome in outcomes.values()
    )


def _result_row(root: str, source_prefix: str) -> dict[str, Any] | None:
    rows = list(ReadView(root).iter_rows(RESULTS_TABLE, where=[(_SOURCE_PREFIX, "==", source_prefix)]))
    return rows[0] if rows else None


def read_finestore_output(root: str, source_prefix: str) -> dict[str, Any] | None:
    """Read one Evalchemy result document from the FineStore results table."""
    row = _result_row(root, source_prefix)
    return None if row is None else json.loads(row[_DOCUMENT])


def completed_finestore_output(root: str, source_prefix: str) -> bool:
    """Whether this task has a committed successful result in FineStore."""
    row = _result_row(root, source_prefix)
    return row is not None and row[_SCORED]


def write_finestore_output(
    root: str,
    source_prefix: str,
    results: Mapping[str, Any],
    samples_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Commit aggregate results and normalized samples in the same FineStore archive."""
    if completed_finestore_output(root, source_prefix):
        return

    scored = _scored_results(results)
    store = DataStore.open(root, writer_id=f"evalchemy-{uuid.uuid4().hex}")
    try:
        store.table(
            RESULTS_TABLE,
            primary_key=(_SOURCE_PREFIX,),
            schema=_RESULTS_SCHEMA,
            on_conflict=OnConflict.SUPERSEDE,
        )
        if scored:
            store.table(
                ARCHIVE_SAMPLES_TABLE,
                primary_key=SAMPLES_MERGE_KEY,
                schema=samples_schema(),
                schema_version=SCHEMA_VERSION,
            )
        with store.unbounded_transaction() as transaction:
            if scored:
                sample_rows = transaction.table(ARCHIVE_SAMPLES_TABLE)
                for task_name, task_samples in samples_by_task.items():
                    normalized_task = source_prefix if len(samples_by_task) == 1 else f"{source_prefix}/{task_name}"
                    for record in task_samples:
                        for sample in samples_from_lm_eval(normalized_task, dict(record)):
                            repeat = record.get("sample_repeat")
                            sample_rows.add(sample_to_archive_row(sample, trial_id="" if repeat is None else str(repeat)))
            transaction.table(RESULTS_TABLE).add(
                {
                    _SOURCE_PREFIX: source_prefix,
                    _DOCUMENT: json.dumps(results, default=handle_non_serializable, ensure_ascii=False),
                    _SCORED: scored,
                }
            )
        if scored:
            store.seal()
    finally:
        store.close()
