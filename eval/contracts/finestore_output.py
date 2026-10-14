"""Native FineStore output for Evalchemy samples and source artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from eval.native_serialization import results_json, safe_artifact_name, sample_line

try:
    from finestore.eval import EvaluationStore
    from rigging.filesystem.storage_path import prefix_join

    from eval.contracts.lm_eval_normalization import samples_from_lm_eval

    _FINESTORE_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _FINESTORE_IMPORT_ERROR = error

_SAMPLE_CONTENT_TYPE = "application/x-ndjson"

# Serialized sample bytes buffered in memory before a durable FineStore commit
# (one immutable parquet shard + manifest per flush). Bounded so an interrupted
# write — e.g. an OOM kill during final aggregation — loses at most the current
# batch instead of the whole run, and the committed batches stay readable in the
# unsealed archive.
SAMPLE_COMMIT_BYTES = 16 * 1024 * 1024


def require_finestore_output() -> None:
    """Fail before evaluation when the FineStore output dependencies are unavailable."""
    if _FINESTORE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "--finestore_output_path requires the evalchemy[serve-eval] extra"
        ) from _FINESTORE_IMPORT_ERROR


def _commit_task_samples(
    store: EvaluationStore,
    normalized_task: str,
    task_samples: Sequence[Mapping[str, Any]],
    commit_bytes: int,
) -> list[bytes]:
    """Append one task's normalized samples, committing durably every ``commit_bytes``
    of serialized records, and return the task's encoded JSONL source lines."""
    lines: list[bytes] = []
    pending_bytes = 0
    for record in task_samples:
        for sample in samples_from_lm_eval(normalized_task, dict(record)):
            repeat = record.get("sample_repeat")
            store.add_sample(sample, trial_id="" if repeat is None else str(repeat))
        line = sample_line(record).encode()
        lines.append(line)
        pending_bytes += len(line)
        if pending_bytes >= commit_bytes:
            store.flush()
            pending_bytes = 0
    if pending_bytes:
        store.flush()
    return lines


def write_finestore_output(
    root: str,
    source_prefix: str,
    results: Mapping[str, Any],
    samples_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    commit_bytes: int = SAMPLE_COMMIT_BYTES,
) -> None:
    """Write Evalchemy's native sources and normalized samples to a FineStore run.

    Samples are committed in bounded batches as they are serialized, so a run
    interrupted while writing its output can be recovered from the archive alone,
    and writing the same run again afterwards is safe. A task's JSONL source
    artifact is written once, after that task's samples are committed.
    """
    require_finestore_output()

    store = EvaluationStore.open(root, writer_id=f"evalchemy-{uuid.uuid4().hex}")
    try:
        source_root = prefix_join(prefix_join("evalchemy", safe_artifact_name(source_prefix)), "native")
        result_name = "__".join(safe_artifact_name(task_name) for task_name in sorted(samples_by_task))
        store.add_source_artifact(
            prefix_join(source_root, f"results_{result_name or 'run'}.json"),
            results_json(results).encode(),
            content_type="application/json",
        )
        for task_name, task_samples in samples_by_task.items():
            if not task_samples:
                continue
            normalized_task = source_prefix if len(samples_by_task) == 1 else f"{source_prefix}/{task_name}"
            lines = _commit_task_samples(store, normalized_task, task_samples, commit_bytes)
            store.add_source_artifact(
                prefix_join(source_root, f"samples_{safe_artifact_name(task_name)}_native.jsonl"),
                b"".join(lines),
                content_type=_SAMPLE_CONTENT_TYPE,
            )
        store.seal()
    finally:
        store.close()
