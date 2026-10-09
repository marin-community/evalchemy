"""Native FineStore output for Evalchemy samples and source artifacts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from eval.native_serialization import results_json, safe_artifact_name, samples_jsonl

try:
    from finestore.eval import EvaluationStore
    from finestore.reader import ReadView
    from rigging.filesystem.storage_path import prefix_join

    from eval.contracts.lm_eval_normalization import samples_from_lm_eval

    _FINESTORE_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _FINESTORE_IMPORT_ERROR = error

_SAMPLE_CONTENT_TYPE = "application/x-ndjson"
_COMPLETION_FILE = "completed.json"


def _scored_results(results: Mapping[str, Any]) -> bool:
    outcomes = results.get("task_outcomes", {})
    return bool(results.get("results")) and all(
        outcome.get("status") in ("succeeded", "exported") for outcome in outcomes.values()
    )


def completed_finestore_output(root: str, source_prefix: str) -> bool:
    """Whether this task's native results and completion marker share a committed archive view."""
    require_finestore_output()
    view = ReadView(root)
    source_root = prefix_join(prefix_join("evalchemy", safe_artifact_name(source_prefix)), "native")
    marker = view.read_blob(prefix_join("sources", prefix_join(source_root, _COMPLETION_FILE)))
    return marker is not None and json.loads(marker)["scored"] is True


def require_finestore_output() -> None:
    """Fail before evaluation when the FineStore output dependencies are unavailable."""
    if _FINESTORE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "--finestore_output_path requires the evalchemy[serve-eval] extra"
        ) from _FINESTORE_IMPORT_ERROR


def write_finestore_output(
    root: str,
    source_prefix: str,
    results: Mapping[str, Any],
    samples_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Write Evalchemy's native sources and normalized samples to a FineStore run."""
    require_finestore_output()
    if completed_finestore_output(root, source_prefix):
        return

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
            safe_task_name = safe_artifact_name(task_name)
            normalized_task = source_prefix if len(samples_by_task) == 1 else f"{source_prefix}/{task_name}"
            store.add_source_artifact(
                prefix_join(source_root, f"samples_{safe_task_name}_native.jsonl"),
                samples_jsonl(task_samples).encode(),
                content_type=_SAMPLE_CONTENT_TYPE,
            )
            for record in task_samples:
                for sample in samples_from_lm_eval(normalized_task, dict(record)):
                    repeat = record.get("sample_repeat")
                    store.add_sample(sample, trial_id="" if repeat is None else str(repeat))
        store.add_source_artifact(
            prefix_join(source_root, _COMPLETION_FILE),
            json.dumps({"scored": _scored_results(results)}).encode(),
            content_type="application/json",
        )
        store.seal()
    finally:
        store.close()
