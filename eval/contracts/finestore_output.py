"""Native FineStore output for Evalchemy samples and source artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from eval.native_serialization import results_json, safe_artifact_name, samples_jsonl

try:
    from finestore.eval import EvaluationStore
    from rigging.filesystem.storage_path import prefix_join

    from eval.contracts.lm_eval_normalization import samples_from_lm_eval

    _FINESTORE_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _FINESTORE_IMPORT_ERROR = error

_SAMPLE_CONTENT_TYPE = "application/x-ndjson"


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
                    store.add_sample(sample)
        store.seal()
    finally:
        store.close()
