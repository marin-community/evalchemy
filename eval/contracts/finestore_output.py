"""Native FineStore output for Evalchemy samples and source artifacts."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from lm_eval.utils import handle_non_serializable

try:
    from finestore.eval import EvaluationStore, samples_from_lm_eval

    _FINESTORE_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _FINESTORE_IMPORT_ERROR = error

_SOURCE_CONTENT_TYPE = "application/x-ndjson"


def _results_json_bytes(results: Mapping[str, Any]) -> bytes:
    return json.dumps(
        results,
        indent=2,
        default=handle_non_serializable,
        ensure_ascii=False,
    ).encode()


def _sample_jsonl_bytes(samples: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(sample, default=handle_non_serializable, ensure_ascii=False) + "\n" for sample in samples
    ).encode()


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w.-]", "_", value) or "task"


def write_finestore_output(
    root: str,
    source_prefix: str,
    results: Mapping[str, Any],
    samples_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Write Evalchemy's native sources and normalized samples to a FineStore run."""
    if _FINESTORE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "--finestore_output_path requires the evalchemy[serve-eval] extra"
        ) from _FINESTORE_IMPORT_ERROR

    store = EvaluationStore.open(root, writer_id=f"evalchemy-{uuid.uuid4().hex}")
    try:
        source_root = f"evalchemy/{_safe_name(source_prefix)}/native"
        result_name = "__".join(_safe_name(task_name) for task_name in sorted(samples_by_task))
        store.add_source_artifact(
            f"{source_root}/results_{result_name or 'run'}.json",
            _results_json_bytes(results),
            content_type="application/json",
        )
        for task_name, task_samples in samples_by_task.items():
            if not task_samples:
                continue
            safe_task_name = _safe_name(task_name)
            store.add_source_artifact(
                f"{source_root}/samples_{safe_task_name}_native.jsonl",
                _sample_jsonl_bytes(task_samples),
                content_type=_SOURCE_CONTENT_TYPE,
            )
            for record in task_samples:
                for sample in samples_from_lm_eval(task_name, dict(record)):
                    store.add_sample(sample)
        store.seal()
    finally:
        store.close()
