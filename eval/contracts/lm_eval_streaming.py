"""Stage lm-eval samples at its scoring boundary without retaining logged samples."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, hash_string

from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.sample_manifest import SampleCoverageError, SampleEntry, SampleManifest
from eval.contracts.sample_results import validate_sample_metrics
from eval.sample_logging import canonicalize_sample, manifest_entries_for_task


def streams_scored_samples(task_manager: TaskManager, task_name: str, predict_only: bool) -> bool:
    """Keep lm-eval's required sample logging for bypass/output-only tasks."""
    if predict_only:
        return False
    task_entry = task_manager.task_index.get(task_name)
    metrics = (task_entry.cfg or {}).get("metric_list", []) if task_entry is not None else []
    return not any(isinstance(metric, Mapping) and metric.get("metric") == "bypass" for metric in metrics)


class FineStoreSampleCallback:
    """Canonicalize and stage each scored lm-eval record with manifest identity."""

    def __init__(self, sample_manifest: SampleManifest, output: FineStoreOutput) -> None:
        self._manifest = sample_manifest
        self._output = output
        self._entries: dict[str, tuple[SampleEntry, ...]] = {}
        self._has_manifest: dict[str, bool] = {}

    def __call__(self, task_name: str, sample: dict[str, Any], index: int) -> None:
        if task_name not in self._entries:
            self._has_manifest[task_name] = self._manifest.expected_sample_count > 0
            self._entries[task_name] = manifest_entries_for_task(task_name, self._manifest)[0]
        entries = self._entries[task_name]
        if self._has_manifest[task_name] and index >= len(entries):
            raise SampleCoverageError(f"{task_name}: scored sample {index} has no manifest entry")
        entry = entries[index] if entries else None
        record = canonicalize_sample(task_name, sample, index, entry)
        validate_sample_metrics(task_name, [record])
        self._output.write_sample(task_name, record)


class StreamingTaskManager:
    """Decorate loaded lm-eval tasks so each scored document reaches a callback."""

    def __init__(self, task_manager: TaskManager, on_sample: Callable[[str, dict[str, Any], int], None]) -> None:
        self._task_manager = task_manager
        self._on_sample = on_sample

    def __getattr__(self, name: str) -> Any:
        return getattr(self._task_manager, name)

    def load(self, tasks: list[Any]) -> dict[str, Any]:
        """Load tasks and attach sample callbacks before lm-eval builds requests."""
        loaded = self._task_manager.load(tasks)
        for task_name, task in loaded["tasks"].items():
            self._decorate(task_name, task)
        return loaded

    def _decorate(self, task_name: str, task: Any) -> None:
        original_iterator = task.doc_iterator
        original_process_results = task.process_results
        original_apply_filters = task.apply_filters
        filter_index = -1
        scoring = False
        current: tuple[int, int, int, str] | None = None
        requests_by_doc_id: dict[int, list[Any]] | None = None

        def apply_filters() -> None:
            nonlocal filter_index, requests_by_doc_id, scoring
            original_apply_filters()
            filter_index = -1
            requests_by_doc_id = None
            scoring = True

        def doc_iterator(*args: Any, **kwargs: Any) -> Iterator[tuple[int, Any]]:
            nonlocal filter_index, current, requests_by_doc_id
            if not scoring:
                yield from original_iterator(*args, **kwargs)
                return
            # lm-eval visits the documents once for each filter in this order.
            filter_index += 1
            if requests_by_doc_id is None:
                requests_by_doc_id = defaultdict(list)
                for request in task.instances:
                    requests_by_doc_id[request.doc_id].append(request)
                for requests in requests_by_doc_id.values():
                    requests.sort(key=lambda request: request.idx)
            filter_name = tuple(task.instances[0].filtered_resps)[filter_index]
            indices = kwargs.get("samples")
            for ordinal, (doc_id, doc) in enumerate(original_iterator(*args, **kwargs)):
                true_doc_id = indices[doc_id] if indices else doc_id
                current = (doc_id, true_doc_id, ordinal, filter_name)
                yield doc_id, doc
            current = None

        def process_results(doc: Any, responses: Any) -> dict[str, Any]:
            metrics = original_process_results(doc, responses)
            assert current is not None and requests_by_doc_id is not None
            doc_id, true_doc_id, ordinal, filter_name = current
            requests = requests_by_doc_id[doc_id]
            target = task.doc_to_target(doc)
            sample = {
                "doc_id": true_doc_id,
                "doc": doc,
                "target": target,
                "arguments": [request.args for request in requests],
                "resps": [request.resps for request in requests],
                "filtered_resps": [request.filtered_resps[filter_name] for request in requests],
                "filter": filter_name,
                "metrics": list(metrics),
                "doc_hash": hash_string(json.dumps(requests[0].doc, indent=2, default=handle_non_serializable, ensure_ascii=False)),
                "prompt_hash": hash_string(requests[0].arguments[0]),
                "target_hash": hash_string(str(target)),
            }
            sample.update(metrics)
            self._on_sample(task_name, sample, ordinal)
            return metrics

        task.apply_filters = apply_filters
        task.doc_iterator = doc_iterator
        task.process_results = process_results
