"""Stage lm-eval samples at its scoring boundary without retaining logged samples."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

from lm_eval.utils import handle_non_serializable, hash_string


class StreamingTaskManager:
    """Decorate loaded lm-eval tasks so each scored document reaches a callback."""

    def __init__(self, task_manager: Any, on_sample: Callable[[str, dict[str, Any], int], None]) -> None:
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
