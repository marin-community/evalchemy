"""Scored lm-eval records reach FineStore as each document is processed."""

from pathlib import Path
from types import SimpleNamespace

from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView

from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.lm_eval_streaming import StreamingTaskManager
from eval.sample_logging import canonicalize_sample


class _Task:
    def __init__(self):
        self.instances = [
            SimpleNamespace(
                doc_id=0,
                idx=0,
                doc={"question": "2 + 2?"},
                args=["2 + 2?", {}],
                arguments=["2 + 2?", {}],
                resps=["4"],
                filtered_resps={"strict": "wrong", "flexible": "4"},
            )
        ]

    def apply_filters(self):
        pass

    def doc_iterator(self, **_kwargs):
        yield 0, {"question": "2 + 2?"}

    def process_results(self, _doc, responses):
        return {"exact_match": float(responses[0] == "4")}

    def doc_to_target(self, _doc):
        return "4"


class _TaskManager:
    def __init__(self, task):
        self.task = task

    def load(self, _tasks):
        return {"tasks": {"gsm8k": self.task}, "groups": {}}


def test_streaming_lm_eval_scores_filter_variants_before_result_commit(tmp_path: Path):
    root = str(tmp_path / "archive")
    task = _Task()
    with FineStoreOutput(root, "gsm8k_0shot", ["gsm8k"]) as output:
        manager = StreamingTaskManager(
            _TaskManager(task),
            lambda name, sample, index: output.write_sample(name, canonicalize_sample(name, sample, index)),
        )
        scored_task = manager.load(["gsm8k"])["tasks"]["gsm8k"]
        list(scored_task.doc_iterator())  # lm-eval also iterates documents while building requests
        scored_task.apply_filters()

        for filter_name in scored_task.instances[0].filtered_resps:
            for _, doc in scored_task.doc_iterator():
                response = scored_task.instances[0].filtered_resps[filter_name]
                scored_task.process_results(doc, [response])
        output.finish_task()

        staged = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        assert staged is not None
        samples = {
            row["filter"]: sample_from_archive_row(row)
            for row in staged.to_pylist(maps_as_pydicts="strict")
        }
        assert {name: sample.correct for name, sample in samples.items()} == {
            "strict": False,
            "flexible": True,
        }
        output.finish({"results": {"gsm8k": {"exact_match": 0.5}}})
