"""Offline source-to-judge fixtures for judge-backed benchmarks."""

import pytest

from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.WildBench import eval_instruct as wildbench_source
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ReducerSpec, SourceRef
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.conversations import turn_ratings
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import evaluate_source


class RoutedEndpoint:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return self.response(request) if callable(self.response) else self.response


def test_mixeval_preserves_both_split_identities_and_judge_parsers():
    source = MixEvalBenchmark(debug=True)
    candidate = RoutedEndpoint("candidate")
    judge = RoutedEndpoint(
        lambda request: "[[1.0]]" if "correctness" in request.prompt[-1]["content"].lower() else "[[C]]"
    )

    metrics, samples = evaluate_source(
        source,
        SourceOptions("MixEval", SourceRef("fixture"), limit=1),
        candidate,
        judges={"MixEval": judge},
    )

    assert metrics["overall"] == 1.0
    assert {sample["sample_namespace"] for sample in samples} == {"freeform", "multichoice"}
    assert len(judge.requests) == 2


def test_wildbench_preserves_published_truncation_and_weighted_tags(monkeypatch):
    row = {
        "session_id": "session-1",
        "conversation_input": [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "question"}],
        "checklist": ["be right"],
        "primary_tag": "Math",
        "secondary_tags": [],
    }
    monkeypatch.setattr(wildbench_source, "load_dataset", lambda *_args, **_kwargs: [row])
    source = wildbench_source.WildBenchBenchmark(config=wildbench_source.WildBenchConfig(max_words_to_eval=1))
    candidate = RoutedEndpoint("one two")
    judge = RoutedEndpoint('{"score": "7"}')

    metrics, samples = evaluate_source(
        source,
        SourceOptions("WildBench", SourceRef("fixture")),
        candidate,
        judges={"WildBench": judge},
    )

    assert metrics["score"] == 7.0
    assert metrics["avg_output_length"] == 0.0
    assert metrics["task_categorized_scores"] == {"Math & Data Analysis": 4.0}
    assert "one... (truncated)" in judge.requests[0].prompt
    assert samples[0]["adjusted_score"] == 4.0


def _scored_item(task, source_id, grade, tags):
    spec = EvalSpec(
        task=task, source=SourceRef("fixture"), source_id=source_id, ordinal=0, requests=(),
        grader=GraderSpec("fixture"), tags=tags,
    )
    return ItemResult(spec, (), grade, {"sample_id": str(source_id)})


def test_mixeval_reducer_preserves_split_rounding_and_public_keys():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(name="rounded_split_mean"))
    for source_id, split, score in [("freeform-1", "freeform", 0.3333), ("freeform-2", "freeform", 1.0), ("multi", "multichoice", 0.0)]:
        metrics.add(
            _scored_item(
                "MixEval", source_id, Grade({"accuracy": score}),
                {"split": split, "benchmark_split": f"BBH|{split}"},
            )
        )

    assert metrics.finish("MixEval")["results"]["MixEval"] == {
        "overall": pytest.approx(0.44466666666666665),
        "BBH": pytest.approx(0.44466666666666665),
    }


def test_mtbench_excludes_invalid_turns_and_weights_partial_averages():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(trial_metric=None))
    first = turn_ratings(None, ["[[10]]", "[[-1]]"])
    second = turn_ratings(None, ["[[2]]", "[[2]]"])
    metrics.add(_scored_item("MTBench", "first", first, {}))
    metrics.add(_scored_item("MTBench", "second", second, {}))

    reduced = metrics.finish("MTBench")["results"]["MTBench"]
    assert reduced["Turn 1"] == 6.0
    assert reduced["Turn 2"] == 2.0
    assert reduced["Average"] == pytest.approx(14 / 3)


def test_mtbench_parser_preserves_fastchat_rating_selection():
    grade = turn_ratings(None, ["[[-1]] then [[2]]", "[11]"])
    assert grade.metrics == {"Turn 1": 2.0, "Turn 2": 11.0, "Average": 6.5}


def test_mtbench_all_invalid_judgments_do_not_create_aggregate_scores():
    metrics = SourceMetrics(repeated=False, policy=ReducerSpec(trial_metric=None))
    metrics.add(_scored_item("MTBench", "invalid", turn_ratings(None, ["[[-1]]", "invalid"]), {}))
    assert metrics.finish("MTBench")["results"]["MTBench"] == {}
