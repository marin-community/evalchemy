"""Offline source-to-judge fixtures for migrated judge-backed benchmarks."""

from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.WildBench import eval_instruct as wildbench_source
from eval.contracts.eval_spec import SourceRef
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

    assert metrics["accuracy"] == 1.0
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
