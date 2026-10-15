"""Normalized source scores and samples agree with the pinned task definitions."""

import asyncio
import json

import pytest
import yaml
from lm_eval.api.model import LM
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from eval.contracts.eval_spec import RequestKind
from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Graders
from eval.sources.lm_eval import LMEvalMetrics, LMEvalSource
from eval.sources.common import SourceOptions
from eval.contracts.eval_spec import SourceRef
from eval.graders.specs import GRADERS
from eval.execution.metrics import SourceMetrics


class _FixtureLM(LM):
    def generate_until(self, requests):
        return ["2" if index % 3 < 2 else "wrong" for index, _ in enumerate(requests)]

    def loglikelihood(self, requests):
        return [(-1.0 if request.args[1].strip() == "2" else -10.0, True) for request in requests]

    def loglikelihood_rolling(self, requests):
        return [-float(len(request.args[0])) for request in requests]


class _FixtureBackend:
    def __init__(self):
        self.generations = 0

    async def execute(self, request):
        if request.kind is RequestKind.GENERATE:
            self.generations += 1
            return "2" if (self.generations - 1) % 3 < 2 else "wrong"
        if request.kind is RequestKind.LIKELIHOOD:
            return (-1.0 if request.continuation.strip() == "2" else -10.0, True)
        return -float(len(request.prompt))


@pytest.mark.parametrize("kind", ["generate_until", "multiple_choice", "loglikelihood_rolling"])
def test_lm_eval_iterator_matches_native_scoring_with_filters_and_choices(tmp_path, kind):
    (tmp_path / "data.jsonl").write_text(json.dumps({"question": "1+1?", "answer": "2", "choices": ["1", "2"]}) + "\n")
    config = {
        "task": "spec_probe", "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(tmp_path / "data.jsonl")}},
        "test_split": "test", "output_type": kind,
        "doc_to_text": "{{question}}", "doc_to_target": "{{answer}}",
        "num_fewshot": 0,
    }
    if kind == "multiple_choice":
        config.update(doc_to_choice="choices", doc_to_target=1, metric_list=[{"metric": "acc"}, {"metric": "acc_norm"}])
    elif kind == "generate_until":
        config.update(
            repeats=3, metric_list=[{"metric": "exact_match"}],
            filter_list=[
                {"name": "first", "filter": [{"function": "take_first"}]},
                {"name": "majority", "filter": [{"function": "majority_vote"}, {"function": "take_first"}]},
            ],
        )
    else:
        config.update(metric_list=[{"metric": name} for name in ("word_perplexity", "byte_perplexity", "bits_per_byte")])
    (tmp_path / "spec_probe.yaml").write_text(yaml.safe_dump(config))
    baseline = evaluator.simple_evaluate(
        model=_FixtureLM(), tasks=["spec_probe"],
        task_manager=TaskManager(include_path=str(tmp_path)), limit=1, bootstrap_iters=0,
    )
    loaded = TaskManager(include_path=str(tmp_path)).load(["spec_probe"])
    source = LMEvalSource("spec_probe", loaded["tasks"]["spec_probe"], 1, 0, {}, 1234)
    reducer = LMEvalMetrics(loaded["tasks"], loaded["groups"], bootstrap_iters=0)
    samples = CollectSamples()
    engine = Evaluator(_FixtureBackend(), Graders({source.grader_name: source.grader()}), samples, workers=2)

    async def run():
        async for result in engine.run(source):
            assert result.error is None
            reducer.add(result.result)

    asyncio.run(run())
    actual = reducer.finish("spec_probe")
    assert actual["results"] == baseline["results"]
    assert actual["n-samples"] == baseline["n-samples"]
    record = samples.samples["spec_probe"][0]
    for name in ("resps", "arguments", "filtered_resps", "target", "doc_hash", "prompt_hash", "target_hash"):
        assert record[name] == baseline["samples"]["spec_probe"][0][name]
    if kind == "generate_until":
        assert {variant["filter"] for variant in record["filter_variants"]} == {"first", "majority"}


@pytest.mark.parametrize("program, expected", [("print(input())", 1.0), ("print(float(input()))", 0.0)])
def test_codeelo_source_uses_common_dispatch_and_preserves_exact_output(monkeypatch, program, expected):
    from eval.chat_benchmarks.CodeElo import eval_instruct as codeelo

    problem = {
        "id": "example", "title": "Echo", "description": "Echo an integer",
        "input": "One integer", "output": "The same integer", "rating": 800,
        "time_limit_ms": 1000, "interaction": "", "note": "", "examples": [("2", "2-\n")],
    }
    monkeypatch.setattr(codeelo, "load_dataset", lambda *_args, **_kwargs: {"test": [problem]})
    source = codeelo.CodeEloBenchmark()
    options = SourceOptions("CodeElo", SourceRef("Qwen/CodeElo"))

    class Model:
        async def execute(self, request):
            return f"```python\n{program}\n```"

    samples = CollectSamples()
    metrics = SourceMetrics(repeated=True, policy=source.REDUCER)
    grading = Graders(GRADERS)
    engine = Evaluator(Model(), grading, samples, workers=2)

    async def run():
        async for result in engine.run(source.iter_specs(options)):
            assert result.error is None
            metrics.add(result.result)

    try:
        asyncio.run(run())
    finally:
        grading.close()
    assert metrics.finish("CodeElo")["results"]["CodeElo"]["accuracy_avg"] == expected
    assert sorted(record["sample_repeat"] for record in samples.samples["CodeElo"]) == [0, 1, 2]
    assert len(samples.samples["CodeElo"]) == 3
