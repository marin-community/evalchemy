"""Parity at the source/runtime boundary for judge calls and instruction weights."""

import asyncio
import json

import pytest

from eval.chat_benchmarks.IFEval.eval_instruct import IFEvalBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import evaluate_source
from eval.runtime.checkpoints import input_digest


class FixedEndpoint:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def execute(self, request):
        self.requests.append(request)
        return self.response


class MemoryCheckpoints:
    def __init__(self):
        self.rows = {}

    def records(self):
        yield from sorted(self.rows.values(), key=lambda row: (row.position, input_digest(row.unit)))

    def record(self, row):
        self.rows[row.position, input_digest(row.unit)] = row

    def finalize(self):
        pass


def test_simpleqa_restores_candidate_and_judge_operations_without_repaying():
    source = SimpleQABenchmark(debug=True)
    options = SourceOptions("SimpleQA", SourceRef(source.data_file), limit=1)
    candidate = FixedEndpoint("An answer")
    judge = FixedEndpoint("A")
    checkpoints = MemoryCheckpoints()
    metrics, samples = evaluate_source(source, options, candidate, judges={"SimpleQA": judge},
                                       checkpoints={"SimpleQA": checkpoints})
    assert metrics["accuracy"] == 1.0
    assert samples[0]["judge_label"] == "correct"
    assert "An answer" in judge.requests[0].prompt
    candidate_again, judge_again = FixedEndpoint("wrong"), FixedEndpoint("B")
    resumed_metrics, resumed_samples = evaluate_source(
        source, options, candidate_again, judges={"SimpleQA": judge_again}, checkpoints={"SimpleQA": checkpoints},
    )
    assert resumed_metrics == metrics
    assert resumed_samples == samples
    assert candidate_again.requests == judge_again.requests == []


def test_instruction_reducer_weights_by_constraints_instead_of_averaging_prompt_rates(tmp_path):
    records = []
    for key, keywords in enumerate((["apple"], ["apple", "banana", "orange"])):
        records.append({"key": key, "prompt": "Say " + ", ".join(keywords),
                        "instruction_id_list": ["keywords:existence"] * len(keywords),
                        "kwargs": [{"keywords": [word]} for word in keywords]})
    (tmp_path / "input_data.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    source = IFEvalBenchmark(data_dir=str(tmp_path), start_idx=0, end_idx=2)
    metrics, samples = evaluate_source(source, SourceOptions("IFEval", SourceRef("fixture")), FixedEndpoint("apple"))
    assert metrics["prompt_level_strict"] == 0.5
    assert metrics["instruction_level_strict"] == 0.5
    assert metrics["instruction-level"] == 0.5
    assert sorted(sample["instruction_level_strict"] for sample in samples) == [1 / 3, 1.0]


def test_explicit_trials_share_binary_pass_at_k_reduction():
    from eval.contracts.eval_spec import ReducerSpec
    from eval.sources.common import text_specs

    class Source:
        REDUCER = ReducerSpec()

        def iter_specs(self, options):
            return text_specs(
                [{"id": "a"}, {"id": "b"}], options,
                prompt=lambda row: row["id"], target=lambda row: "yes", grader="exact_match",
                generation={"do_sample": True},
                seed_repeats=True,
            )

    class Backend:
        async def execute(self, request):
            return "yes" if request.generation["seed"] < 2 else "no"

    from eval.execution.engine import CollectSamples, Evaluator
    from eval.execution.grading import Graders
    from eval.execution.metrics import SourceMetrics
    from eval.graders.specs import GRADERS

    source = Source()
    options = SourceOptions("trials", SourceRef("fixture"), limit=1, num_samples=4, seed=0)
    metrics = SourceMetrics(repeated=False, pass_at_k=(1, 2))
    sink = CollectSamples()
    grading = Graders(GRADERS)
    async def run():
        engine = Evaluator(Backend(), grading, sink, workers=3)
        async for result in engine.run(source.iter_specs(options)):
            assert result.error is None
            metrics.add(result.result)
    try:
        asyncio.run(run())
    finally:
        grading.close()
    assert metrics.finish("trials")["results"]["trials"] == {"pass@1": 0.5, "pass@2": pytest.approx(5 / 6)}
    assert len(sink.samples["trials"]) == 4
    assert {record["source_id"] for record in sink.samples["trials"]} == {"a"}
