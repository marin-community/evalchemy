"""ZeroEval source routing and canonical answer interpretation."""

import json

import pytest
from datasets import Dataset

from eval.chat_benchmarks.zeroeval.eval_instruct import ZeroEvalBenchmark
from eval.chat_benchmarks.zeroeval.src.evaluation.crux_eval import grade_model as grade_literal
from eval.chat_benchmarks.zeroeval.src.evaluation.math_eval import grade_model as grade_math
from eval.contracts.eval_spec import SourceRef
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, evaluate_source


def _source(monkeypatch, task, target):
    source = ZeroEvalBenchmark(tasks=[task])
    monkeypatch.setattr(
        "eval.chat_benchmarks.zeroeval.src.task_configs.load_dataset",
        lambda *args, **kwargs: Dataset.from_list([{
            "id": "zero", "question": "Choose an answer", "problem": "Choose an answer",
            "choices": ["first", "second", "third", "fourth"], "answer": target,
        }]),
    )
    return source, SourceOptions("zeroeval", SourceRef("fixture"))


@pytest.mark.parametrize("answer,target,expected", [
    ("1/4+1/4", "0.5", 100.0),
    ("1/2+1", "1", 0.0),
    ("-(1/4 + 1/4)", "-0.5", 100.0),
    ("2**3/4", "2", 100.0),
    ("1/2", "0.5", 100.0),
    ("$1,200 miles", "1200", 100.0),
])
def test_zeroeval_numeric_answers_preserve_native_arithmetic(monkeypatch, answer, target, expected):
    response = json.dumps({"answer": answer, "reasoning": "computed"})
    source, options = _source(monkeypatch, "math-l5", target)

    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))
    native = grade_math("fixture", [{"id": "zero", "answer": target, "output": [response]}])[0]

    assert float(native["matched"]) * 100 == expected
    assert metrics["accuracy"] == expected
    assert samples[0]["accuracy"] == expected


@pytest.mark.parametrize("answer,target,expected", [("B", "B", 100.0), ("B", "A", 0.0), ("1/2+1", "1", 0.0)])
def test_zeroeval_mmlu_redux_keeps_literal_grading(monkeypatch, answer, target, expected):
    response = json.dumps({"answer": answer, "reasoning": "selected"})
    source, options = _source(monkeypatch, "mmlu-redux", target)

    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))
    native = grade_literal("fixture", [{"id": "zero", "answer": target, "output": [response]}])[0]

    assert float(native["matched"]) * 100 == expected
    assert metrics["accuracy"] == expected
    assert samples[0]["accuracy"] == expected


def test_zeroeval_arithmetic_does_not_execute_python(monkeypatch, tmp_path):
    marker = tmp_path / "executed"
    answer = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')/2"
    source, options = _source(monkeypatch, "math-l5", "0.5")

    metrics, _ = evaluate_source(source, options, FixedEndpoint(json.dumps({"answer": answer})))

    assert metrics["accuracy"] == 0.0
    assert not marker.exists()
