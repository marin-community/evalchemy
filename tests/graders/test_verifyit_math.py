"""Native Math mode composition and actual custom benchmark failure boundaries."""

import importlib

import pytest

from eval.graders.answer_equivalence import verifyit_math_answers_equivalent
from verifyit.grade import InvalidTask


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        (r"\frac{1}{2}", "0.5", True),
        (r"\frac{1}{2}", "0.6", False),
        ("2*x", "x+x", True),
        ("2*x", "3*x", False),
        ("42", r"\boxed{42}", True),
        ("42", r"\boxed{42", False),
        ("42", "reasoning 99\n42", True),
        ("42", "42 elephants and 7 giraffes", False),
        ("42", "", False),
        ("10", r"10\\%", False),
        ("10", "10%", False),
        ("10", r"10\%", False),
        (r"10\%", r"10\%", True),
        ("12", r"12^{\\text{th}}\\text{ grade}", False),
        ("12", r"12^{\text{th}}\text{ grade}", False),
        ("12", r"12^{\mathrm{th}} \text{ grade}", False),
    ],
)
def test_native_math_compares_final_expressions_without_source_fallback(reference, candidate, expected):
    assert verifyit_math_answers_equivalent(candidate, [reference]) is expected


@pytest.mark.parametrize("candidate", ["2", None])
def test_later_invalid_reference_cannot_be_hidden_by_an_earlier_match(candidate):
    with pytest.raises(InvalidTask):
        verifyit_math_answers_equivalent(candidate, ["2", r"\frac{"])


@pytest.mark.parametrize("name", ["AIME24", "AIME25", "MATH500"])
def test_custom_math_invalid_reference_aborts_before_sample_metrics(name):
    module = importlib.import_module(f"eval.chat_benchmarks.{name}.eval_instruct")
    benchmark = getattr(module, name + "Benchmark")(verifyit_enabled=True)
    reference_key = "expected_answer" if name == "AIME24" else "answer"
    examples = [
        {reference_key: reference, "model_answer": "2", "model_answers": ["2"] * 10} for reference in ["2", None]
    ]
    with pytest.raises(InvalidTask):
        benchmark.evaluate_responses({"examples": examples})
    assert all("sample_metrics" not in example for example in examples)


@pytest.mark.parametrize("name", ["AIME24", "AIME25", "MATH500"])
def test_opt_in_rejects_broken_box_while_default_keeps_source_behavior(name):
    module = importlib.import_module(f"eval.chat_benchmarks.{name}.eval_instruct")
    constructor = getattr(module, name + "Benchmark")
    reference_key = "expected_answer" if name == "AIME24" else "answer"
    metrics = []
    for options in ({}, {"verifyit_enabled": False}, {"verifyit_enabled": True}):
        benchmark = constructor(**options)
        example = {reference_key: "42", "model_answer": r"\boxed{42", "model_answers": [r"\boxed{42"] * 10}
        result = benchmark.evaluate_responses({"examples": [example]})
        metrics.append(result["accuracy"] if name == "MATH500" else result["accuracy_avg"])
    assert metrics == [1.0, 1.0, 0.0]


@pytest.mark.parametrize("name", ["AIME24", "AIME25", "MATH500"])
@pytest.mark.parametrize("candidate", [None, 42])
def test_absent_or_nontext_extracted_candidate_scores_zero_without_invalidating_task(name, candidate):
    module = importlib.import_module(f"eval.chat_benchmarks.{name}.eval_instruct")
    benchmark = getattr(module, name + "Benchmark")(verifyit_enabled=True)
    reference_key = "expected_answer" if name == "AIME24" else "answer"
    example = {reference_key: "42", "model_answer": candidate, "model_answers": [candidate] * 10}
    result = benchmark.evaluate_responses({"examples": [example]})
    accuracy = result["accuracy"] if name == "MATH500" else result["accuracy_avg"]
    assert accuracy == 0
