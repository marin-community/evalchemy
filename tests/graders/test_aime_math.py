import pytest

from eval.chat_benchmarks.AIME24.eval_instruct import AIME24Benchmark
from eval.chat_benchmarks.AIME25.eval_instruct import AIME25Benchmark
from eval.completion_response import (
    CompletionContentPolicy,
    CompletionResponse,
    CompletionText,
)


@pytest.mark.parametrize(
    ("benchmark_type", "answer_field"),
    [(AIME24Benchmark, "expected_answer"), (AIME25Benchmark, "answer")],
)
def test_aime_equivalent_integer_forms_score_correctly(benchmark_type, answer_field):
    benchmark = benchmark_type()
    example = {answer_field: "25", "model_answers": [r"\frac{50}{2}"] * benchmark.n_repeat}

    result = benchmark.evaluate_responses({"examples": [example]})

    assert result["accuracy_avg"] == 1.0
    assert example["sample_metrics"]["accuracy"] == 1.0


@pytest.mark.parametrize("benchmark_type", [AIME24Benchmark, AIME25Benchmark])
def test_aime_boxed_answer_comes_from_final_content(benchmark_type):
    response = CompletionResponse(
        content=r"\boxed{25}",
        reasoning_content=r"The instruction quotes \boxed Answer: 17.",
        finish_reason="stop",
        usage=None,
        provider_metadata={},
        raw_choice={},
    )
    output = CompletionText(
        response.normalized_content(CompletionContentPolicy.COMBINE),
        response,
        CompletionContentPolicy.COMBINE,
    )

    assert benchmark_type.extract_answer(None, output) == "25"
    assert benchmark_type.extract_answer(None, r"<|end_think|>\boxed{25}") == "25"


@pytest.mark.parametrize("benchmark_type", [AIME24Benchmark, AIME25Benchmark])
def test_aime_reasoning_only_response_has_no_scoreable_answer(benchmark_type):
    response = CompletionResponse(
        content=None,
        reasoning_content=r"I considered \boxed{25} but have not finished.",
        finish_reason="length",
        usage=None,
        provider_metadata={},
        raw_choice={},
    )
    output = CompletionText(
        response.normalized_content(CompletionContentPolicy.COMBINE),
        response,
        CompletionContentPolicy.COMBINE,
    )

    assert benchmark_type.extract_answer(None, output) == ""
