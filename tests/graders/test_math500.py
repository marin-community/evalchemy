import pytest

from eval.chat_benchmarks.MATH500.eval_instruct import MATH500Benchmark
from eval.completion_response import (
    CompletionContentPolicy,
    CompletionResponse,
    CompletionText,
)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ("11111111100", r"11,\! 111,\! 111,\! 100"),
        (r"\textbf{(B)}", r"\text{(B)}"),
        ("0.09", r"\frac{9}{100}"),
        ("58500", "58,500"),
        ("ellipse", r"\text{ellipse}"),
        (r"3,\ 5,\ 7", "3, 5, 7"),
        (
            r"\begin{pmatrix} \frac{1}{5} \\ -\frac{18}{5} \end{pmatrix}",
            r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
        ),
        (r"5r^{5}", r"5r^5"),
        (r"\frac{9a+11}{20}", r"\frac{11+9a}{20}"),
    ],
)
def test_math500_scores_reported_equivalent_answer_forms(candidate, reference):
    example = {"answer": reference, "model_answer": candidate}

    results = MATH500Benchmark().evaluate_responses({"examples": [example]})

    assert results["accuracy"] == 1.0
    assert example["sample_metrics"] == {"accuracy": 1.0}


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        (
            r"<|start_think|>Mark your solution with \boxed Answer: 63."
            r"<|end_think|>\boxed{63}",
            "63",
        ),
        (r"Reasoning. \boxed{17}", "17"),
    ],
)
def test_math500_extracts_final_box_without_reasoning_contamination(output, answer):
    assert MATH500Benchmark().extract_answer(output) == answer


@pytest.mark.parametrize(("content", "answer"), [(r"\boxed{63}", "63"), (None, "")])
def test_math500_extracts_only_structured_final_content(content, answer):
    response = CompletionResponse(
        content=content,
        reasoning_content=r"The prompt says \boxed Answer: 17.",
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

    assert MATH500Benchmark().extract_answer(output) == answer


def test_math500_distinguishes_structural_commas_from_grouping_commas():
    example = {"answer": "3, 5, 7", "model_answer": "357"}

    results = MATH500Benchmark().evaluate_responses({"examples": [example]})

    assert results["accuracy"] == 0.0


def test_math500_pass_at_k_uses_equivalent_answer_forms():
    benchmark = MATH500Benchmark(num_samples=2, pass_at_k=[1, 2])
    results = {
        "pass_at_k": True,
        "examples": [
            {
                "answer": r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}",
                "model_answers": [
                    "wrong",
                    r"\begin{pmatrix} \frac{1}{5} \\ -\frac{18}{5} \end{pmatrix}",
                ],
            }
        ],
    }

    scored = benchmark.evaluate_responses(results)

    assert scored["num_correct"] == [1]
    assert scored["pass@2"] == 1.0
