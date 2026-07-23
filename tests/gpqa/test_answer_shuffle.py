"""Regression tests for GPQA-Diamond answer placement."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.chat_benchmarks.GPQADiamond.eval_instruct import GPQADiamondBenchmark


def _question(index: int) -> dict[str, str]:
    return {
        "Question": f"Question {index}",
        "Correct Answer": "correct",
        "Incorrect Answer 1": "wrong one",
        "Incorrect Answer 2": "wrong two",
        "Incorrect Answer 3": "wrong three",
    }


def test_answer_positions_vary_reproducibly_across_questions():
    benchmark = GPQADiamondBenchmark.__new__(GPQADiamondBenchmark)
    questions = [_question(index) for index in range(32)]

    first = [benchmark.generate_multiple_choice_answers(question)[1] for question in questions]
    second = [benchmark.generate_multiple_choice_answers(question)[1] for question in questions]

    assert set(first) == {"A", "B", "C", "D"}
    assert first == second
