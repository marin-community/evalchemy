"""English OlymMATH-HARD final-answer benchmark."""

from typing import Any, Dict

from eval.chat_benchmarks.final_answer_math import (
    OLYMMATH_PROMPT,
    FinalAnswerMathBenchmark,
    GradeOutcome,
    olymmath_grade,
)


class OlymMATHHardBenchmark(FinalAnswerMathBenchmark):
    DEFAULT_DATA_FILE = "eval/chat_benchmarks/OlymMATHHard/data/olymmath_en_hard.jsonl"
    EXPECTED_ROWS = 100
    REQUIRED_FIELDS = ("unique_id", "problem", "answer", "subject")
    ID_FIELD = "unique_id"
    GROUP_FIELD = "subject"
    PROMPT_TEMPLATE = OLYMMATH_PROMPT
    PASS_TEMPERATURE = 0.6
    PASS_TOP_P = 0.95

    def grade_answer(self, gold: str, prediction: str, example: Dict[str, Any]) -> GradeOutcome:
        return olymmath_grade(gold, prediction)
