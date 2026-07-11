"""RIMO-N final-answer benchmark."""

from typing import Any, Dict

from eval.chat_benchmarks.final_answer_math import (
    FinalAnswerMathBenchmark,
    GradeOutcome,
    exact_string_grade,
)


class RIMONBenchmark(FinalAnswerMathBenchmark):
    DEFAULT_DATA_FILE = "eval/chat_benchmarks/RIMON/data/rimo_n.jsonl"
    EXPECTED_ROWS = 335
    REQUIRED_FIELDS = ("problem_id", "problem", "answer", "type")
    ID_FIELD = "problem_id"
    GROUP_FIELD = "type"

    def grade_answer(self, gold: str, prediction: str, example: Dict[str, Any]) -> GradeOutcome:
        return exact_string_grade(gold, prediction)

    def group_labels(self, example: Dict[str, Any]) -> Dict[str, str]:
        labels = super().group_labels(example)
        labels["answer_class"] = "binary" if str(example["answer"]).strip() in {"0", "1"} else "non_binary"
        return labels
