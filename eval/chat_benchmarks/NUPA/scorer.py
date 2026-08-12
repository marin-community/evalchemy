"""Clean-room implementation of the official NUPA text metrics.

The implementation reproduces the observable behavior of Number Cookbook's
text evaluator without copying its GPL-licensed source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

INTEGER = "Integer"
FLOAT = "Float"
FRACTION = "Fraction"
SCIENTIFIC = "ScientificNotation"

_ANSWER_MARKER_RE = re.compile(r"(?i)^(?:the\s+answer\s+is|so\s+the\s+answer\s+is)\s+")
_ANSWER_PATTERNS = {
    INTEGER: re.compile(r"^\d+"),
    FLOAT: re.compile(r"^\d+\.\d+"),
    FRACTION: re.compile(r"^\d+/\d+"),
    SCIENTIFIC: re.compile(r"^\d+\.\d+[eE][+-]?\d+"),
}
_READ_FROM_RIGHT = {
    INTEGER: (True,),
    FLOAT: (True, False),
    FRACTION: (True, True),
    SCIENTIFIC: (True, False, True),
}


@dataclass(frozen=True)
class ExampleScore:
    exact_match: float
    digit_match: float
    dlength: float
    format_valid: float
    no_answer: float


def extract_answer(text: object, answer_format: str) -> str | None:
    """Extract an answer at the start of a direct-answer completion."""
    if answer_format not in _ANSWER_PATTERNS:
        raise ValueError(f"Unsupported NUPA answer format: {answer_format}")
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    stripped = _ANSWER_MARKER_RE.sub("", stripped, count=1)
    match = _ANSWER_PATTERNS[answer_format].match(stripped)
    if match is None:
        return None
    return match.group().replace("+", "").replace("-", "").replace("E", "e")


def normalize_answer(answer: str | None, answer_format: str) -> str | None:
    """Validate an extracted answer without changing its digit representation."""
    if answer is None:
        return None
    extracted = extract_answer(answer, answer_format)
    if extracted is None or extracted != answer.replace("+", "").replace("-", "").replace("E", "e"):
        return None
    return extracted


def score_prediction(prediction: str | None, gold: str, answer_format: str) -> ExampleScore:
    """Score a completion with the official text-evaluator semantics."""
    extracted = extract_answer(prediction, answer_format)
    gold_parts = _digit_parts(gold, answer_format)
    prediction_parts = _digit_parts(extracted or "", answer_format)
    format_valid = extracted is not None
    return ExampleScore(
        exact_match=1.0 if format_valid and prediction_parts == gold_parts else 0.0,
        digit_match=_digit_match(prediction_parts, gold_parts, answer_format),
        dlength=float(abs(_total_part_length(prediction_parts) - _total_part_length(gold_parts))),
        format_valid=1.0 if format_valid else 0.0,
        no_answer=0.0 if format_valid else 1.0,
    )


def length_bucket(digit: int, *, max_digit: int | None = None) -> str:
    """Return the NUPA S/M/L/XL interval for a digit length."""
    if max_digit is None:
        max_digit = 20 if digit <= 20 else 100
    if max_digit <= 20:
        if digit <= 4:
            return "S"
        if digit <= 8:
            return "M"
        if digit <= 14:
            return "L"
        return "XL"
    if digit <= 10:
        return "S"
    if digit <= 20:
        return "M"
    if digit <= 60:
        return "L"
    return "XL"


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _digit_parts(answer: str, answer_format: str) -> tuple[str, ...]:
    separators = {
        INTEGER: (),
        FLOAT: (".",),
        FRACTION: ("/",),
        SCIENTIFIC: (".", "e"),
    }
    if answer_format not in separators:
        raise ValueError(f"Unsupported NUPA answer format: {answer_format}")
    parts = [answer]
    for separator in separators[answer_format]:
        next_parts = []
        for part in parts:
            next_parts.extend(part.replace("E", "e").split(separator, 1))
        parts = next_parts
    expected_parts = len(separators[answer_format]) + 1
    if len(parts) != expected_parts:
        return tuple("" for _ in range(expected_parts))
    return tuple("".join(character for character in part if character.isdigit()) for part in parts)


def _digit_match(prediction: tuple[str, ...], gold: tuple[str, ...], answer_format: str) -> float:
    correct = 0
    total = _total_part_length(gold)
    for prediction_part, gold_part, align_right in zip(prediction, gold, _READ_FROM_RIGHT[answer_format], strict=True):
        if align_right:
            prediction_part = prediction_part[::-1]
            gold_part = gold_part[::-1]
        correct += sum(predicted == expected for predicted, expected in zip(prediction_part, gold_part))
    return correct / total if total else 0.0


def _total_part_length(parts: Iterable[str]) -> int:
    return sum(len(part) for part in parts)
