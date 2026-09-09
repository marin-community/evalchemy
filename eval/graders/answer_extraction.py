"""Shared answer-extraction contracts for generated benchmark responses."""

from collections.abc import Sequence
from typing import TypedDict

from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string, remove_boxed

from eval.generation_stops import END_OF_TURN_SEQUENCES, truncate_at_stop


class AnswerExtractionError(ValueError):
    """A generated response cannot be converted into a scoreable answer."""


class EmptyResponseError(AnswerExtractionError):
    """The model response was empty."""


class MissingAnswerError(AnswerExtractionError):
    """The response contained no answer in the required syntax."""


class ExtractionFailure(TypedDict):
    """JSON representation of an answer-extraction error."""

    type: str
    message: str


def extraction_failure(exc: AnswerExtractionError) -> ExtractionFailure:
    """Return the stable artifact fields for an extraction error."""
    return {"type": type(exc).__name__, "message": str(exc)}


def _first_boxed_only_string(response: str) -> str | None:
    box_starts = [index for marker in ("\\boxed", "\\fbox") if (index := response.find(marker)) >= 0]
    if not box_starts:
        return None
    start = min(box_starts)
    later_box_starts = [
        index
        for marker in ("\\boxed", "\\fbox")
        if (index := response.find(marker, start + 1)) >= 0
    ]
    end = min(later_box_starts) if later_box_starts else len(response)
    return last_boxed_only_string(response[start:end])


def extract_boxed_answer(
    response: str, stops: Sequence[str] = END_OF_TURN_SEQUENCES
) -> str:
    """Extract the first complete answer box or raise a typed error."""
    if not response.strip():
        raise EmptyResponseError("response is empty")
    relevant_response = truncate_at_stop(response, stops)
    boxed = _first_boxed_only_string(relevant_response)
    if boxed is None:
        raise MissingAnswerError("response contains no boxed answer before the task boundary")
    try:
        return remove_boxed(boxed)
    except (AssertionError, TypeError, ValueError) as exc:
        raise MissingAnswerError("response contains a malformed boxed answer") from exc
