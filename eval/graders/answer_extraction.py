"""Shared answer-extraction contracts for generated benchmark responses."""

from collections.abc import Sequence

from lm_eval.tasks.hendrycks_math.utils import remove_boxed

from eval.generation_stops import END_OF_TURN_SEQUENCES, truncate_at_stop


class AnswerExtractionError(ValueError):
    """A generated response cannot be converted into a scoreable answer."""


class EmptyResponseError(AnswerExtractionError):
    """The model response was empty."""


class MissingAnswerError(AnswerExtractionError):
    """The response contained no answer in the required syntax."""


def _first_boxed_only_string(response: str) -> str | None:
    box_starts = [index for marker in ("\\boxed", "\\fbox") if (index := response.find(marker)) >= 0]
    if not box_starts:
        return None
    start = min(box_starts)
    if response.startswith("\\boxed ", start):
        return response[start:].split("$", maxsplit=1)[0]

    depth = 0
    opened = False
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
            opened = True
        elif response[index] == "}":
            depth -= 1
            if opened and depth == 0:
                return response[start : index + 1]
    return None


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
