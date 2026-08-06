"""Extract contracted short answers from chat completions."""

from __future__ import annotations

import re

from eval.generation_stops import truncate_at_stop

_INVALID_ANSWER = "[invalid]"
_ANSWER_MARKER = re.compile(
    r"(?im)^[ \t]*(?:(?:final)[ \t]+)?(?:answer|a)[ \t]*:[ \t]*(?P<answer>\S[^\r\n]*)"
)


def extract_marked_short_answer(response: object) -> str:
    """Return the last line-level marked answer or an invalid scoring value."""
    if not isinstance(response, str):
        return _INVALID_ANSWER
    matches = list(_ANSWER_MARKER.finditer(truncate_at_stop(response)))
    return matches[-1]["answer"].strip() if matches else _INVALID_ANSWER


def marked_short_answer_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract one contracted short answer from each lm-eval completion."""
    del docs
    return [
        [extract_marked_short_answer(response) for response in responses]
        for responses in resps
    ]
