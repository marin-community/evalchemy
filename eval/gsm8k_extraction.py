"""GSM8K answer extraction for OpenAI-compatible model responses."""

import re
from collections.abc import Iterable, Sequence
from typing import Any

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_BOXED_ANSWER = re.compile(rf"\\boxed\{{\s*({_NUMBER})\s*\}}")
_FINAL_ANSWER = re.compile(rf"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_NUMBER})")
_NUMERIC_CANDIDATE = re.compile(_NUMBER)
_FALLBACK = "[invalid]"


@register_filter("gsm8k_flexible_extract")
class GSM8KFlexibleExtractFilter(Filter):
    """Prefer final-answer syntax before falling back to the last number."""

    @staticmethod
    def extract(response: str) -> str:
        for pattern in (_BOXED_ANSWER, _FINAL_ANSWER, _NUMERIC_CANDIDATE):
            matches = pattern.findall(response)
            if matches:
                return matches[-1].strip()
        return _FALLBACK

    def apply(self, resps: Iterable[Sequence[str]], docs: Sequence[dict[str, Any]]) -> list[list[str]]:
        del docs
        return [
            [self.extract(response) if isinstance(response, str) else _FALLBACK for response in responses]
            for responses in resps
        ]
