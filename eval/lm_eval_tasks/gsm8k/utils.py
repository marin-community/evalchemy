"""GSM8K answer extraction for OpenAI-compatible model responses."""

import re

from eval.generation_stops import GSM8K_STOP_SEQUENCES, truncate_at_stop

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_BOXED_ANSWER = re.compile(rf"\\boxed\{{\s*({_NUMBER})\s*\}}")
_FINAL_ANSWER = re.compile(rf"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_NUMBER})")
_NUMERIC_CANDIDATE = re.compile(_NUMBER)
_FALLBACK = "[invalid]"


def extract_gsm8k_flexible_answer(response: str) -> str:
    """Prefer final-answer syntax before falling back to the last number."""
    response = truncate_at_stop(response, GSM8K_STOP_SEQUENCES)
    for pattern in (_BOXED_ANSWER, _FINAL_ANSWER, _NUMERIC_CANDIDATE):
        matches = pattern.findall(response)
        if matches:
            return matches[-1].strip()
    return _FALLBACK


def gsm8k_flexible_extraction_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract one auditable flexible answer per GSM8K completion."""
    del docs
    return [
        [extract_gsm8k_flexible_answer(response) if isinstance(response, str) else _FALLBACK for response in responses]
        for responses in resps
    ]
