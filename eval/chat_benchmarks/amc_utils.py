import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional, Sequence

try:
    from lm_eval.tasks.hendrycks_math.utils import is_equiv as _hendrycks_is_equiv
except Exception:  # pragma: no cover - optional dependency in lightweight tests
    _hendrycks_is_equiv = None


def extract_boxed_answer(output: Any) -> str:
    """Extract the final boxed answer, returning an empty string on malformed output."""
    text = "" if output is None else str(output)
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return ""

    prefix_len = len("\\boxed") if text.startswith("\\boxed", start) else len("\\fbox")
    idx = start + prefix_len
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return ""

    if text[idx] == "{":
        return _extract_balanced_brace_content(text, idx).strip()

    match = re.match(r"([^\s$.]+)", text[idx:])
    return match.group(1).strip() if match else ""


def normalize_amc_answer(answer: Any) -> str:
    """Normalize AMC answers without changing their mathematical meaning."""
    text = "" if answer is None else str(answer)
    text = text.strip()

    text = _strip_math_wrappers(text)
    text = _strip_boxed_wrapper(text)
    text = _strip_math_wrappers(text)

    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\;": "",
        "\\!": "",
        "\\ ": "",
        "{:}": ":",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", "", text)
    text = _strip_outer_braces(text)

    if re.fullmatch(r"[A-Za-z]", text):
        return text.lower()
    return text


def amc_answer_is_correct(
    expected: Any, predicted: Any, accepted_answers: Optional[Sequence[Any]] = None
) -> bool:
    """Score an AMC prediction with symbolic equivalence plus exact normalized fallback."""
    candidates = [expected]
    if accepted_answers:
        candidates.extend(accepted_answers)

    predicted_text = "" if predicted is None else str(predicted)
    predicted_norm = normalize_amc_answer(predicted_text)
    for candidate in candidates:
        expected_text = "" if candidate is None else str(candidate)
        expected_norm = normalize_amc_answer(expected_text)

        if expected_norm == predicted_norm:
            return True
        if _decimal_equal(expected_norm, predicted_norm):
            return True
        if _safe_is_equiv(expected_text, predicted_text):
            return True
        if _safe_is_equiv(expected_norm, predicted_norm):
            return True
    return False


def _safe_is_equiv(expected: str, predicted: str) -> bool:
    if _hendrycks_is_equiv is None:
        return False
    try:
        return bool(_hendrycks_is_equiv(expected, predicted))
    except Exception:
        return False


def _decimal_equal(expected: str, predicted: str) -> bool:
    try:
        return Decimal(expected) == Decimal(predicted)
    except (InvalidOperation, ValueError):
        return False


def _extract_balanced_brace_content(text: str, open_idx: int) -> str:
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx]
    return ""


def _strip_math_wrappers(text: str) -> str:
    wrappers = (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
    changed = True
    while changed:
        changed = False
        text = text.strip()
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left) : -len(right)].strip()
                changed = True
    return text


def _strip_boxed_wrapper(text: str) -> str:
    text = text.strip()
    prefixes = ("\\boxed", "\\fbox")
    for prefix in prefixes:
        if not text.startswith(prefix):
            continue
        rest = text[len(prefix) :].strip()
        if rest.startswith("{") and rest.endswith("}") and _balanced_outer_braces(rest):
            return rest[1:-1].strip()
    return text


def _strip_outer_braces(text: str) -> str:
    while text.startswith("{") and text.endswith("}") and _balanced_outer_braces(text):
        text = text[1:-1].strip()
    return text


def _balanced_outer_braces(text: str) -> bool:
    depth = 0
    for idx, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0
