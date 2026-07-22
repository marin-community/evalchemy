"""Extract short answers from verbose DROP completions before scoring."""

from __future__ import annotations

import re
from typing import Any

# Re-export the upstream DROP task functions unchanged so drop.yaml can reference them
# via `!function utils.<name>` and the scoring stays byte-identical to upstream.
from lm_eval.tasks.drop.utils import get_answers, parse_answer, process_docs, process_results  # noqa: F401

__all__ = [
    "get_answers",
    "parse_answer",
    "process_docs",
    "process_results",
    "extract_drop_short_answer",
    "drop_answer_extraction_filter",
]


# A number: optional sign / currency, digits with optional thousands separators, optional
# decimal, optional trailing percent. Kept deliberately simple — DROP normalization
# (`_fix_number`) re-parses via float(), so we only need to isolate the numeric token.
_NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# A date span in DROP gold order: optional day, month name, optional 4-digit year
# (e.g. "24 May 1993", "May 1993", "December"). Matches the `parse_answer` date join.
_DATE_RE = re.compile(
    rf"\b(?:(\d{{1,2}})\s+)?({_MONTHS})\.?(?:\s+(\d{{3,4}}))?\b",
    re.IGNORECASE,
)

# Explicit answer markers ("answer:", "the final answer is", "answer -"). We take the
# text AFTER the last such marker, which is where a model that does answer puts it.
_MARKER_RE = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=|-|—)?\s*",
)


def _clean_number(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").rstrip("%").strip()


def _date_in(span: str) -> str | None:
    m = _DATE_RE.search(span)
    if not m:
        return None
    day, month, year = m.group(1), m.group(2), m.group(3)
    parts = [p for p in (day, month, year) if p]
    return " ".join(parts).strip() or None


def _best_answer_in(span: str) -> str | None:
    span = span.strip()
    if not span:
        return None
    date = _date_in(span)
    if date:
        return date
    nums = _NUMBER_RE.findall(span)
    if nums:
        return _clean_number(nums[-1])
    return span


def extract_drop_short_answer(text: Any) -> str:
    """Extract a short DROP answer from a completion.

    Extraction prefers the tail after the last answer marker, then a date, the final
    number, and the last non-empty line. It does not inspect the gold answer.

    Args:
        text: Model completion. Non-string values produce an empty answer.

    Returns:
        The extracted answer text.
    """
    if not isinstance(text, str):
        return ""
    stripped = text.strip()
    if not stripped:
        return ""

    # 1. Explicit answer marker -> tail after the LAST marker, first line only.
    markers = list(_MARKER_RE.finditer(stripped))
    if markers:
        tail = stripped[markers[-1].end() :]
        tail_line = tail.splitlines()[0].strip() if tail.strip() else ""
        # Trim a trailing sentence terminator so "42." -> "42".
        cand = _best_answer_in(tail_line)
        if cand:
            return cand.rstrip(".").strip() or cand

    # 2. Date span anywhere.
    date = _date_in(stripped)
    if date:
        return date

    # 3. Last number anywhere.
    nums = _NUMBER_RE.findall(stripped)
    if nums:
        return _clean_number(nums[-1])

    # 4. Fallback: last non-empty line, else the whole text.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    return lines[-1] if lines else stripped


def drop_answer_extraction_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract short answers from lm-eval response samples.

    Args:
        resps: Response samples grouped by document.
        docs: Unused documents required by the custom-filter interface.

    Returns:
        Response samples with each completion replaced by its extracted answer.
    """
    return [[extract_drop_short_answer(r) for r in resp] for resp in resps]
