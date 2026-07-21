"""Answer-extraction filter for the DROP task (marin-community/evalchemy#31).

Why this exists
---------------
``drop`` is an ``lm-eval-harness`` ``generate_until`` task scored with the strict DROP
token-F1 metric. Token-F1 divides by the size of the *predicted* token bag, so when a
model runs WITHOUT a chat template (``local-completions`` — required so the MC /
loglikelihood tasks in the same run behave) the completion is a long, verbose
continuation and F1 collapses to ~0 even when the completion literally contains the
correct answer. Grid-wide this showed up as f1 ~= 0.002 for models scoring gsm8k
0.68-0.95.

The fix is a scoring-side ``filter_list`` that extracts the short answer span out of a
verbose completion BEFORE token-F1, so a correct-but-verbose completion scores
correctly. It is a pure post-processing step — it does NOT touch generation
(``max_gen_toks`` / ``until``) and is scoped to ``drop`` only.

This module is loaded two ways, both pointing at this same file:
  * ``import eval.lm_eval_tasks.drop.utils`` (tests / normal Python import), and
  * ``!function utils.<name>`` inside ``drop.yaml``, which lm-eval loads standalone by
    path from this directory.
So it must stay importable both as a package submodule and as a bare ``utils`` module.

The DROP metric functions (``process_docs`` / ``process_results``) are re-exported
verbatim from the upstream lm-eval task so the override YAML reuses the *identical*
scoring — only a ``filter_list`` is added on top.
"""

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
    """Strip currency / percent / thousands separators from a matched number token."""
    return tok.replace("$", "").replace(",", "").rstrip("%").strip()


def _date_in(span: str) -> str | None:
    """Return the first date span (in DROP gold order) found in ``span``, else None."""
    m = _DATE_RE.search(span)
    if not m:
        return None
    day, month, year = m.group(1), m.group(2), m.group(3)
    parts = [p for p in (day, month, year) if p]
    return " ".join(parts).strip() or None


def _best_answer_in(span: str) -> str | None:
    """Pick the most answer-like short token from ``span``.

    Preference order matches DROP answer-type frequency: a date span, else the last
    number, else (for short spans) the span itself. Returns None if ``span`` is empty.
    """
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
    """Extract a short DROP answer span from a (possibly verbose) completion.

    Strategy, in order (robust to no-template verbose output):
      1. If an explicit answer marker is present, take the text after the LAST marker
         (up to the end of that line) and pull the best short answer out of it.
      2. Otherwise, a date span anywhere in the text (DROP dates are month-name spans).
      3. Otherwise, the LAST number in the text (numbers dominate DROP answers, and the
         final number is typically the model's bottom-line figure).
      4. Otherwise, fall back to the last non-empty line, then the whole text.

    A genuinely wrong completion still scores ~0: extraction never invents the gold, it
    only surfaces whatever short answer the completion actually committed to.
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
    """lm-eval ``custom`` filter_fn: extract the short answer from each response.

    Shape mirrors lm-eval's other custom filters: ``resps`` is a per-doc list of sample
    strings; returns the same structure with each sample replaced by its extracted short
    answer. ``docs`` is accepted for signature compatibility and intentionally unused —
    extraction must not peek at the gold answer.
    """
    return [[extract_drop_short_answer(r) for r in resp] for resp in resps]
