"""LLM-as-judge for FinanceBench.

Implements a SimpleQA-style grader (the same template OpenAI's simple-evals uses for
short-form factual Q&A): an LLM judge compares the model's answer against the gold answer
and returns one of three labels -- ``correct``, ``incorrect``, ``not_attempted`` -- which
is then aggregated into accuracy. Adapted from the HLE judge
(eval/chat_benchmarks/HLE/run_judge_results.py) and OpenAI's SimpleQA grader template.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Tuple

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# SimpleQA-style judge prompt. The judge sees the question, the gold answer, and the
# model's answer, and must emit exactly one of the three labels. The rubric explicitly
# tolerates formatting / surface-form differences and a small numerical margin (financial
# answers are frequently dollar amounts or percentages), and routes "I don't know"-style
# non-answers to ``not_attempted`` rather than ``incorrect`` (the SimpleQA convention so
# the metric is not penalized for safe abstention).
JUDGE_PROMPT = """You are an expert and precise grader. You will be given a question, a gold reference answer, and a model's predicted answer. Your task is to judge whether the predicted answer is correct given the gold answer.

Judge based on the following rubric:
- "correct": The predicted answer matches the gold answer in substance. Numerical answers (dollar amounts, percentages, counts) are correct if they match within a small margin of error and refer to the same quantity. Surface-form differences (formatting, extra words, units written differently) do not matter as long as the substance matches.
- "incorrect": The predicted answer attempts to answer the question but is wrong, contradicts the gold answer, or is missing key information that materially changes the meaning.
- "not_attempted": The predicted answer does not attempt to answer the question (e.g. "I don't know", "I am not sure", a refusal, or empty).

Respond with EXACTLY one of: correct, incorrect, not_attempted. Do not output anything else.

Question: {question}
Gold answer: {gold}
Predicted answer: {predicted}

Judgment:"""

# How hard to scrub the judge's free-form text down to a label.
_LABEL_PATTERN = re.compile(r"correct|incorrect|not_attempted", re.IGNORECASE)

# Concurrency for the async judge fan-out. The OpenAI chat-completions endpoint is I/O
# bound, so a modest semaphore keeps throughput high without tripping rate limits.
DEFAULT_NUM_WORKERS = 16

# Per-request timeout + retries on the judge client. 300s matches HLE; judge completions
# are short (a single label) so this only fires on a stalled connection.
client = AsyncOpenAI(timeout=300.0, max_retries=2)


def _parse_judgment(text: str) -> str:
    """Reduce the judge's free-form text to one of the three labels (lowercased).

    The judge is prompted to emit a bare label, but in practice models occasionally wrap
    it in prose ("The answer is correct.") or capitalization; pull the first label token
    out so a chatty judge is still scored. Falls back to ``"incorrect"`` only when no
    label is recoverable (treated as a failed-to-answer, the conservative bucket).
    """
    if not text:
        return "not_attempted"
    match = _LABEL_PATTERN.search(text)
    if match is None:
        return "incorrect"
    label = match.group(0).lower()
    # ``not_attempted`` contains the substring "attempted" and ``incorrect`` contains
    # ``correct``; prefer the longest label match at this position by checking the
    # match's actual span rather than a naive substring test.
    return label


async def judge_answer(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model: str,
) -> Tuple[str, str]:
    """Judge a single (question, gold, predicted) triple via the LLM judge.

    Returns a ``(label, raw)`` tuple where ``label`` is one of
    ``correct``/``incorrect``/``not_attempted`` and ``raw`` is the judge's verbatim
    completion (kept for debugging / per-sample audit). On any API failure the item is
    conservatively labeled ``incorrect`` so a flaky judge never inflates accuracy.
    """
    prompt = JUDGE_PROMPT.format(
        question=question, gold=gold_answer, predicted=predicted_answer
    )
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            max_tokens=16,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        return _parse_judgment(raw), raw
    except Exception as e:  # network / rate-limit / parse -- never crash the eval
        logger.warning(f"Judge call failed for question '{question[:60]}...': {e}")
        return "incorrect", f"<judge_error: {e}>"


async def judge_all(
    items: List[Dict[str, Any]],
    judge_model: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[Tuple[str, str]]:
    """Judge every item in ``items`` concurrently.

    Each item must carry ``question``, ``answer`` (the gold), and ``model_output`` (the
    prediction) keys. Returns the ``(label, raw)`` list in input order so the caller can
    zip it back onto the examples.
    """
    semaphore = asyncio.Semaphore(num_workers)

    async def bound(item: Dict[str, Any]) -> Tuple[str, str]:
        async with semaphore:
            return await judge_answer(
                item["question"],
                str(item["answer"]),
                item.get("model_output", "") or "",
                judge_model,
            )

    return await asyncio.gather(*[bound(item) for item in items])


def judge(
    items: List[Dict[str, Any]],
    judge_model: str,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> List[Tuple[str, str]]:
    """Synchronous entry point: run the async judge fan-out and block on the result."""
    return asyncio.run(judge_all(items, judge_model, num_workers=num_workers))
