"""Single-answer JSON, numeric, grid, and code-line grading."""

from collections.abc import Sequence
from typing import Any
from difflib import SequenceMatcher
from eval.contracts.eval_spec import EvalSpec

import re
from fractions import Fraction

from eval.contracts.eval_spec import Grade
from eval.graders.gsm8k import FALLBACK, extract_flexible


def gsm8k_flexible(spec, responses):
    answer = extract_flexible(str(responses[-1]))
    predicted = re.search(r"-?\d+(\.\d+)?", answer.replace("$", "").replace(",", ""))
    target = re.search(r"-?\d+(\.\d+)?", str(spec.target).replace("$", "").replace(",", ""))
    correct = predicted is not None and target is not None and float(predicted.group()) == float(target.group())
    return Grade({"accuracy": float(correct), "no_answer": float(answer == FALLBACK)}, {"filtered_resps": [answer]})


def json_answer(spec, responses):
    from eval.chat_benchmarks.zeroeval.src.evaluation.eval_utils import (
        extract_first_complete_json, extract_last_complete_json, extract_values_from_json,
    )

    text = str(responses[-1])
    kind = spec.grader.reference["kind"]
    if kind == "grid":
        parsed = extract_last_complete_json(text)
        predicted = (parsed or {}).get("solution") or {}
        solution = spec.target
        cells = [(f"House {index + 1}", column, row[column_index]) for index, row in enumerate(solution["rows"])
                 for column_index, column in enumerate(solution["header"]) if column_index]
        correct = 0
        for house, column, truth in cells:
            value = predicted.get(house, {}).get(column)
            if isinstance(value, list):
                value = value[0] if value else ""
            if isinstance(value, str) and value.lower().strip() == truth.lower().strip():
                correct += 1
        metrics = {"accuracy": float(correct == len(cells)) * 100, "cell_acc": correct / len(cells) * 100,
                   "no_answer": float(not predicted) * 100, "reason_lens": len((parsed or {}).get("reasoning", ""))}
        return Grade(metrics, weights={"cell_acc": len(cells), "reason_lens": float(bool(predicted))})
    parsed = extract_first_complete_json(text)
    if parsed is None or "answer" not in parsed:
        parsed = extract_values_from_json(text, allow_no_quotes=True)
    missing = parsed is None or "answer" not in parsed
    answer = str((parsed or {}).get("answer", ""))
    if kind == "literal":
        correct = answer.strip("'\"").replace("\n", "\\n") == str(spec.target).strip("'\"")
    else:
        values = []
        for value in (answer, str(spec.target).replace("#", "")):
            value = value.replace("$", "").replace(",", "").strip()
            if "/" in value:
                try:
                    value = str(float(Fraction(value)))
                except (ValueError, ZeroDivisionError):
                    pass
            match = re.search(r"-?\d+(\.\d+)?", value)
            values.append(float(match.group()) if match else None)
        correct = values[0] is not None and values[0] == values[1]
    return Grade({"accuracy": float(correct and not missing) * 100, "no_answer": float(missing) * 100,
                  "reason_lens": len((parsed or {}).get("reasoning", ""))},
                 {"filtered_resps": [answer]}, {"reason_lens": float(not missing)})


def code_line(spec, responses):
    from fuzzywuzzy import fuzz
    from eval.chat_benchmarks.RepoBench.archive_data.utils import get_first_line_not_comment

    answer = get_first_line_not_comment(str(responses[-1]), language=spec.grader.reference["language"])
    target = str(spec.target)
    return Grade({"EM": float(answer.split() == target.split()) * 100, "ES": float(fuzz.ratio(target, answer))},
                 {"filtered_resps": [answer]})


def nonce_similarity(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    response = str(responses[-1])
    nonce = spec.grader.reference["nonce"]
    hit = response.startswith(nonce)
    score = SequenceMatcher(None, response.removeprefix(nonce), str(spec.target).removeprefix(nonce)).ratio() if hit else 0.0
    return Grade({"accuracy": score, "prefix_hit_rate": float(hit)})
