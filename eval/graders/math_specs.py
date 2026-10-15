"""Reusable item graders selected by sources through GraderSpec."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import re
from typing import Any

from eval.contracts.eval_spec import EvalSpec, Grade, JudgePlan
from eval.completion_response import CompletionClassification, CompletionText
from eval.graders.answer_equivalence import math_answers_equivalent
from eval.graders.answer_extraction import extract_final_boxed_answer
from eval.graders.answer_extraction import final_response_text
from eval.chat_benchmarks.GPQADiamond.testing_utils import get_multiple_choice_answer
from eval.lm_eval_tasks.gsm8k.utils import extract_gsm8k_flexible_answer
from eval.graders.answer_extraction import AnswerExtractionError, extract_boxed_answer, extraction_failure

from eval.graders.judged import equivalence_judge

def boxed_math(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = extract_final_boxed_answer(responses[-1])
    correct = math_answers_equivalent(answer, [str(spec.target)])
    return Grade({"accuracy": float(correct)}, {"filtered_resps": [answer]})

def gsm8k(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = extract_gsm8k_flexible_answer(str(responses[-1]))
    return Grade({"accuracy": float(answer == str(spec.target))}, {"filtered_resps": [answer]})

def letter_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = get_multiple_choice_answer(str(responses[-1]))
    return Grade({"accuracy": float(answer == spec.target)}, {"filtered_resps": [answer]})

def mmlu_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    output = responses[-1]
    if isinstance(output, CompletionText):
        completion = output.response
        output = completion.content or (
            completion.reasoning_content if completion.classification is CompletionClassification.REASONING_ONLY else ""
        ) or ""
    text = final_response_text(output)
    answer = None
    for pattern in (r"answer is \(?([A-J])\)?", r"Answer:\s*([A-J])", r"\b([A-J])\b(?!.*\b[A-J]\b)"):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).upper()
            break
    return Grade({"accuracy": float(answer == spec.target)}, {"filtered_resps": [answer]})

def matharena(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    # Optional grader code is imported only after the HMMT source has loaded it.
    from eval.chat_benchmarks.HMMT.matharena.parser import check_answers, extract_answer, parse_answer

    target, _ = parse_answer(str(spec.target))
    answer, _ = extract_answer(str(responses[-1]), False, True, "," in str(spec.target))
    return Grade({"accuracy": float(check_answers(answer, target))}, {"filtered_resps": [str(answer)]})

def answer_number(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    text = str(responses[-1]).lower()
    answer = None
    for pattern in (r"answer:.*?(\d+)", r"answer is.*?(\d+)", r"boxed{(\d+)}"):
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1]
            break
    return Grade({"accuracy": float(math_answers_equivalent(answer, [str(spec.target)]))},
                 {"filtered_resps": [answer]})

def jee(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    from eval.chat_benchmarks.JEEBench.utils import compute_score

    answer = extract_final_boxed_answer(responses[-1])
    return Grade({"accuracy": float(compute_score(spec.target, answer, spec.grader.reference["type"]))},
                 {"filtered_resps": [answer]})

def nupa(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    scorer = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")
    answer_format = spec.grader.reference["format"]
    score = scorer.score_prediction(str(responses[-1]), spec.target, answer_format)
    return Grade({
        "exact_match": score.exact_match, "digit_match": score.digit_match, "dlength": score.dlength,
        "format_valid_rate": score.format_valid, "no_answer_rate": score.no_answer,
    }, {"filtered_resps": [scorer.extract_answer(str(responses[-1]), answer_format)]})

def math_equivalence(spec: EvalSpec, responses: Sequence[Any]) -> Grade | JudgePlan:
    try:
        answer = extract_boxed_answer(responses[-1])
    except AnswerExtractionError as error:
        return Grade({"accuracy": 0.0}, {"filtered_resps": [""], "answer_extraction_error": extraction_failure(error)})
    correct = bool(answer.strip()) and math_answers_equivalent(answer, spec.grader.reference["answers"])
    if correct or not spec.grader.reference["judge"]:
        return Grade({"accuracy": float(correct)}, {"filtered_resps": [answer], "method": "minerva"})
    return equivalence_judge(spec, [answer])
