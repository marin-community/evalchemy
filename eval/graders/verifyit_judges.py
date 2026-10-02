"""Map Evalchemy prompts and result types to the shared Judge mode."""

import asyncio
import json

from verifyit.grade import InvalidTask
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import EmptyOutputPolicy, JudgeSpec

from eval.graders.answer_equivalence import (
    DEFAULT_NUM_WORKERS,
    JUDGE_PROMPT,
    EquivalenceJudgment,
    EquivalenceMethod,
    EquivalenceResult,
    JudgeLabel,
    verifyit_math_answers_equivalent,
)
from eval.graders.simpleqa import SIMPLEQA_GRADER_TEMPLATE


def _judge_one(request, config, profile):
    question, reference, candidate = request
    simpleqa = profile == "simpleqa"
    template = SIMPLEQA_GRADER_TEMPLATE if simpleqa else JUDGE_PROMPT
    template = template.replace("{target}", "{reference}").replace("{predicted_answer}", "{candidate}")
    template = template.replace("{reference_answers}", "{reference}").replace("{candidate_answer}", "{candidate}")
    labels = (
        {"A": JudgeLabel.CORRECT, "B": JudgeLabel.INCORRECT, "C": JudgeLabel.NOT_ATTEMPTED}
        if simpleqa
        else {label.value: label for label in JudgeLabel}
    )
    labels.update({label.lower() if simpleqa else label.upper(): value for label, value in list(labels.items())})
    spec = JudgeSpec(
        references=(reference,),
        question=question,
        rubric="labels",
        model=config.model,
        exact_gate=False,
        prompt_template=template,
        label_scores={label: float(value == JudgeLabel.CORRECT) for label, value in labels.items()},
        label_scan="lines",
        request_timeout=300,
        max_completion_tokens=128,
        incomplete_retry_tokens=2048,
        empty_output=EmptyOutputPolicy.ZERO,
    )
    verdict = grade_judge_candidate(spec, candidate, connection=JudgeConnection(config.base_url, config.api_key))
    if verdict.detail.get("reason") == "empty_output":
        return EquivalenceJudgment(JudgeLabel.NOT_ATTEMPTED, "")
    return EquivalenceJudgment(labels[verdict.detail["verdict"]], verdict.detail["completion"])


async def _judge(prompts, config, profile, num_workers):
    if type(num_workers) is not int or num_workers < 1:
        raise InvalidTask("Judge concurrency must be a positive integer")
    semaphore = asyncio.Semaphore(num_workers)

    async def one(prompt):
        async with semaphore:
            return await asyncio.to_thread(_judge_one, prompt, config, profile)

    outcomes = await asyncio.gather(*(one(prompt) for prompt in prompts), return_exceptions=True)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return outcomes


def _required(value, description):
    if not isinstance(value, str) or not value.strip():
        raise InvalidTask(description + " must be a nonempty string")


async def judge_simpleqa(requests, config, num_workers=DEFAULT_NUM_WORKERS):
    prompts = []
    for request in requests:
        _required(request.question, "Question")
        _required(request.target, "Reference")
        candidate = request.predicted_answer if isinstance(request.predicted_answer, str) else ""
        prompts.append((request.question, request.target, candidate))
    return await _judge(prompts, config, "simpleqa", num_workers)


async def judge_equivalence(requests, config, num_workers=DEFAULT_NUM_WORKERS):
    prompts = []
    for request in requests:
        _required(request.question, "Question")
        if not isinstance(request.reference_answers, (list, tuple)) or not request.reference_answers:
            raise InvalidTask("Equivalence judge requires reference answers")
        for reference in request.reference_answers:
            _required(reference, "Reference")
        candidate = request.candidate_answer if isinstance(request.candidate_answer, str) else ""
        prompts.append((request.question, json.dumps(request.reference_answers, ensure_ascii=False), candidate))
    return await _judge(prompts, config, "equivalence", num_workers)


async def grade_math_equivalence(requests, config, num_workers=DEFAULT_NUM_WORKERS):
    if config is not None:
        judgments = await judge_equivalence(requests, config, num_workers)
        return [
            EquivalenceResult(judgment.label == JudgeLabel.CORRECT, EquivalenceMethod.LLM_JUDGE, judgment)
            for judgment in judgments
        ]
    outcomes = []
    for request in requests:
        _required(request.question, "Question")
        if not isinstance(request.reference_answers, (list, tuple)) or not request.reference_answers:
            raise InvalidTask("Math grading requires reference answers")
        for reference in request.reference_answers:
            _required(reference, "Reference")
        outcomes.append(
            EquivalenceResult(
                verifyit_math_answers_equivalent(request.candidate_answer, request.reference_answers),
                EquivalenceMethod.MINERVA,
            )
        )
    return outcomes
