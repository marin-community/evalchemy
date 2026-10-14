"""Explicit source preparation and bounded composition of Math and Judge."""

import asyncio
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from harbor_config.errors import ErrorCategory
from verifyit.execution.worker import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.modes.grade_math import grade_math_candidate
from verifyit.preparation.errors import PreparationFailure
from verifyit.spec import EmptyOutputPolicy, JudgeSpec, MathProfile, MathSpec

from eval.contracts.failures import GradingBoundaryError
from eval.graders.answer_equivalence import (
    DEFAULT_NUM_WORKERS, JUDGE_PROMPT, EquivalenceJudgment, EquivalenceMethod,
    EquivalenceResult, JudgeLabel,
)
from eval.graders.simpleqa import SIMPLEQA_GRADER_TEMPLATE


class JudgePolicy(StrEnum):
    SOURCE_WHOLE = "source_whole_label_nontext_empty_v1"
    LEGACY_LINES = "legacy_lines_nontext_empty_v1"


class OlympiadPolicy(StrEnum):
    SOURCE = "source_first_box_dollar_alternatives_math_then_judge_v1"


@dataclass(frozen=True)
class RawJudgeInput:
    question: object
    reference: object
    candidate: object
    candidate_stage: str = "response"


@dataclass(frozen=True)
class PreparedJudgeInput:
    raw: RawJudgeInput
    question: str
    references: tuple[str, ...]
    candidate: str
    provenance: dict


@dataclass(frozen=True)
class JudgeOutcome(EquivalenceJudgment):
    verdict: Reward
    prepared: PreparedJudgeInput


@dataclass(frozen=True)
class MathOutcome(EquivalenceResult):
    verdict: Reward | None = None
    prepared: PreparedJudgeInput | None = None
    components: tuple[Reward, ...] = ()


@dataclass(frozen=True)
class FailedBoundary:
    failure: PreparationFailure
    verdict: Reward


def capture_judge_input(question, reference, candidate, *, candidate_stage="response"):
    """Capture source values before any coercion, selection, or extraction."""
    return RawJudgeInput(question, reference, candidate, candidate_stage)


def prepare_judge_input(raw, profile, policy, olympiad_policy):
    policy = JudgePolicy(policy)
    if not isinstance(raw.question, str) or not raw.question.strip():
        raise InvalidTask("Question must be a nonempty string")
    references = raw.reference
    provenance = {"judge_policy": policy.value, "candidate_stage": raw.candidate_stage,
                  "empty_candidate": "zero_not_attempted", "completion_tokens": [128, 2048],
                  "transport_retries": 0, "request_timeout": 300}
    candidate = raw.candidate if isinstance(raw.candidate, str) else ""
    if profile == "olympiad":
        # The source helper owns its first-box/turn-stop and dollar-group policies.
        from eval.chat_benchmarks.OlympiadBench.eval_instruct import _flatten_reference_answers
        from eval.graders.answer_extraction import AnswerExtractionError, extract_boxed_answer

        olympiad_policy = OlympiadPolicy(olympiad_policy)
        values = references if isinstance(references, (list, tuple)) else [references]
        if not values or any(type(value) not in (str, int, float) or
                             (isinstance(value, float) and not math.isfinite(value)) or
                             not str(value).strip() for value in values):
            raise InvalidTask("Olympiad references require nonempty finite text or numbers")
        references = tuple(_flatten_reference_answers(references))
        provenance["olympiad_policy"] = olympiad_policy.value
        if raw.candidate_stage == "response":
            try:
                candidate = extract_boxed_answer(candidate)
            except AnswerExtractionError as error:
                candidate = ""
                provenance["extraction_error"] = type(error).__name__
        elif raw.candidate_stage != "extracted_answer":
            raise InvalidTask("Unknown Olympiad candidate stage")
    elif profile == "simpleqa":
        references = (references,)
    if not isinstance(references, (tuple, list)) or not references or any(
        not isinstance(value, str) or not value.strip() for value in references
    ):
        raise InvalidTask("Reference answers must be nonempty strings")
    return PreparedJudgeInput(raw, raw.question, tuple(references), candidate, provenance)


def _judge_one(prepared, config, profile, policy):
    simpleqa = profile == "simpleqa"
    template = SIMPLEQA_GRADER_TEMPLATE if simpleqa else JUDGE_PROMPT
    template = template.replace("{target}", "{reference}").replace("{predicted_answer}", "{candidate}")
    template = template.replace("{reference_answers}", "{reference}").replace("{candidate_answer}", "{candidate}")
    labels = ({"A": JudgeLabel.CORRECT, "B": JudgeLabel.INCORRECT, "C": JudgeLabel.NOT_ATTEMPTED}
              if simpleqa else {label.value.upper(): label for label in JudgeLabel})
    reference = prepared.references[0] if simpleqa else json.dumps(prepared.references, ensure_ascii=False)
    spec = JudgeSpec(references=(reference,), question=prepared.question, rubric="labels", model=config.model,
                     exact_gate=False, prompt_template=template,
                     label_scores={label: float(value == JudgeLabel.CORRECT) for label, value in labels.items()},
                     label_scan="whole" if policy is JudgePolicy.SOURCE_WHOLE else "lines", label_case="upper",
                     request_timeout=300, max_completion_tokens=128, incomplete_retry_tokens=2048,
                     empty_output=EmptyOutputPolicy.ZERO)
    verdict = grade_judge_candidate(spec, prepared.candidate, connection=JudgeConnection(config.base_url, config.api_key))
    if verdict.detail.get("reason") == "empty_output":
        return JudgeOutcome(JudgeLabel.NOT_ATTEMPTED, "", verdict, prepared)
    return JudgeOutcome(labels[verdict.detail["verdict"]], verdict.detail["completion"], verdict, prepared)


def _failure(error, stage):
    status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
    failure = PreparationFailure(status, ErrorCategory.UNKNOWN, type(error).__name__, str(error), stage)
    return FailedBoundary(failure, finalize_preparation_failure(**asdict(failure)))


def _run_batch(raw, config, profile, policy, olympiad_policy, num_workers):
    stage = "preparation"
    try:
        try:
            policy = JudgePolicy(policy)
            OlympiadPolicy(olympiad_policy)
        except ValueError as error:
            raise InvalidTask(str(error)) from error
        if type(num_workers) is not int or num_workers < 1:
            raise InvalidTask("Judge concurrency must be a positive integer")
        prepared = [prepare_judge_input(item, profile, policy, olympiad_policy) for item in raw]
        stage = "grading"
        if profile != "olympiad":
            return asyncio.run(_judge_batch(prepared, config, profile, policy, num_workers))
        outcomes = []
        for item in prepared:
            components = tuple(grade_math_candidate(MathSpec(expected=reference, profile=MathProfile.BOXED), item.candidate)
                               for reference in item.references)
            verdict = aggregate_rewards(components, expected_total=len(components), policy=Aggregation.MAX)
            judgment = None
            method = EquivalenceMethod.MINERVA
            if verdict.status is Status.SCORED and verdict.reward != 1 and config is not None:
                judgment = _judge_one(item, config, profile, policy)
                verdict = judgment.verdict
                method = EquivalenceMethod.LLM_JUDGE
            if verdict.status is Status.INVALID_TASK:
                raise InvalidTask(str(verdict.detail))
            if verdict.status is not Status.SCORED:
                raise RuntimeError(str(verdict.detail))
            outcomes.append(MathOutcome(verdict.reward == 1, method, judgment, verdict, item, components))
        return outcomes
    except Exception as error:
        return _failure(error, stage)


async def _judge_batch(prepared, config, profile, policy, num_workers):
    semaphore = asyncio.Semaphore(num_workers)

    async def one(item):
        async with semaphore:
            return await asyncio.to_thread(_judge_one, item, config, profile, policy)

    return await asyncio.gather(*(one(item) for item in prepared))


async def grade_raw(raw, config, profile, *, policy=JudgePolicy.SOURCE_WHOLE,
                    olympiad_policy=OlympiadPolicy.SOURCE, timeout=300, num_workers=DEFAULT_NUM_WORKERS):
    """Bound preparation, queued work, Math and Judge together in one owned worker."""
    try:
        result = await asyncio.to_thread(call_bounded, _run_batch, raw, config, profile, policy,
                                         olympiad_policy, num_workers, timeout=timeout)
    except Exception as error:
        result = _failure(error, "runtime")
    if isinstance(result, FailedBoundary):
        raise GradingBoundaryError(asdict(result.failure), asdict(result.verdict))
    return result


async def judge_simpleqa(requests, config, num_workers=DEFAULT_NUM_WORKERS, *, policy=JudgePolicy.SOURCE_WHOLE, timeout=300):
    raw = [capture_judge_input(item.question, item.target, item.predicted_answer) for item in requests]
    return await grade_raw(raw, config, "simpleqa", policy=policy, timeout=timeout, num_workers=num_workers)


async def judge_equivalence(requests, config, num_workers=DEFAULT_NUM_WORKERS, *, policy=JudgePolicy.SOURCE_WHOLE, timeout=300):
    raw = [capture_judge_input(item.question, item.reference_answers, item.candidate_answer) for item in requests]
    return await grade_raw(raw, config, "equivalence", policy=policy, timeout=timeout, num_workers=num_workers)


async def grade_math_equivalence(requests, config, num_workers=DEFAULT_NUM_WORKERS, *, policy=JudgePolicy.SOURCE_WHOLE, timeout=300):
    raw = [capture_judge_input(item.question, item.reference_answers, item.candidate_answer,
                              candidate_stage="extracted_answer") for item in requests]
    return await grade_raw(raw, config, "olympiad", policy=policy, timeout=timeout, num_workers=num_workers)
