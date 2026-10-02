"""Bounded source preparation and Math/MAX grading without a judge."""

import copy
import hashlib
import json
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict

from harbor_config.errors import error_category
from verifyit.bounded import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Status, aggregate_rewards, finalize_preparation_failure
from verifyit.modes.grade_math import grade_math_candidate
from verifyit.preparation.errors import PreparationFailure
from verifyit.spec import MathProfile, MathSpec

from eval.contracts.failures import GradingBoundaryError
from eval.graders.verifyit_judges import JudgePolicy, OlympiadPolicy, capture_judge_input, prepare_judge_input

POLICY = "source_first_box_dollar_alternatives_meaningful_math_failclosed_v1"
_DEADLINE = ContextVar("olympiad_deadline", default=None)


def _grade_batch(examples, count, policy, timeout):
    if policy != POLICY:
        raise InvalidTask("unknown deterministic Olympiad preparation policy")
    if type(count) is not int or count <= 0 or not isinstance(examples, list) or not examples:
        raise InvalidTask("Olympiad requires a nonempty completion cohort")
    captured = []
    prepared_references = []
    for example in examples:
        if not isinstance(example, dict):
            raise InvalidTask("Malformed Olympiad task")
        raw = capture_judge_input(example.get("problem", example.get("question")), example.get("answer"), example.get("model_outputs", example.get("model_output")))
        captured.append(raw)
    for raw in captured:
        reference_view = capture_judge_input(raw.question, raw.reference, "", candidate_stage="extracted_answer")
        prepared = prepare_judge_input(reference_view, "olympiad", JudgePolicy.SOURCE_WHOLE, OlympiadPolicy.SOURCE)
        for reference in prepared.references:
            grade_math_candidate(MathSpec(expected=reference, profile=MathProfile.BOXED), "")
        prepared_references.append(prepared.references)
    outputs = []
    for example in examples:
        values = example.get("model_outputs") if count != 1 or "model_outputs" in example else [example.get("model_output")]
        if not isinstance(values, list) or len(values) != count or ("model_outputs" not in example and "model_output" not in example):
            raise RuntimeError("Olympiad model outputs must align with completion count")
        outputs.append(tuple(values))
    batches = []
    for raw, references, values in zip(captured, prepared_references, outputs, strict=True):
        records = []
        for output in values:
            item = prepare_judge_input(capture_judge_input(raw.question, raw.reference, output), "olympiad",
                                       JudgePolicy.SOURCE_WHOLE, OlympiadPolicy.SOURCE)
            components = [grade_math_candidate(MathSpec(expected=reference, profile=MathProfile.BOXED), item.candidate)
                          for reference in references]
            verdict = aggregate_rewards(components, expected_total=len(references), policy=Aggregation.MAX)
            if verdict.status is Status.INVALID_TASK:
                raise InvalidTask(str(verdict.detail))
            if verdict.status is not Status.SCORED:
                raise RuntimeError(str(verdict.detail))
            records.append({"reward": verdict.reward, "status": verdict.status.value, "candidate": item.candidate,
                            "policy": policy, "reference_sha256": hashlib.sha256(json.dumps(raw.reference).encode()).hexdigest(),
                            "output_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
                            "effective_options": {"math_profile": MathProfile.BOXED.value, "aggregation": "max", "timeout": timeout}})
        batches.append(records)
    return batches


def _boundary_error(error):
    status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
    failure = PreparationFailure(status, error_category(type(error).__name__), type(error).__name__, str(error), "olympiad_batch")
    return GradingBoundaryError(asdict(failure), asdict(finalize_preparation_failure(**asdict(failure))))


def stage_results(results, *, num_samples, repeats):
    """Validate the producer's branch marker and stage results without mutation."""
    try:
        if type(num_samples) is not int or num_samples <= 0 or type(repeats) is not int or repeats <= 0:
            raise InvalidTask("Olympiad completion counts must be positive integers")
        if not isinstance(results, dict):
            raise InvalidTask("Olympiad results must be a mapping")
        if type(results.get("pass_at_k", False)) is not bool or results.get("pass_at_k", False) != (num_samples > 1):
            raise InvalidTask("Olympiad pass-at-k marker must match the configured completion count")
        examples = results.get("examples")
        if not isinstance(examples, list) or not examples or any(not isinstance(item, dict) for item in examples):
            raise InvalidTask("Olympiad requires a nonempty completion cohort")
        staged = copy.deepcopy(results)
        count = num_samples if num_samples > 1 else repeats
        for example in staged["examples"]:
            if count > 1:
                example["model_answers"] = [""] * count
            else:
                example["model_answer"] = ""
        return staged
    except Exception as error:
        raise _boundary_error(error) from error


def _timeout_budget(timeout):
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise InvalidTask("Olympiad timeout must be finite, representable, and positive")
    return float(timeout)


def remaining_budget(timeout):
    """Charge parent preparation and metric projection to the same call budget."""
    budget = _timeout_budget(timeout)
    deadline = _DEADLINE.get()
    remaining = budget if deadline is None else min(budget, deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("Olympiad total grading deadline exceeded")
    return remaining


@contextmanager
def total_deadline(timeout):
    """Keep deadline state local to this invocation, including nested calls."""
    started = time.monotonic()
    try:
        deadline = started + _timeout_budget(timeout)
        parent = _DEADLINE.get()
        token = _DEADLINE.set(deadline if parent is None else min(deadline, parent))
        try:
            yield
        finally:
            _DEADLINE.reset(token)
    except GradingBoundaryError:
        raise
    except Exception as error:
        raise _boundary_error(error) from error


def grade_batch(examples, count, *, policy=POLICY, timeout=30):
    """Validate all trusted alternatives and finish the cohort before mutation."""
    try:
        budget = remaining_budget(timeout)
        return call_bounded(_grade_batch, examples, count, policy, timeout, timeout=budget)
    except GradingBoundaryError:
        raise
    except Exception as error:
        raise _boundary_error(error) from error
