"""Capture and prepare JEEBench batches before publishing any scored samples."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from eval.chat_benchmarks.JEEBench.utils import last_boxed_only_string, remove_boxed
from verifyit.adapters.evalchemy_jee import JEEPolicy, capture_jee_input, grade_prepared_jee, prepare_jee_input
from harbor_config.errors import error_category
from verifyit.bounded import call_bounded
from verifyit.grade import InvalidTask, Status, finalize_preparation_failure
from verifyit.preparation.errors import PreparationFailure

from eval.contracts.failures import GradingBoundaryError

BOXED_POLICY = "source_last_boxed_remove_boxed_nontext_empty_v1"


@dataclass(frozen=True)
class JEEBatchItem:
    gold: object
    question_type: str
    model_outputs: object


def capture_batch(examples, repeats):
    """Retain source outputs and trusted task fields without answer selection."""
    if type(repeats) is not int or repeats <= 0 or not isinstance(examples, list) or not examples:
        raise InvalidTask("JEEBench requires a nonempty repeated batch")
    captured = []
    for example in examples:
        if not isinstance(example, dict):
            raise InvalidTask("Malformed JEEBench task")
        outputs = example.get("model_outputs")
        captured.append(JEEBatchItem(example.get("gold"), example.get("type"),
                                     tuple(outputs) if isinstance(outputs, list) else outputs))
    return tuple(captured)


def prepare_batch(captured, repeats, policy, boxed_policy):
    """Apply the explicit source box and answer policies after trusted preflight."""
    if boxed_policy != BOXED_POLICY:
        raise InvalidTask("unknown JEEBench boxed extraction policy")
    for item in captured:
        prepare_jee_input(capture_jee_input(item.gold, None, item.question_type), policy)
    for item in captured:
        if not isinstance(item.model_outputs, tuple) or len(item.model_outputs) != repeats:
            raise RuntimeError("JEEBench model outputs must align with repetitions")
    prepared = []
    for item in captured:
        answers = []
        for raw in item.model_outputs:
            answer = ""
            if isinstance(raw, str):
                boxed = last_boxed_only_string(raw)
                if boxed is not None:
                    try:
                        answer = remove_boxed(boxed)
                    except AssertionError:
                        answer = ""
            answers.append(prepare_jee_input(capture_jee_input(item.gold, answer, item.question_type), policy))
        prepared.append(tuple(answers))
    return tuple(prepared)


def _grade_batch(examples, repeats, policy, boxed_policy, timeout):
    """Return rewards, statuses, policies, effective options, and input hashes.

    Raw and prepared inputs remain private to external audit instrumentation.
    """
    try:
        policy = JEEPolicy(policy)
    except ValueError as error:
        raise InvalidTask("unknown JEEBench preparation policy") from error
    captured = capture_batch(examples, repeats)
    prepared = prepare_batch(captured, repeats, policy, boxed_policy)
    records = []
    for item, answers in zip(captured, prepared, strict=True):
        verdicts = [grade_prepared_jee(answer) for answer in answers]
        for verdict in verdicts:
            if verdict.status == Status.INVALID_TASK:
                raise InvalidTask(str(verdict.detail))
            if verdict.status != Status.SCORED:
                raise RuntimeError(str(verdict.detail))
        records.append({"raw": asdict(item), "prepared": [asdict(answer) for answer in answers],
                        "boxed_policy": boxed_policy, "verdicts": [asdict(verdict) for verdict in verdicts]})
    return [{
        "verdicts": [{"reward": verdict["reward"], "status": verdict["status"]} for verdict in record["verdicts"]],
        "policy": policy.value,
        "boxed_policy": boxed_policy,
        "reference_sha256": hashlib.sha256(json.dumps(record["raw"]["gold"]).encode()).hexdigest(),
        "outputs_sha256": hashlib.sha256(json.dumps(record["raw"]["model_outputs"]).encode()).hexdigest(),
        "effective_options": {"item_credit": 0.25, "tolerance_abs": 0.01, "tolerance_rel": 0, "timeout": timeout},
    } for record in records]


def grade_batch(examples, repeats, *, policy=JEEPolicy.SOURCE, boxed_policy=BOXED_POLICY, timeout=30):
    """Bound the full cohort preparation and primitive grading by one deadline."""
    try:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise InvalidTask("JEEBench timeout must be finite and positive")
        return call_bounded(_grade_batch, examples, repeats, policy, boxed_policy, timeout, timeout=timeout)
    except Exception as error:
        status = Status.INVALID_TASK if isinstance(error, InvalidTask) else Status.INFRA_ERROR
        failure = PreparationFailure(status, error_category(type(error).__name__), type(error).__name__, str(error), "jee_batch")
        verdict = finalize_preparation_failure(**asdict(failure))
        raise GradingBoundaryError(asdict(failure), asdict(verdict)) from error
