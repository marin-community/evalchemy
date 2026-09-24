"""Benchmark-independent evaluation failure taxonomy and classification."""

from __future__ import annotations

from enum import StrEnum

from eval.limits import ContextWindowExceededError, MissingContextLengthError


class FailurePhase(StrEnum):
    """Lifecycle boundary at which an exception escaped."""

    PREPARATION = "preparation"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    GRADING = "grading"
    SERIALIZATION = "serialization"


class FailureCategory(StrEnum):
    """Failure classes needed at the task execution boundary."""

    PREPARATION = "preparation"
    RESOURCE = "resource"
    AGENT_TIMEOUT = "AgentTimeoutError"
    MODEL_TRANSPORT = "model_transport"
    MALFORMED_MODEL_RESPONSE = "malformed_model_response"
    GENERATION_POLICY = "generation_policy"
    GENERATION = "generation"
    GRADER_INFRASTRUCTURE = "grader_infrastructure"
    SAMPLE_EXECUTION = "sample_execution"
    GRADING = "grading"
    INCOMPLETE_EVALUATION = "incomplete_evaluation"
    SERIALIZATION = "serialization"
    INVALID_RESULT = "invalid_result"


class ModelRequestValidationError(ValueError):
    """Raised before generation when a representative request is malformed."""


def classify_task_exception(phase: FailurePhase, exception: BaseException) -> FailureCategory:
    """Map escaped exceptions onto the benchmark-independent failure taxonomy."""
    if phase is FailurePhase.PREPARATION:
        if isinstance(exception, (FileNotFoundError, ModuleNotFoundError, ImportError)):
            return FailureCategory.RESOURCE
        return FailureCategory.PREPARATION
    if phase is FailurePhase.SERIALIZATION:
        return FailureCategory.SERIALIZATION
    if isinstance(exception, TimeoutError):
        return FailureCategory.AGENT_TIMEOUT
    if phase in {FailurePhase.EVALUATION, FailurePhase.GRADING}:
        if phase is FailurePhase.EVALUATION and isinstance(exception, ConnectionError):
            return FailureCategory.MODEL_TRANSPORT
        return FailureCategory.GRADER_INFRASTRUCTURE
    if isinstance(exception, (ContextWindowExceededError, MissingContextLengthError, ModelRequestValidationError)):
        return FailureCategory.GENERATION_POLICY
    if isinstance(exception, ConnectionError):
        return FailureCategory.MODEL_TRANSPORT
    return FailureCategory.GENERATION
