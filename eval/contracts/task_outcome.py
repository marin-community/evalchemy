"""Typed task outcomes shared by custom and lm-eval benchmark routes."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from numbers import Real
from typing import Any, Literal

from .sample_manifest import SampleCoverageError, SampleManifest

TASK_OUTCOME_SCHEMA_VERSION = 1


class TaskStatus(StrEnum):
    """Terminal state of one requested evaluation task."""

    SUCCEEDED = "succeeded"
    EXPORTED = "exported"
    FAILED = "failed"


class TaskRoute(StrEnum):
    """Adapter that executed a requested task."""

    CUSTOM = "Evalchemy chat benchmark"
    LM_EVAL = "lm-eval"
    UNKNOWN = "unknown"


class FailureCategory(StrEnum):
    """Failure classes needed at the task execution boundary."""

    PREPARATION = "preparation"
    GENERATION = "generation"
    GRADING = "grading"
    INCOMPLETE_EVALUATION = "incomplete_evaluation"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True)
class TaskFailure:
    """Machine-readable reason a task did not produce a valid outcome."""

    category: FailureCategory
    message: str
    exception_type: str | None = None


@dataclass(frozen=True)
class TaskOutcome:
    """Validated terminal outcome for one requested task."""

    task_name: str
    route: TaskRoute
    status: TaskStatus
    metrics: Mapping[str, Any]
    expected_count: int | None
    generated_count: int | None
    scored_count: int | None
    failure: TaskFailure | None = None
    schema_version: Literal[1] = TASK_OUTCOME_SCHEMA_VERSION

    @classmethod
    def failed(
        cls,
        task_name: str,
        route: TaskRoute,
        category: FailureCategory,
        message: str,
        *,
        exception: BaseException | None = None,
    ) -> "TaskOutcome":
        return cls(
            task_name=task_name,
            route=route,
            status=TaskStatus.FAILED,
            metrics={},
            expected_count=None,
            generated_count=None,
            scored_count=None,
            failure=TaskFailure(
                category=category,
                message=message,
                exception_type=type(exception).__name__ if exception is not None else None,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible task-outcome envelope."""
        return {
            "schema_version": self.schema_version,
            "task_name": self.task_name,
            "route": self.route,
            "status": self.status,
            "metrics": dict(self.metrics),
            "expected_count": self.expected_count,
            "generated_count": self.generated_count,
            "scored_count": self.scored_count,
            "failure": asdict(self.failure) if self.failure is not None else None,
        }


class EvaluationRunError(RuntimeError):
    """Raised when any requested task lacks a successful terminal outcome."""

    def __init__(self, outcomes: Sequence[TaskOutcome]):
        self.outcomes = tuple(outcomes)
        failures = [
            f"{outcome.task_name}: {outcome.failure.message if outcome.failure else outcome.status}"
            for outcome in self.outcomes
            if outcome.status is TaskStatus.FAILED
        ]
        super().__init__("Evaluation did not complete: " + "; ".join(failures))


def custom_task_outcome(
    task_name: str,
    route: TaskRoute,
    generation_result: Any,
    scored_result: Any,
    sample_manifest: SampleManifest | None = None,
) -> TaskOutcome:
    """Adapt a legacy custom benchmark result to the shared outcome contract."""
    generated_count = _record_count(generation_result)
    expected_count = _integer_field(generation_result, "total_examples")
    scored_count = _scored_count(scored_result, generated_count)
    manifest_result = _complete_manifest_coverage(
        task_name,
        route,
        sample_manifest,
        {"scored": _reported_scored_count(scored_result)},
    )
    if isinstance(manifest_result, TaskOutcome):
        return manifest_result
    if manifest_result is not None:
        expected_count, generated_count, scored_count = manifest_result
    return _adapt_completed_outcome(
        task_name,
        route,
        scored_result,
        expected_count=expected_count,
        generated_count=generated_count,
        scored_count=scored_count,
    )


def lm_eval_task_outcome(
    task_name: str,
    route: TaskRoute,
    result: Any,
    sample_manifest: SampleManifest | None = None,
) -> TaskOutcome:
    """Adapt one lm-eval invocation to the shared outcome contract."""
    if not isinstance(result, Mapping):
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.INVALID_RESULT,
            "lm-eval returned no result document",
        )

    all_metrics = result.get("results")
    if not isinstance(all_metrics, Mapping) or not all_metrics:
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.INCOMPLETE_EVALUATION,
            "lm-eval returned no task metrics",
        )

    if task_name in all_metrics:
        metrics = all_metrics[task_name]
    else:
        groups = result.get("groups")
        if not isinstance(groups, Mapping) or task_name not in groups:
            return TaskOutcome.failed(
                task_name,
                route,
                FailureCategory.INCOMPLETE_EVALUATION,
                "lm-eval omitted the requested task from its result document",
            )
        metrics = groups[task_name]
    expected_count, scored_count = lm_eval_task_counts(task_name, result)
    manifest_result = _complete_manifest_coverage(
        task_name,
        route,
        sample_manifest,
        {"expected": expected_count, "scored": scored_count},
    )
    if isinstance(manifest_result, TaskOutcome):
        return manifest_result
    if manifest_result is not None:
        expected_count, generated_count, scored_count = manifest_result
    else:
        generated_count = scored_count
    return _adapt_completed_outcome(
        task_name,
        route,
        metrics,
        expected_count=expected_count,
        generated_count=generated_count,
        scored_count=scored_count,
    )


def exported_task_outcome(task_name: str, route: TaskRoute, generation_result: Any) -> TaskOutcome:
    """Represent an upload-only task that intentionally does not run a grader."""
    generated_count = _record_count(generation_result)
    if generated_count == 0:
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.INCOMPLETE_EVALUATION,
            "task exported zero generations",
        )
    return TaskOutcome(
        task_name=task_name,
        route=route,
        status=TaskStatus.EXPORTED,
        metrics={},
        expected_count=_integer_field(generation_result, "total_examples"),
        generated_count=generated_count,
        scored_count=None,
    )


def validate_requested_outcomes(requested_tasks: Sequence[str], outcomes: Sequence[TaskOutcome]) -> None:
    """Require exactly one non-failed outcome for every requested task."""
    by_task: dict[str, list[TaskOutcome]] = {}
    for outcome in outcomes:
        by_task.setdefault(outcome.task_name, []).append(outcome)

    completed = list(outcomes)
    for task_name in dict.fromkeys(requested_tasks):
        task_outcomes = by_task.get(task_name, [])
        if not task_outcomes:
            completed.append(
                TaskOutcome.failed(
                    task_name,
                    TaskRoute.UNKNOWN,
                    FailureCategory.INCOMPLETE_EVALUATION,
                    "requested task produced no outcome",
                )
            )
        elif len(task_outcomes) > 1:
            completed.append(
                TaskOutcome.failed(
                    task_name,
                    task_outcomes[0].route,
                    FailureCategory.INVALID_RESULT,
                    "requested task produced multiple outcomes",
                )
            )

    if any(outcome.status is TaskStatus.FAILED for outcome in completed):
        raise EvaluationRunError(completed)


def validate_result_document(result: Mapping[str, Any]) -> None:
    """Reject aggregate persistence that bypasses the typed outcome boundary."""
    serialized = result.get("task_outcomes")
    if not isinstance(serialized, Mapping) or not serialized:
        raise EvaluationRunError(
            [
                TaskOutcome.failed(
                    "<run>",
                    TaskRoute.UNKNOWN,
                    FailureCategory.INCOMPLETE_EVALUATION,
                    "result document has no task outcomes",
                )
            ]
        )
    from .preflight import validate_task_preparations

    try:
        validate_task_preparations(result.get("task_preparations", {}), [str(name) for name in serialized])
    except (TypeError, ValueError) as exc:
        raise EvaluationRunError(
            [
                TaskOutcome.failed(
                    "<run>",
                    TaskRoute.UNKNOWN,
                    FailureCategory.INVALID_RESULT,
                    str(exc),
                )
            ]
        ) from exc
    failed = []
    for task_name, value in serialized.items():
        try:
            _task_outcome_from_mapping(str(task_name), value)
        except (TypeError, ValueError) as exc:
            failed.append(
                TaskOutcome.failed(
                    str(task_name),
                    _serialized_route(value),
                    FailureCategory.INVALID_RESULT,
                    str(exc),
                )
            )
    if failed:
        raise EvaluationRunError(failed)


def _task_outcome_from_mapping(task_name: str, value: Any) -> TaskOutcome:
    if not isinstance(value, Mapping):
        raise TypeError("task outcome must be a mapping")
    if value.get("schema_version") != TASK_OUTCOME_SCHEMA_VERSION:
        raise ValueError(f"task outcome schema_version must be {TASK_OUTCOME_SCHEMA_VERSION}")
    if value.get("task_name") != task_name:
        raise ValueError("task outcome key does not match task_name")
    try:
        status = TaskStatus(value.get("status"))
        route = TaskRoute(value.get("route"))
    except (TypeError, ValueError):
        raise ValueError("task outcome has an unknown status or route") from None
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("task outcome metrics must be a mapping")
    if status is TaskStatus.FAILED:
        raise ValueError("failed task outcome cannot be persisted")
    if value.get("failure") is not None:
        raise ValueError("completed task outcome contains a failure")
    outcome = TaskOutcome(
        task_name=task_name,
        route=route,
        status=status,
        metrics=dict(metrics),
        expected_count=_serialized_count(value, "expected_count"),
        generated_count=_serialized_count(value, "generated_count"),
        scored_count=_serialized_count(value, "scored_count"),
        failure=None,
    )
    if violation := _outcome_contract_violation(outcome):
        raise ValueError(violation.message)
    return outcome


def _serialized_route(value: Any) -> TaskRoute:
    if not isinstance(value, Mapping):
        return TaskRoute.UNKNOWN
    try:
        return TaskRoute(value.get("route", TaskRoute.UNKNOWN))
    except ValueError:
        return TaskRoute.UNKNOWN


def _adapt_completed_outcome(
    task_name: str,
    route: TaskRoute,
    metrics: Any,
    *,
    expected_count: int | None,
    generated_count: int | None,
    scored_count: int | None,
) -> TaskOutcome:
    if not isinstance(metrics, Mapping):
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.INVALID_RESULT,
            "task metrics must be a mapping",
        )
    if "error" in metrics:
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.GRADING,
            str(metrics["error"]),
        )
    outcome = TaskOutcome(
        task_name=task_name,
        route=route,
        status=TaskStatus.SUCCEEDED,
        metrics=dict(metrics),
        expected_count=expected_count,
        generated_count=generated_count,
        scored_count=scored_count,
    )
    if violation := _outcome_contract_violation(outcome):
        return TaskOutcome(
            task_name=task_name,
            route=route,
            status=TaskStatus.FAILED,
            metrics={},
            expected_count=expected_count,
            generated_count=generated_count,
            scored_count=scored_count,
            failure=violation,
        )
    return outcome


def _outcome_contract_violation(outcome: TaskOutcome) -> TaskFailure | None:
    if outcome.status is TaskStatus.FAILED:
        return outcome.failure or TaskFailure(FailureCategory.INVALID_RESULT, "failed outcome has no failure detail")
    if outcome.failure is not None:
        return TaskFailure(FailureCategory.INVALID_RESULT, "completed task outcome contains a failure")
    if outcome.status is TaskStatus.EXPORTED:
        if outcome.generated_count == 0:
            return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task exported zero generations")
        return None
    if not _contains_finite_number(outcome.metrics):
        return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task returned no finite numeric metrics")
    if 0 in {
        count for count in (outcome.expected_count, outcome.generated_count, outcome.scored_count) if count is not None
    }:
        return TaskFailure(FailureCategory.INCOMPLETE_EVALUATION, "task reported zero sample coverage")
    return None


def _serialized_count(value: Mapping[str, Any], key: str) -> int | None:
    count = value.get(key)
    if count is None:
        return None
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"task outcome {key} must be a non-negative integer or null")
    return count


def _contains_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return any(_contains_finite_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_finite_number(item) for item in value)
    return False


def _record_count(result: Any) -> int | None:
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return len(result)
    if not isinstance(result, Mapping):
        return None
    explicit = _integer_field(result, "num_examples")
    if explicit is not None:
        return explicit
    for key in ("examples", "model_outputs", "generations", "responses", "samples"):
        records = result.get(key)
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            return len(records)
    nested_counts = [
        len(records)
        for records in result.values()
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes))
    ]
    return sum(nested_counts) if nested_counts else None


def _scored_count(scored_result: Any, generated_count: int | None) -> int | None:
    reported = _reported_scored_count(scored_result)
    return reported if reported is not None else generated_count


def _reported_scored_count(scored_result: Any) -> int | None:
    if isinstance(scored_result, Mapping):
        for key in (
            "scored_count",
            "num_examples",
            "total_examples",
            "num_questions",
            "sample_len",
            "total_samples",
        ):
            count = _integer_field(scored_result, key)
            if count is not None:
                return count
    return None


def _complete_manifest_coverage(
    task_name: str,
    route: TaskRoute,
    sample_manifest: SampleManifest | None,
    reported_counts: Mapping[str, int | None],
) -> tuple[int, int, int] | TaskOutcome | None:
    if sample_manifest is None or not sample_manifest.expected_sample_count:
        return None
    try:
        sample_manifest.validate_generated()
        generated_count = sample_manifest.generated_sample_count
        for label, reported in reported_counts.items():
            if reported is not None and reported != generated_count:
                raise SampleCoverageError(
                    f"{task_name}: evaluator reported {reported} {label} samples but the "
                    f"manifest contains {generated_count} generated samples"
                )
        sample_manifest.mark_scored(reported_counts.get("scored"))
        sample_manifest.validate_scored()
    except SampleCoverageError as exc:
        return TaskOutcome.failed(
            task_name,
            route,
            FailureCategory.INCOMPLETE_EVALUATION,
            str(exc),
            exception=exc,
        )
    return (
        sample_manifest.expected_sample_count,
        generated_count,
        sample_manifest.scored_sample_count,
    )


def _integer_field(value: Any, key: str) -> int | None:
    if not isinstance(value, Mapping):
        return None
    field = value.get(key)
    return field if isinstance(field, int) and not isinstance(field, bool) else None


def lm_eval_task_counts(task_name: str, result: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """Return lm-eval's original and effective counts from its result document."""
    counts = result.get("n-samples")
    if isinstance(counts, Mapping):
        task_counts = counts.get(task_name)
        if isinstance(task_counts, Mapping):
            expected = _integer_field(task_counts, "original")
            scored = _integer_field(task_counts, "effective")
            return (
                expected if expected is not None else scored,
                scored if scored is not None else expected,
            )

    samples = result.get("samples")
    if isinstance(samples, Mapping):
        task_samples = samples.get(task_name)
        if isinstance(task_samples, Sequence) and not isinstance(task_samples, (str, bytes)):
            count = len(task_samples)
            return count, count

    metrics = result.get("results")
    if isinstance(metrics, Mapping):
        task_metrics = metrics.get(task_name)
        count = _integer_field(task_metrics, "sample_len")
        return count, count
    return None, None
