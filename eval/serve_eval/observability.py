# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort telemetry for the serve-and-eval runner."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from enum import StrEnum

from rigging import telemetry

from eval.serve_eval.results import EvalResults

SERVICE_NAME = "evalchemy"
SHUTDOWN_TIMEOUT = 2.0
_DRAIN_POLL_INTERVAL = 0.01
_SNAPSHOT_GAUGE_ATTRIBUTES = telemetry.snapshot_attributes("gauge", telemetry.CURRENT_SNAPSHOT)
_CAPABILITY_TOKEN = re.compile(r"(/proxy/t/)[^/\s]+")

logger = logging.getLogger("eval.serve_eval.telemetry")


class FailureStage(StrEnum):
    OUTPUT = "output"
    PROVIDER = "provider"
    EVALUATION = "evaluation"
    RESULTS = "results"
    CLEANUP = "cleanup"


class TelemetryOutcome(StrEnum):
    READY = "ready"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    START_FAILED = "start_failed"
    INTERRUPTED = "interrupted"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    JOB_NOT_FOUND = "job_not_found"

_PROVIDER_DURATION = telemetry.histogram("provider_duration_seconds", unit="s")
_READINESS_ATTEMPTS = telemetry.counter("readiness_attempts")
_READINESS_DURATION = telemetry.histogram("readiness_duration_seconds", unit="s")
_SUBPROCESS_DURATION = telemetry.histogram("subprocess_duration_seconds", unit="s")
_SUBPROCESS_EXIT_CODE = telemetry.gauge("subprocess_exit_code")
_RUNS = telemetry.counter("runs")
_TASK_SAMPLES = telemetry.gauge("task_samples")
_TASK_METRIC = telemetry.gauge("task_metric")
_RESULTS_PERSISTED = telemetry.counter("results_persisted")
_CLEANUP_DURATION = telemetry.histogram("cleanup_duration_seconds", unit="s")
_CLEANUPS = telemetry.counter("cleanups")


def configure(endpoint: str | None, *, run_id: str, model: str, provider: str, tasks: list[str]) -> None:
    """Configure the process exporter when a Finelog endpoint was provided."""
    if endpoint is None:
        return
    try:
        telemetry.configure(
            endpoint=endpoint,
            service=SERVICE_NAME,
            attributes={
                "run_id": run_id,
                "model": model,
                "provider": provider,
                "tasks": ",".join(tasks),
            },
        )
    except Exception as error:
        # Telemetry is explicitly best-effort and must not change the eval path.
        logger.warning(
            "telemetry configuration failed (%s); continuing without telemetry",
            type(error).__name__,
        )
        return


def shutdown() -> None:
    """Attempt a bounded final export without changing the runner outcome."""
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT
    try:
        # Rigging shutdown sets its stop flag before joining. The exporter then
        # abandons records after the current batch, so wait on its public resident
        # count first; a public flush primitive would replace this poll.
        status = telemetry.runtime_status()
        while status.configured and status.queued_records:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_DRAIN_POLL_INTERVAL, remaining))
            status = telemetry.runtime_status()
    except Exception as error:
        logger.warning("telemetry drain status failed (%s); continuing shutdown", type(error).__name__)
    try:
        telemetry.shutdown(max(0.0, deadline - time.monotonic()))
    except Exception as error:
        logger.warning("telemetry shutdown failed (%s); evaluation outcome is unchanged", type(error).__name__)


def run_started(output_dir: str) -> None:
    telemetry.event("run_started", {"output_dir": output_dir})


def run_terminal(state: TelemetryOutcome) -> None:
    attributes = {"state": state}
    _RUNS.add(attributes=attributes)
    telemetry.event("run_terminal", {"state": state}, attributes=attributes)


def record_failure(
    stage: FailureStage,
    error: BaseException,
    *,
    attributes: Mapping[str, str] | None = None,
) -> None:
    record_attributes = {"stage": stage, **dict(attributes or {})}
    redacted = _CAPABILITY_TOKEN.sub(r"\1<redacted>", str(error))
    message = redacted.encode("utf-8")[:4096].decode("utf-8", "ignore")
    telemetry.event(
        "failure",
        {"error_type": type(error).__name__, "message": message},
        attributes=record_attributes,
    )


def provider_starting() -> None:
    telemetry.event("provider_starting", {})


def provider_terminal(duration: float, outcome: TelemetryOutcome) -> None:
    attributes = {"outcome": outcome}
    _PROVIDER_DURATION.record(duration, attributes=attributes)
    telemetry.event("provider_terminal", {"duration_seconds": duration}, attributes=attributes)


def readiness_terminal(duration: float, attempts: int, outcome: TelemetryOutcome) -> None:
    attributes = {"outcome": outcome}
    _READINESS_ATTEMPTS.add(attempts, attributes=attributes)
    _READINESS_DURATION.record(duration, attributes=attributes)
    telemetry.event(
        "readiness_terminal",
        {"attempts": attempts, "duration_seconds": duration},
        attributes=attributes,
    )


def subprocess_started(process_id: int) -> None:
    telemetry.event("subprocess_started", {}, attributes={"process_id": str(process_id)})


def subprocess_terminal(duration: float, outcome: TelemetryOutcome, *, exit_code: int | None = None) -> None:
    attributes = {"outcome": outcome}
    _SUBPROCESS_DURATION.record(duration, attributes=attributes)
    body: dict[str, float | int] = {"duration_seconds": duration}
    if exit_code is not None:
        _SUBPROCESS_EXIT_CODE.set(exit_code, attributes=_SNAPSHOT_GAUGE_ATTRIBUTES)
        body["exit_code"] = exit_code
    telemetry.event("subprocess_terminal", body, attributes=attributes)


def output_ready() -> None:
    telemetry.event("output_ready", {})


def results_persisted(results_path: str) -> None:
    _RESULTS_PERSISTED.add()
    telemetry.event("results_persisted", {"results_path": results_path})


def record_task_results(results: EvalResults, tasks: list[str]) -> None:
    for task in tasks:
        samples = results.sample_count(task)
        if samples is not None:
            _TASK_SAMPLES.set(samples, attributes={"task": task, **_SNAPSHOT_GAUGE_ATTRIBUTES})
        for metric, value in results.numeric_metrics(task).items():
            _TASK_METRIC.set(value, attributes={"task": task, "metric": metric, **_SNAPSHOT_GAUGE_ATTRIBUTES})


def cleanup_terminal(duration: float, outcome: TelemetryOutcome) -> None:
    attributes = {"outcome": outcome}
    _CLEANUP_DURATION.record(duration, attributes=attributes)
    _CLEANUPS.add(attributes=attributes)
    telemetry.event("cleanup_terminal", {"duration_seconds": duration}, attributes=attributes)
