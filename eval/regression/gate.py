# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate a run's :class:`EvalResults` against a golden :class:`Baseline`.

The gate is deliberately a **connectivity + coarse-quality smoke check**: a small
``--limit`` run of a 0.6B model moves in coarse increments (gsm8k strict-match swings
~3/20 run-to-run even greedy), so it asserts (a) the endpoint answered the expected
number of samples and (b) each tracked metric clears a floor -- plus, optionally, stays
within ``tolerance`` of a recorded ``reference``. See ``eval/regression/README.md``.

Reads a run's :class:`~eval.serve_eval.results.EvalResults` and a
:class:`~eval.regression.gatespec.GateSpec`, and returns a :class:`GateReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from eval.regression.gatespec import GateSpec, MetricThreshold
from eval.serve_eval.results import EvalResults


@dataclass
class MetricCheck:
    task: str
    metric: str
    observed: Optional[float]
    min_threshold: Optional[float] = None
    reference: Optional[float] = None
    tolerance: Optional[float] = None
    error: Optional[str] = None  # set when the metric could not be read

    @property
    def ok(self) -> bool:
        if self.error is not None or self.observed is None:
            return False
        if self.min_threshold is not None and self.observed < self.min_threshold:
            return False
        if self.reference is not None and self.tolerance is not None:
            if abs(self.observed - self.reference) > self.tolerance:
                return False
        return True

    def describe(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        if self.error is not None:
            return f"[{status}] {self.task}/{self.metric}: {self.error}"
        parts = [f"observed={self.observed:.4f}"]
        if self.min_threshold is not None:
            parts.append(f"min={self.min_threshold:.4f}")
        if self.reference is not None and self.tolerance is not None:
            parts.append(f"ref={self.reference:.4f}±{self.tolerance:.4f}")
        return f"[{status}] {self.task}/{self.metric}: " + ", ".join(parts)


@dataclass
class SampleCheck:
    task: str
    expected: int
    observed: Optional[int]

    @property
    def ok(self) -> bool:
        return self.observed is not None and self.observed == self.expected

    def describe(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        obs = "unknown" if self.observed is None else str(self.observed)
        return f"[{status}] {self.task}: samples observed={obs} expected={self.expected}"


@dataclass
class GateReport:
    metric_checks: List[MetricCheck] = field(default_factory=list)
    sample_checks: List[SampleCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.metric_checks) and all(c.ok for c in self.sample_checks)

    def render(self) -> str:
        lines = [c.describe() for c in self.sample_checks]
        lines += [c.describe() for c in self.metric_checks]
        lines.append("")
        lines.append("GATE: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(lines)

    def failures(self) -> List[str]:
        out = [c.describe() for c in self.sample_checks if not c.ok]
        out += [c.describe() for c in self.metric_checks if not c.ok]
        return out


def _metric_check(results: EvalResults, task: str, metric: str, thresholds: MetricThreshold) -> MetricCheck:
    check = MetricCheck(
        task=task,
        metric=metric,
        observed=None,
        min_threshold=thresholds.min,
        reference=thresholds.reference,
        tolerance=thresholds.tolerance,
    )
    observed = results.metric(task, metric)
    if observed is None:
        available = sorted(results.numeric_metrics(task))
        check.error = f"metric {metric!r} not in results[{task!r}]; available: {available}"
    else:
        check.observed = observed
    return check


def evaluate_gate(results: EvalResults, spec: GateSpec) -> GateReport:
    """Compare a run's results against a gate spec; return a :class:`GateReport`."""
    if not spec.tasks:
        raise ValueError("spec has no 'tasks' entries to check")

    report = GateReport()
    for task, task_spec in spec.tasks.items():
        if task_spec.expected_samples is not None:
            report.sample_checks.append(
                SampleCheck(task=task, expected=int(task_spec.expected_samples), observed=results.sample_count(task))
            )
        if not task_spec.metrics:
            raise ValueError(f"spec task {task!r} has no 'metrics' to check")
        for metric, thresholds in task_spec.metrics.items():
            report.metric_checks.append(_metric_check(results, task, metric, thresholds))
    return report


def gate_paths(results_path: str, spec_path: str) -> GateReport:
    """Convenience: gate a results file (or run dir) against a spec file."""
    return evaluate_gate(EvalResults.load_path_or_dir(results_path), GateSpec.load(spec_path))
