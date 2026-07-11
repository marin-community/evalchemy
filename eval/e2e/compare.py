# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate an evalchemy ``results_*.json`` against a checked-in baseline.

The gate is deliberately a **connectivity + coarse-quality smoke check**, not a
statistical regression test: a small ``--limit`` run of a 0.6B model moves in
coarse increments, so we assert (a) the endpoint actually answered the expected
number of samples and (b) each tracked metric clears a floor (and, optionally,
stays within a tolerance of a recorded reference). See ``eval/e2e/README.md`` and
the baseline's ``provenance`` block for what a given baseline was recorded with.

Pure stdlib; no Marin / lm-eval import, so it runs on stock CI.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


def find_latest_results(output_dir: str) -> str:
    """Return the newest ``results_*.json`` written under ``output_dir`` (recursive).

    evalchemy's ``DCEvaluationTracker`` writes to
    ``<output_path>/<model_name_sanitized>/results_<ts>.json``.
    """
    matches = glob.glob(os.path.join(output_dir, "**", "results_*.json"), recursive=True)
    matches += glob.glob(os.path.join(output_dir, "results_*.json"))
    if not matches:
        raise FileNotFoundError(f"no results_*.json found under {output_dir!r}")
    return max(set(matches), key=os.path.getmtime)


def load_results(results_path: str) -> dict:
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def effective_sample_count(results: dict, task: str) -> Optional[int]:
    """Best-effort effective sample count for ``task`` from the results dict."""
    n_samples = results.get("n-samples") or {}
    entry = n_samples.get(task)
    if isinstance(entry, dict):
        for key in ("effective", "original"):
            if isinstance(entry.get(key), int):
                return entry[key]
    # evalchemy's lm-eval-native (pretrain) path does not populate the top-level
    # ``n-samples`` map; it records the count on the task result as ``sample_len``.
    task_results = (results.get("results") or {}).get(task) or {}
    if isinstance(task_results.get("sample_len"), int):
        return task_results["sample_len"]
    return None


def get_metric(results: dict, task: str, metric: str) -> float:
    """Extract ``results['results'][task][metric]`` as a float, with a clear error."""
    task_results = (results.get("results") or {}).get(task)
    if task_results is None:
        available = sorted((results.get("results") or {}).keys())
        raise KeyError(f"task {task!r} not in results; available tasks: {available}")
    if metric not in task_results:
        available = sorted(k for k, v in task_results.items() if isinstance(v, (int, float)))
        raise KeyError(f"metric {metric!r} not in results[{task!r}]; available numeric metrics: {available}")
    value = task_results[metric]
    if not isinstance(value, (int, float)):
        raise TypeError(f"results[{task!r}][{metric!r}] is not numeric: {value!r}")
    return float(value)


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


def evaluate_gate(results: dict, baseline: dict) -> GateReport:
    """Compare a results dict against a parsed baseline dict; return a report.

    Baseline schema::

        {
          "provenance": {...free-form metadata...},
          "tasks": {
            "<task>": {
              "expected_samples": <int>,             # optional
              "metrics": {
                "<metric-key>": {"min": <float>,      # optional coarse floor
                                 "reference": <float>, # optional
                                 "tolerance": <float>} # required iff reference set
              }
            }
          }
        }
    """
    report = GateReport()
    tasks = baseline.get("tasks") or {}
    if not tasks:
        raise ValueError("baseline has no 'tasks' entries to check")

    for task, spec in tasks.items():
        expected_samples = spec.get("expected_samples")
        if expected_samples is not None:
            report.sample_checks.append(
                SampleCheck(task=task, expected=int(expected_samples), observed=effective_sample_count(results, task))
            )
        metrics = spec.get("metrics") or {}
        if not metrics:
            raise ValueError(f"baseline task {task!r} has no 'metrics' to check")
        for metric, thresholds in metrics.items():
            reference = thresholds.get("reference")
            tolerance = thresholds.get("tolerance")
            if reference is not None and tolerance is None:
                raise ValueError(f"baseline {task}/{metric}: 'reference' requires 'tolerance'")
            check = MetricCheck(
                task=task,
                metric=metric,
                observed=None,
                min_threshold=thresholds.get("min"),
                reference=reference,
                tolerance=tolerance,
            )
            try:
                check.observed = get_metric(results, task, metric)
            except (KeyError, TypeError) as exc:
                check.error = str(exc)
            report.metric_checks.append(check)
    return report


def load_baseline(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gate_results_file(results_path: str, baseline_path: str) -> GateReport:
    return evaluate_gate(load_results(results_path), load_baseline(baseline_path))
