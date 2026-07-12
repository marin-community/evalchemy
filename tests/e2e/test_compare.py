# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the e2e gate (compare.evaluate_gate). Pure; no model needed."""

import pytest

from eval.e2e.compare import evaluate_gate
from eval.e2e.models import Baseline, EvalResults


def _results(strict=0.35, flexible=0.40, n=20):
    return EvalResults.model_validate(
        {
            "results": {"gsm8k": {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}},
            "n-samples": {"gsm8k": {"original": n, "effective": n}},
        }
    )


def _baseline(min_strict=0.05, expected_samples=20, reference=None, tolerance=None):
    metric = {"min": min_strict}
    if reference is not None:
        metric["reference"] = reference
        metric["tolerance"] = tolerance
    return Baseline.model_validate(
        {
            "provenance": {"model": "Qwen/Qwen3-0.6B"},
            "tasks": {"gsm8k": {"expected_samples": expected_samples, "metrics": {"exact_match,strict-match": metric}}},
        }
    )


def test_gate_passes_on_healthy_run():
    report = evaluate_gate(_results(strict=0.35, n=20), _baseline(min_strict=0.05, expected_samples=20))
    assert report.ok, report.render()


def test_gate_fails_below_min_floor():
    report = evaluate_gate(_results(strict=0.02), _baseline(min_strict=0.05))
    assert not report.ok
    assert any("strict-match" in f for f in report.failures())


def test_gate_fails_on_wrong_sample_count():
    # The endpoint only answered 12 of 20 -> connectivity failure.
    report = evaluate_gate(_results(n=12), _baseline(expected_samples=20))
    assert not report.ok
    assert any("samples" in f for f in report.failures())


def test_gate_fails_on_missing_metric():
    results = EvalResults.model_validate(
        {"results": {"gsm8k": {"exact_match,flexible-extract": 0.4}}, "n-samples": {"gsm8k": {"effective": 20}}}
    )
    report = evaluate_gate(results, _baseline())
    assert not report.ok
    assert any("not in results" in f for f in report.failures())


def test_gate_optional_tolerance_band():
    ok = evaluate_gate(_results(strict=0.33), _baseline(reference=0.30, tolerance=0.10))
    assert ok.ok
    bad = evaluate_gate(_results(strict=0.50), _baseline(reference=0.30, tolerance=0.10))
    assert not bad.ok


def test_gate_empty_tasks_rejected():
    with pytest.raises(ValueError):
        evaluate_gate(_results(), Baseline.model_validate({"tasks": {}}))
