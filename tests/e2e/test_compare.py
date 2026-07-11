# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the e2e baseline gate. Pure stdlib; no model needed."""

import os
import time

import pytest

from eval.e2e.compare import (
    effective_sample_count,
    evaluate_gate,
    find_latest_results,
    get_metric,
    load_baseline,
)


def _results(strict=0.35, flexible=0.40, n=20):
    return {
        "results": {"gsm8k": {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}},
        "n-samples": {"gsm8k": {"original": n, "effective": n}},
        "config": {"lm_eval_version": "0.4.12"},
    }


def _baseline(min_strict=0.05, expected_samples=20, reference=None, tolerance=None):
    metric = {"min": min_strict}
    if reference is not None:
        metric["reference"] = reference
        metric["tolerance"] = tolerance
    return {
        "provenance": {"model": "Qwen/Qwen3-0.6B"},
        "tasks": {"gsm8k": {"expected_samples": expected_samples, "metrics": {"exact_match,strict-match": metric}}},
    }


def test_get_metric_reads_value():
    assert get_metric(_results(strict=0.3), "gsm8k", "exact_match,strict-match") == 0.3


def test_get_metric_missing_metric_lists_available():
    with pytest.raises(KeyError) as e:
        get_metric(_results(), "gsm8k", "nope")
    assert "exact_match,strict-match" in str(e.value)


def test_get_metric_missing_task():
    with pytest.raises(KeyError):
        get_metric(_results(), "mmlu", "acc")


def test_effective_sample_count():
    assert effective_sample_count(_results(n=17), "gsm8k") == 17
    assert effective_sample_count(_results(), "absent") is None


def test_effective_sample_count_sample_len_fallback():
    # evalchemy's lm-eval-native path omits top-level `n-samples` and records the
    # count on the task result as `sample_len`; the gate must read that.
    results = {"results": {"gsm8k": {"exact_match,strict-match": 0.2, "sample_len": 20}}}
    assert effective_sample_count(results, "gsm8k") == 20


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
    results = {"results": {"gsm8k": {"exact_match,flexible-extract": 0.4}}, "n-samples": {"gsm8k": {"effective": 20}}}
    report = evaluate_gate(results, _baseline())
    assert not report.ok


def test_gate_tolerance_band():
    ok = evaluate_gate(_results(strict=0.33), _baseline(reference=0.30, tolerance=0.10))
    assert ok.ok
    bad = evaluate_gate(_results(strict=0.50), _baseline(reference=0.30, tolerance=0.10))
    assert not bad.ok


def test_reference_without_tolerance_is_rejected():
    bl = _baseline()
    bl["tasks"]["gsm8k"]["metrics"]["exact_match,strict-match"] = {"min": 0.0, "reference": 0.3}
    with pytest.raises(ValueError):
        evaluate_gate(_results(), bl)


def test_find_latest_results(tmp_path):
    d = tmp_path / "out" / "Qwen__Qwen3-0.6B"
    d.mkdir(parents=True)
    old = d / "results_2020.json"
    old.write_text("{}")
    time.sleep(0.01)
    new = d / "results_2021.json"
    new.write_text("{}")
    os.utime(new, None)
    assert find_latest_results(str(tmp_path / "out")).endswith("results_2021.json")


def test_find_latest_results_none(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_results(str(tmp_path))


def test_shipped_baseline_parses(tmp_path):
    # The committed baseline must be valid against the gate schema, and a run that
    # reproduces its recorded references must PASS. Synthesize observed values FROM
    # the baseline (observed == reference) so this stays green when the baseline is
    # re-recorded with new numbers.
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    baseline = load_baseline(os.path.join(here, "eval", "e2e", "baselines", "qwen3-0.6b.json"))
    spec = baseline["tasks"]["gsm8k"]
    observed = {}
    for metric, thresholds in spec["metrics"].items():
        # A metric at its reference (or, absent a reference, at its floor) must pass.
        observed[metric] = thresholds.get("reference", thresholds.get("min", 0.0))
    synthetic = {
        "results": {"gsm8k": {**observed, "sample_len": spec["expected_samples"]}},
        "config": {"lm_eval_version": "0.4.12"},
    }
    report = evaluate_gate(synthetic, baseline)
    assert report.ok, report.render()
