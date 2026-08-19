# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the regression gate (eval/regression).

The pass/fail decisions the gate must get right, the spec's parse-time boundaries,
and a validate CLI round trip -- the CI entrypoints. Each runs in-process on
hand-built results and specs.
"""

import json
import os

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from eval.regression.validate import DEFAULT_SPEC, GateSpec, MetricThreshold, cli, evaluate_gate
from eval.serve_eval.results import EvalResults

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- the CI gate (gate.evaluate_gate) -----------------------------------------


def _results(strict=0.35, flexible=0.40, n=20):
    task = {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}
    doc = {"results": {"gsm8k": task}, "n-samples": {"gsm8k": {"original": n, "effective": n}}}
    return EvalResults.model_validate(doc)


def _spec(min_strict=0.05, expected_samples=20, reference=None, tolerance=None):
    metric = {"min": min_strict}
    if reference is not None:
        metric["reference"] = reference
        metric["tolerance"] = tolerance
    return GateSpec.model_validate(
        {
            "provenance": {"model": "Qwen/Qwen3-0.6B"},
            "tasks": {"gsm8k": {"expected_samples": expected_samples, "metrics": {"exact_match,strict-match": metric}}},
        }
    )


def test_gate_healthy_run_passes():
    report = evaluate_gate(_results(strict=0.35, n=20), _spec(min_strict=0.05, expected_samples=20))
    assert report.ok, report.render()


def test_gate_metric_below_floor_fails():
    report = evaluate_gate(_results(strict=0.02), _spec(min_strict=0.05))
    assert not report.ok
    assert any("strict-match" in f for f in report.failures())


def test_gate_wrong_sample_count_fails():
    # The endpoint answered only 12 of 20 -> a dropped-request / connectivity failure.
    report = evaluate_gate(_results(n=12), _spec(expected_samples=20))
    assert not report.ok
    assert any("samples" in f for f in report.failures())


def test_gate_missing_metric_fails():
    results = EvalResults.model_validate(
        {"results": {"gsm8k": {"exact_match,flexible-extract": 0.4}}, "n-samples": {"gsm8k": {"effective": 20}}}
    )
    report = evaluate_gate(results, _spec())
    assert not report.ok
    assert any("not in results" in f for f in report.failures())


def test_gate_optional_tolerance_band_bounds_both_sides():
    within = evaluate_gate(_results(strict=0.33), _spec(reference=0.30, tolerance=0.10))
    assert within.ok
    too_high = evaluate_gate(_results(strict=0.50), _spec(reference=0.30, tolerance=0.10))
    assert not too_high.ok


def test_gate_empty_spec_raises():
    with pytest.raises(ValueError):
        evaluate_gate(_results(), GateSpec.model_validate({"tasks": {}}))


# --- spec parse-time boundaries + round trip ----------------------------------


def test_metric_threshold_reference_without_tolerance_raises():
    # A reference band with no tolerance is meaningless; reject at parse time.
    MetricThreshold(reference=0.3, tolerance=0.1)  # ok
    with pytest.raises(ValidationError):
        MetricThreshold(reference=0.3)


def test_shipped_spec_parses():
    assert "gsm8k" in GateSpec.load(DEFAULT_SPEC).tasks


def test_shipped_smoke_spec_allows_score_improvement():
    results = _results(strict=0.32, flexible=0.64, n=100)
    report = evaluate_gate(results, GateSpec.load(DEFAULT_SPEC))
    assert report.ok, report.render()


def test_spec_save_load_round_trip(tmp_path):
    src = {
        "provenance": {"model": "Qwen/Qwen3-0.6B"},
        "tasks": {"gsm8k": {"metrics": {"exact_match,strict-match": {"min": 0.05}}, "expected_samples": 20}},
    }
    path = tmp_path / "spec.json"
    GateSpec.model_validate(src).save(str(path))
    loaded = GateSpec.load(str(path))
    assert loaded.tasks["gsm8k"].metrics["exact_match,strict-match"].min == 0.05
    assert loaded.tasks["gsm8k"].expected_samples == 20


# --- validate CLI (the CI entrypoint) -----------------------------------------

_RESULTS = {
    "results": {"gsm8k": {"exact_match,strict-match": 0.30, "exact_match,flexible-extract": 0.50, "sample_len": 20}},
    "lm_eval_version": "0.4.12",
    "model_source": "local-chat-completions",
    "config": {"limit": 20, "random_seed": 1234},
}
_SPEC = {
    "provenance": {"model": "Qwen/Qwen3-0.6B"},
    "tasks": {
        "gsm8k": {
            "metrics": {"exact_match,strict-match": {"min": 0.05}, "exact_match,flexible-extract": {"min": 0.25}},
            "expected_samples": 20,
        }
    },
}


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)


def test_validate_record_then_check_round_trips(tmp_path):
    res = _write(tmp_path, "results_x.json", _RESULTS)
    out = str(tmp_path / "spec.json")
    recorded = CliRunner().invoke(cli, ["record", "--results", res, "--spec", out, "--model", "m"])
    assert recorded.exit_code == 0, recorded.output
    # The run that produced the spec must pass its own gate.
    checked = CliRunner().invoke(cli, ["check", "--results", res, "--spec", out])
    assert checked.exit_code == 0, checked.output


def test_validate_check_exits_nonzero_on_broken_run(tmp_path):
    broken = dict(_RESULTS)
    broken["results"] = {
        "gsm8k": {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 0.0, "sample_len": 20}
    }
    res = _write(tmp_path, "results_x.json", broken)
    spec = _write(tmp_path, "spec.json", _SPEC)
    result = CliRunner().invoke(cli, ["check", "--results", res, "--spec", spec])
    assert result.exit_code == 1


def test_recorded_tolerance_band_catches_drift_a_floor_would_miss(tmp_path):
    # The whole point of the tight gate: a serving change that moves the score
    # (here 0.30 -> 0.40) must FAIL even though it clears the wide floor.
    res = _write(tmp_path, "results_x.json", _RESULTS)  # strict 0.30, flexible 0.50
    out = str(tmp_path / "spec.json")
    rec = CliRunner().invoke(cli, ["record", "--results", res, "--spec", out, "--model", "m", "--tolerance", "0.02"])
    assert rec.exit_code == 0, rec.output
    assert CliRunner().invoke(cli, ["check", "--results", res, "--spec", out]).exit_code == 0  # its own run passes

    drifted = dict(_RESULTS)
    drifted["results"] = {
        "gsm8k": {"exact_match,strict-match": 0.40, "exact_match,flexible-extract": 0.50, "sample_len": 20}
    }
    res2 = _write(tmp_path, "results_y.json", drifted)
    assert CliRunner().invoke(cli, ["check", "--results", res2, "--spec", out]).exit_code == 1
