# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pydantic models at the JSON/YAML boundaries."""

import os

import pytest
from pydantic import ValidationError

from eval.e2e.models import Baseline, E2EConfig, EvalResults, MetricThreshold

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _results(strict=0.3, flexible=0.5, n=20, use_sample_len=False):
    task = {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}
    doc = {"results": {"gsm8k": task}, "lm_eval_version": "0.4.12"}
    if use_sample_len:
        task["sample_len"] = n
    else:
        doc["n-samples"] = {"gsm8k": {"original": n, "effective": n}}
    return EvalResults.model_validate(doc)


def test_results_metric_and_numeric_metrics():
    r = _results(strict=0.3)
    assert r.metric("gsm8k", "exact_match,strict-match") == 0.3
    assert r.metric("gsm8k", "nope") is None
    assert r.metric("absent-task", "x") is None
    assert "exact_match,strict-match" in r.numeric_metrics("gsm8k")


def test_results_sample_count_from_n_samples():
    assert _results(n=17).sample_count("gsm8k") == 17
    assert _results().sample_count("absent") is None


def test_results_sample_count_sample_len_fallback():
    # evalchemy's lm-eval-native path omits top-level `n-samples`, using `sample_len`.
    assert _results(n=20, use_sample_len=True).sample_count("gsm8k") == 20


def test_results_ignores_extra_fields():
    # lm-eval writes a big doc; the model keeps only what it declares.
    r = EvalResults.model_validate({"results": {}, "git_hash": "abc", "pretty_env_info": "..."})
    assert not hasattr(r, "git_hash") or r.model_extra is None


def test_metric_threshold_reference_requires_tolerance():
    MetricThreshold(min=0.1)  # ok
    MetricThreshold(reference=0.3, tolerance=0.1)  # ok
    with pytest.raises(ValidationError):
        MetricThreshold(reference=0.3)  # reference without tolerance is meaningless


def test_baseline_round_trip(tmp_path):
    src = {
        "provenance": {"model": "Qwen/Qwen3-0.6B"},
        "tasks": {"gsm8k": {"metrics": {"exact_match,strict-match": {"min": 0.05}}, "expected_samples": 20}},
    }
    b = Baseline.model_validate(src)
    path = tmp_path / "bl.json"
    b.save(str(path))
    b2 = Baseline.load(str(path))
    assert b2.tasks["gsm8k"].metrics["exact_match,strict-match"].min == 0.05
    assert b2.tasks["gsm8k"].expected_samples == 20


def test_shipped_config_and_baseline_parse():
    cfg = E2EConfig.load(os.path.join(_HERE, "eval", "e2e", "config.yaml"))
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.eval.apply_chat_template is True
    # the hyphenated `marin-serve` yaml key maps to the marin_serve field via alias.
    assert cfg.provider.marin_serve.tpu == "v5litepod-8"
    baseline = Baseline.load(cfg.baseline)
    assert "gsm8k" in baseline.tasks


def test_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        E2EConfig.model_validate({"eval": {"tasks": ["gsm8k"], "typo_field": 1}})
