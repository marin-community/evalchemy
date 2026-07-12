# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CLI tests for `eval.e2e.validate` (check / record). No model, no network."""

import json

from click.testing import CliRunner

from eval.e2e.models import Baseline
from eval.e2e.validate import cli

_RESULTS = {
    "results": {"gsm8k": {"exact_match,strict-match": 0.30, "exact_match,flexible-extract": 0.50, "sample_len": 20}},
    "lm_eval_version": "0.4.12",
    "model_source": "local-chat-completions",
    "config": {"limit": 20, "random_seed": 1234},
}
_BASELINE = {
    "provenance": {"model": "Qwen/Qwen3-0.6B"},
    "tasks": {
        "gsm8k": {
            "metrics": {"exact_match,strict-match": {"min": 0.05}, "exact_match,flexible-extract": {"min": 0.25}},
            "expected_samples": 20,
        }
    },
}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_check_passes_healthy_run(tmp_path):
    res = _write(tmp_path, "results_x.json", _RESULTS)
    bl = _write(tmp_path, "baseline.json", _BASELINE)
    result = CliRunner().invoke(cli, ["check", "--results", res, "--baseline", bl])
    assert result.exit_code == 0, result.output
    assert "GATE: PASS" in result.output


def test_check_fails_and_exits_nonzero(tmp_path):
    broken = dict(_RESULTS)
    broken["results"] = {
        "gsm8k": {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 0.0, "sample_len": 20}
    }
    res = _write(tmp_path, "results_x.json", broken)
    bl = _write(tmp_path, "baseline.json", _BASELINE)
    result = CliRunner().invoke(cli, ["check", "--results", res, "--baseline", bl])
    assert result.exit_code == 1
    assert "GATE: FAIL" in result.output


def test_check_finds_results_in_a_dir(tmp_path):
    sub = tmp_path / "run" / "Qwen__Qwen3-0.6B"
    sub.mkdir(parents=True)
    (sub / "results_2021.json").write_text(json.dumps(_RESULTS))
    bl = _write(tmp_path, "baseline.json", _BASELINE)
    result = CliRunner().invoke(cli, ["check", "--results", str(tmp_path / "run"), "--baseline", bl])
    assert result.exit_code == 0, result.output


def test_record_writes_floor_only_baseline(tmp_path):
    res = _write(tmp_path, "results_x.json", _RESULTS)
    out = str(tmp_path / "new_baseline.json")
    result = CliRunner().invoke(
        cli, ["record", "--results", res, "--baseline", out, "--model", "Qwen/Qwen3-0.6B", "--margin", "0.25"]
    )
    assert result.exit_code == 0, result.output
    b = Baseline.load(out)
    task = b.tasks["gsm8k"]
    # floor-only: min = max(0.05, observed - 0.25); no reference/tolerance band.
    assert task.metrics["exact_match,flexible-extract"].min == 0.25  # 0.50 - 0.25
    assert task.metrics["exact_match,strict-match"].min == 0.05  # max(0.05, 0.30 - 0.25)
    assert task.metrics["exact_match,strict-match"].reference is None
    assert task.observed["exact_match,strict-match"] == 0.3
    assert task.expected_samples == 20
    assert b.provenance["lm_eval_version"] == "0.4.12"


def test_record_then_check_round_trips(tmp_path):
    res = _write(tmp_path, "results_x.json", _RESULTS)
    out = str(tmp_path / "bl.json")
    assert CliRunner().invoke(cli, ["record", "--results", res, "--baseline", out, "--model", "m"]).exit_code == 0
    # the run that produced the baseline must pass its own gate.
    result = CliRunner().invoke(cli, ["check", "--results", res, "--baseline", out])
    assert result.exit_code == 0, result.output
