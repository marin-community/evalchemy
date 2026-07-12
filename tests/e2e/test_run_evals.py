# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for run_evals.main -> build_provider. No model, no network."""

import json

from click.testing import CliRunner

import eval.e2e.run_evals as run_evals
from eval.e2e.eval_args import ServedModel

_RESULTS = {
    "results": {"gsm8k": {"exact_match,strict-match": 0.3, "sample_len": 20}},
    "lm_eval_version": "0.4.12",
}


class _FakeProvider:
    def __enter__(self):
        return ServedModel(base_url="http://unused/v1", model="Qwen/Qwen3-0.6B", tokenizer="Qwen/Qwen3-0.6B")

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, captured):
    def fake_build_provider(provider, model, **kwargs):
        captured["provider"] = provider
        captured["model"] = model
        captured.update(kwargs)
        return _FakeProvider()

    def fake_run_eval(inv, python):
        captured["inv"] = inv

    monkeypatch.setattr(run_evals, "build_provider", fake_build_provider)
    monkeypatch.setattr(run_evals, "run_eval", fake_run_eval)


def _out_with_results(tmp_path):
    out = tmp_path / "out" / "Qwen__Qwen3-0.6B"
    out.mkdir(parents=True)
    (out / "results_2021.json").write_text(json.dumps(_RESULTS))
    return str(tmp_path / "out")


def test_marin_workspace_and_overrides_thread_to_provider(monkeypatch, tmp_path):
    # Regression: main() must pass --marin-workspace/--region/--tpu through to
    # build_provider (marin-serve bundles cwd, so the workspace must be a marin
    # checkout -- previously the arg was dropped and it bundled evalchemy).
    captured = {}
    _patch(monkeypatch, captured)
    out = _out_with_results(tmp_path)
    result = CliRunner().invoke(
        run_evals.main,
        [
            "--provider",
            "marin-serve",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--marin-workspace",
            "/path/to/marin",
            "--region",
            "us-east5",
            "--tpu",
            "v5litepod-8",
            "--config",
            "/nonexistent-config.yaml",
            "--output-dir",
            out,
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["provider"] == "marin-serve"
    assert captured["marin_workspace"] == "/path/to/marin"
    assert captured["region"] == "us-east5"
    assert captured["tpu"] == "v5litepod-8"


def test_run_evals_prints_summary_and_paths(monkeypatch, tmp_path):
    captured = {}
    _patch(monkeypatch, captured)
    out = _out_with_results(tmp_path)
    result = CliRunner().invoke(
        run_evals.main,
        [
            "--provider",
            "endpoint",
            "--base-url",
            "http://x/v1",
            "--model",
            "m",
            "--config",
            "/none",
            "--output-dir",
            out,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "gsm8k" in result.output
    assert "results:" in result.output
    assert "validate check" in result.output  # points the user at the CI gate


def test_limit_defaults_to_200_and_zero_means_full(monkeypatch, tmp_path):
    captured = {}
    _patch(monkeypatch, captured)
    out = _out_with_results(tmp_path)
    base = [
        "--provider",
        "endpoint",
        "--base-url",
        "http://x/v1",
        "--model",
        "m",
        "--config",
        "/none",
        "--output-dir",
        out,
    ]
    # no --limit and no config -> human default of 200
    CliRunner().invoke(run_evals.main, base)
    assert captured["inv"].limit == 200
    # --limit 0 -> full task (lm-eval reads no limit as "all samples")
    CliRunner().invoke(run_evals.main, base + ["--limit", "0"])
    assert captured["inv"].limit is None


def test_extra_args_after_dashdash_pass_through(monkeypatch, tmp_path):
    captured = {}
    _patch(monkeypatch, captured)
    out = _out_with_results(tmp_path)
    result = CliRunner().invoke(
        run_evals.main,
        [
            "--provider",
            "endpoint",
            "--base-url",
            "http://x/v1",
            "--model",
            "m",
            "--config",
            "/none",
            "--output-dir",
            out,
            "--tasks",
            "MATH500",
            "--",
            "--num_samples",
            "8",
            "--pass_at_k",
            "1,8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert list(captured["inv"].extra_args) == ["--num_samples", "8", "--pass_at_k", "1,8"]
