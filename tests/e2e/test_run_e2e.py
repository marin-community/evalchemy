# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Wiring tests for run_e2e.main -> build_provider. No model, no network."""

import eval.e2e.run_e2e as run_e2e
from eval.e2e.eval_args import ServedModel


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

    monkeypatch.setattr(run_e2e, "build_provider", fake_build_provider)
    monkeypatch.setattr(run_e2e, "_run_eval", lambda inv, python: None)
    monkeypatch.setattr(run_e2e, "_finish", lambda *a, **k: 0)


def test_marin_workspace_is_threaded_to_provider(monkeypatch, tmp_path):
    # Regression: main() must pass --marin-workspace through to build_provider.
    # It previously dropped it, so marin-serve bundled the caller's cwd (evalchemy)
    # instead of a marin checkout and the container build failed.
    captured = {}
    _patch(monkeypatch, captured)
    ws = str(tmp_path / "marin_ws")
    rc = run_e2e.main(
        [
            "--provider",
            "marin-serve",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--marin-workspace",
            ws,
            "--tpu",
            "v5litepod-8",
            "--config",
            "/nonexistent-config.yaml",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert captured["provider"] == "marin-serve"
    assert captured["marin_workspace"] == ws
    assert captured["tpu"] == "v5litepod-8"


def test_cli_overrides_reach_provider(monkeypatch, tmp_path):
    # --region / --access / --cluster overrides must reach the provider too.
    captured = {}
    _patch(monkeypatch, captured)
    rc = run_e2e.main(
        [
            "--provider",
            "marin-serve",
            "--model",
            "Qwen/Qwen3-0.6B",
            "--region",
            "us-east5",
            "--access",
            "private",
            "--cluster",
            "marin",
            "--config",
            "/nonexistent-config.yaml",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert captured["region"] == "us-east5"
    assert captured["access"] == "private"
    assert captured["cluster"] == "marin"
