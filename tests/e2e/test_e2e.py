# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the e2e eval harness (eval/e2e).

One file, only the subtle behaviors that would SILENTLY break the harness or its
CI gate: the gate pass/fail decisions, URL normalization, the bare
``--apply_chat_template`` footgun, model_args semantics, sample-count fallback,
pydantic config boundaries, limit resolution, the marin-serve PTY parse, the
readiness poll, and a validate CLI round trip. Pure/at-a-boundary; no model,
Marin, or lm-eval needed.
"""

import json
import os
import pty
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from eval.e2e.compare import evaluate_gate
from eval.e2e.eval_args import (
    LOCAL_CHAT_COMPLETIONS,
    LOCAL_COMPLETIONS,
    EvalInvocation,
    ServedModel,
    build_eval_argv,
    build_model_args,
    endpoint_url,
)
from eval.e2e.models import Baseline, E2EConfig, EvalResults, MetricThreshold
from eval.e2e.providers import EndpointProvider, MarinServeProvider, build_provider, wait_for_models
from eval.e2e.validate import cli

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- the CI gate (compare.evaluate_gate) --------------------------------------


def _results(strict=0.35, flexible=0.40, n=20):
    task = {"exact_match,strict-match": strict, "exact_match,flexible-extract": flexible}
    doc = {"results": {"gsm8k": task}, "n-samples": {"gsm8k": {"original": n, "effective": n}}}
    return EvalResults.model_validate(doc)


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


def test_gate_healthy_run_passes():
    report = evaluate_gate(_results(strict=0.35, n=20), _baseline(min_strict=0.05, expected_samples=20))
    assert report.ok, report.render()


def test_gate_metric_below_floor_fails():
    report = evaluate_gate(_results(strict=0.02), _baseline(min_strict=0.05))
    assert not report.ok
    assert any("strict-match" in f for f in report.failures())


def test_gate_wrong_sample_count_fails():
    # The endpoint only answered 12 of 20 -> a connectivity failure, not a score.
    report = evaluate_gate(_results(n=12), _baseline(expected_samples=20))
    assert not report.ok
    assert any("samples" in f for f in report.failures())


def test_gate_missing_metric_fails():
    results = EvalResults.model_validate(
        {"results": {"gsm8k": {"exact_match,flexible-extract": 0.4}}, "n-samples": {"gsm8k": {"effective": 20}}}
    )
    report = evaluate_gate(results, _baseline())
    assert not report.ok
    assert any("not in results" in f for f in report.failures())


def test_gate_optional_tolerance_band_bounds_both_sides():
    within = evaluate_gate(_results(strict=0.33), _baseline(reference=0.30, tolerance=0.10))
    assert within.ok
    too_high = evaluate_gate(_results(strict=0.50), _baseline(reference=0.30, tolerance=0.10))
    assert not too_high.ok


def test_gate_empty_baseline_raises():
    with pytest.raises(ValueError):
        evaluate_gate(_results(), Baseline.model_validate({"tasks": {}}))


# --- endpoint URL normalization -----------------------------------------------


@pytest.mark.parametrize(
    "base_url, adapter, expected",
    [
        ("http://h/v1", LOCAL_CHAT_COMPLETIONS, "http://h/v1/chat/completions"),
        ("http://h/v1/", LOCAL_COMPLETIONS, "http://h/v1/completions"),
        # A caller who redundantly included the adapter path must not get it twice.
        ("http://h/v1/chat/completions", LOCAL_CHAT_COMPLETIONS, "http://h/v1/chat/completions"),
        ("http://h/v1/completions", LOCAL_COMPLETIONS, "http://h/v1/completions"),
    ],
)
def test_endpoint_url_appends_adapter_path_exactly_once(base_url, adapter, expected):
    assert endpoint_url(base_url, adapter) == expected


# --- model_args semantics (lm-eval parses to a dict; order is NOT the contract) --


def _parse_model_args(text):
    return dict(pair.split("=", 1) for pair in text.split(","))


def test_model_args_carry_v1_rooted_chat_url_and_no_tokenized_requests():
    served = ServedModel(base_url="http://127.0.0.1:8000/v1", model="served-model", api_key="k", tokenizer="tok")
    parsed = _parse_model_args(build_model_args(served, LOCAL_CHAT_COMPLETIONS, extra={"num_concurrent": 2}))
    assert parsed["base_url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert parsed["tokenized_requests"] == "False"
    assert parsed["api_key"] == "k"
    assert parsed["tokenizer"] == "tok"
    assert parsed["num_concurrent"] == "2"


def test_model_args_omit_api_key_and_tokenizer_when_unset():
    served = ServedModel(base_url="http://h/v1", model="m")
    parsed = _parse_model_args(build_model_args(served, LOCAL_COMPLETIONS))
    assert "api_key" not in parsed
    assert "tokenizer" not in parsed
    assert parsed["base_url"] == "http://h/v1/completions"


def test_model_args_reject_comma_in_value():
    # A comma would be read as a pair boundary, silently dropping the value.
    served = ServedModel(base_url="http://h/v1", model="m")
    with pytest.raises(ValueError):
        build_model_args(served, LOCAL_COMPLETIONS, extra={"bad": "a,b"})


def test_apply_chat_template_flag_is_bare_never_a_value():
    # The parser option is nargs="?", const=True: a following value would be read
    # as a chat-*template name*, not the boolean. The flag must appear bare.
    inv = EvalInvocation(
        served=ServedModel(base_url="http://h/v1", model="m"),
        tasks=["gsm8k"],
        output_path="/out",
        apply_chat_template=True,
    )
    argv = build_eval_argv(inv)
    assert "--apply_chat_template" in argv
    following = argv[argv.index("--apply_chat_template") + 1 :]
    assert not following or following[0].startswith("--")
    assert "True" not in argv


# --- pydantic config boundaries -----------------------------------------------


def test_metric_threshold_reference_without_tolerance_raises():
    # A reference band with no tolerance is meaningless; reject at parse time.
    MetricThreshold(reference=0.3, tolerance=0.1)  # ok
    with pytest.raises(ValidationError):
        MetricThreshold(reference=0.3)


def test_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        E2EConfig.model_validate({"tasks": ["gsm8k"], "typo_field": 1})


def test_shipped_config_and_baseline_parse():
    cfg = E2EConfig.load(os.path.join(_HERE, "eval", "e2e", "qwen-tiny.yaml"))
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.apply_chat_template is True
    assert cfg.tpu == "v5litepod-8"
    assert "gsm8k" in Baseline.load(cfg.baseline).tasks


def test_baseline_save_load_round_trip(tmp_path):
    src = {
        "provenance": {"model": "Qwen/Qwen3-0.6B"},
        "tasks": {"gsm8k": {"metrics": {"exact_match,strict-match": {"min": 0.05}}, "expected_samples": 20}},
    }
    path = tmp_path / "bl.json"
    Baseline.model_validate(src).save(str(path))
    loaded = Baseline.load(str(path))
    assert loaded.tasks["gsm8k"].metrics["exact_match,strict-match"].min == 0.05
    assert loaded.tasks["gsm8k"].expected_samples == 20


# --- providers: readiness poll + factory fail-fast ----------------------------


class _ModelsHandler(BaseHTTPRequestHandler):
    ready_after = 0
    _hits = {"n": 0}

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/models"):
            _ModelsHandler._hits["n"] += 1
            if _ModelsHandler._hits["n"] > _ModelsHandler.ready_after:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"data": []}')
                return
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):  # silence
        pass


def _serve(ready_after=0):
    _ModelsHandler._hits["n"] = 0
    _ModelsHandler.ready_after = ready_after
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_wait_for_models_returns_when_ready():
    server = _serve(ready_after=0)
    try:
        port = server.server_address[1]
        wait_for_models(f"http://127.0.0.1:{port}/v1", None, timeout_s=5, interval_s=0.1)
    finally:
        server.shutdown()


def test_wait_for_models_times_out_when_never_ready():
    server = _serve(ready_after=10_000)
    try:
        port = server.server_address[1]
        with pytest.raises(TimeoutError):
            wait_for_models(f"http://127.0.0.1:{port}/v1", None, timeout_s=0.5, interval_s=0.1)
    finally:
        server.shutdown()


def test_endpoint_provider_yields_normalized_served_model():
    server = _serve(ready_after=0)
    try:
        port = server.server_address[1]
        # Pass a URL that redundantly includes the adapter path; the provider must
        # normalize to the /v1 root and default the tokenizer to the model.
        prov = EndpointProvider(
            base_url=f"http://127.0.0.1:{port}/v1/chat/completions", model="m", readiness_timeout_s=5
        )
        with prov as served:
            assert served.base_url == f"http://127.0.0.1:{port}/v1"
            assert served.model == "m"
            assert served.tokenizer == "m"
    finally:
        server.shutdown()


def test_build_provider_endpoint_requires_base_url():
    with pytest.raises(ValueError):
        build_provider("endpoint", "m")


def test_build_provider_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_provider("nope", "m")


def test_marin_serve_default_job_name_is_dot_free():
    # marin-serve rejects '.' in the endpoint name (/serve/<name>), so a model id
    # like Qwen3-0.6B must not leak a dot into the job name.
    name = MarinServeProvider.default_job_name("Qwen/Qwen3-0.6B")
    assert "." not in name and "/" not in name
    assert name == "evalchemy-e2e-qwen3-0-6b"


def test_read_until_ready_parses_capability_url_from_a_blocking_child():
    # Regression for the PTY fix: marin-serve prints the capability `base_url`,
    # then blocks holding the tunnel WITHOUT exiting. _read_until_ready must return
    # that URL (and capture the job id), stripping ANSI colour, while the child is
    # still alive -- not deadlock waiting for more output / an exit.
    fake = textwrap.dedent(r"""
        import time
        print("  job          /app/evalchemy-e2e-test")
        print("        OpenAI:    https://h/proxy/serve.evalchemy-e2e-test/v1")
        print("  Shared capability URL (token in the path):")
        print("    base_url   \x1b[32mhttps://h/proxy/t/TOK123/serve.evalchemy-e2e-test/v1\x1b[0m")
        print("    example    curl https://h/.../models")
        time.sleep(30)
        """)
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B", access="link")
    master, slave = pty.openpty()
    prov._master_fd = master
    prov._proc = subprocess.Popen(
        [sys.executable, "-c", fake], stdout=slave, stderr=slave, stdin=subprocess.DEVNULL, close_fds=True
    )
    os.close(slave)
    try:
        url = prov._read_until_ready(15.0)
        assert url == "https://h/proxy/t/TOK123/serve.evalchemy-e2e-test/v1"  # ANSI stripped
        assert prov._job_id == "/app/evalchemy-e2e-test"
        assert prov._proc.poll() is None  # returned while the child is still blocking
    finally:
        prov._proc.kill()
        prov._proc.wait()
        os.close(master)


# --- validate CLI (the CI entrypoint) -----------------------------------------

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
    path = tmp_path / name
    path.write_text(json.dumps(obj))
    return str(path)


def test_validate_record_then_check_round_trips(tmp_path):
    res = _write(tmp_path, "results_x.json", _RESULTS)
    out = str(tmp_path / "bl.json")
    recorded = CliRunner().invoke(cli, ["record", "--results", res, "--baseline", out, "--model", "m"])
    assert recorded.exit_code == 0, recorded.output
    # The run that produced the baseline must pass its own gate.
    checked = CliRunner().invoke(cli, ["check", "--results", res, "--baseline", out])
    assert checked.exit_code == 0, checked.output


def test_validate_check_exits_nonzero_on_broken_run(tmp_path):
    broken = dict(_RESULTS)
    broken["results"] = {
        "gsm8k": {"exact_match,strict-match": 0.0, "exact_match,flexible-extract": 0.0, "sample_len": 20}
    }
    res = _write(tmp_path, "results_x.json", broken)
    bl = _write(tmp_path, "baseline.json", _BASELINE)
    result = CliRunner().invoke(cli, ["check", "--results", res, "--baseline", bl])
    assert result.exit_code == 1
