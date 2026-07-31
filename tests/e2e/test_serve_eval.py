# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for the serve-and-eval runner (eval/serve_eval).

Only the subtle behaviors that would SILENTLY break the runner: model_args
semantics, the bare ``--apply_chat_template`` footgun, URL normalization, the run
config boundaries, the readiness poll, the provider factory, and the marin-serve
PTY parse. Each runs in-process against stdlib stubs -- a stub HTTP server and a PTY.
"""

import json
import os
import pty
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from eval.robust_api import request_failure_placeholder
from eval.serve_eval.config import RunConfig
from eval.serve_eval.providers import (
    EndpointProvider,
    MarinServeProvider,
    ServedModel,
    build_provider,
    wait_for_models,
)
from eval.serve_eval.run import (
    LOCAL_CHAT_COMPLETIONS,
    LOCAL_COMPLETIONS,
    build_eval_argv,
    build_model_args,
    endpoint_url,
    main,
)

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def test_request_failure_placeholder_identifies_infrastructure_error():
    marker = request_failure_placeholder(TimeoutError("endpoint timed out"))

    assert marker.startswith("[EVALCHEMY_INFRASTRUCTURE_ERROR]")
    assert "TimeoutError" in marker
    assert "endpoint timed out" in marker


# --- model_args semantics (lm-eval parses to a dict; the test compares parsed keys) --


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
    served = ServedModel(base_url="http://h/v1", model="m")
    cfg = RunConfig.load(None, tasks=["gsm8k"], apply_chat_template=True)
    argv = build_eval_argv(served, cfg, "/out", limit=None, extra_args=[], python="python")
    assert "--apply_chat_template" in argv
    following = argv[argv.index("--apply_chat_template") + 1 :]
    assert not following or following[0].startswith("--")
    assert "True" not in argv


# --- run config boundaries ----------------------------------------------------


def test_config_rejects_unknown_key():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"tasks": ["gsm8k"], "typo_field": 1})


def test_shipped_config_parses():
    cfg = RunConfig.load(os.path.join(_HERE, "eval", "serve_eval", "configs", "qwen-tiny.yaml"))
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.apply_chat_template is True
    assert cfg.tpu == "v5litepod-8"


# --- providers: readiness poll + factory fail-fast ----------------------------


class _ModelsHandler(BaseHTTPRequestHandler):
    ready_after = 0
    _hits: ClassVar[dict[str, int]] = {"n": 0}
    telemetry_status = 200
    telemetry_batches: ClassVar[list[dict]] = []

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

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/v1/telemetry"):
            length = int(self.headers["Content-Length"])
            batch = json.loads(self.rfile.read(length))
            _ModelsHandler.telemetry_batches.append(batch)
            self.send_response(_ModelsHandler.telemetry_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if _ModelsHandler.telemetry_status == 200:
                response = {"batch_id": batch["batch_id"], "status": "accepted"}
                self.wfile.write(json.dumps(response).encode())
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_args):  # silence
        pass


def _serve(ready_after=0, telemetry_status=200):
    _ModelsHandler._hits["n"] = 0
    _ModelsHandler.ready_after = ready_after
    _ModelsHandler.telemetry_status = telemetry_status
    _ModelsHandler.telemetry_batches = []
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
    fake = textwrap.dedent(
        r"""
        import time
        print("  job          /app/evalchemy-e2e-test")
        print("        OpenAI:    https://h/proxy/serve.evalchemy-e2e-test/v1")
        print("  Shared capability URL (token in the path):")
        print("    base_url   \x1b[32mhttps://h/proxy/t/TOK123/serve.evalchemy-e2e-test/v1\x1b[0m")
        print("    example    curl https://h/.../models")
        time.sleep(30)
        """
    )
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B")
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


def _fake_iris(tmp_path, list_output: str):
    """A stand-in iris binary that logs every argv line and answers `job list`."""
    log = tmp_path / "iris-calls.log"
    script = tmp_path / "iris"
    script.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{log}"\n'
        'case "$*" in *"job list"*) cat <<"OUT"\n'
        f"{list_output}\n"
        "OUT\n"
        ";; esac\n"
    )
    script.chmod(0o755)
    return str(script), log


def test_exit_stops_by_canonical_id_resolved_from_job_list(tmp_path):
    # Regression: `iris job stop` rejects bare names, so when the job line was never
    # parsed, teardown must resolve /<user>/<job> from `job list` -- not pass the name.
    iris_bin, log = _fake_iris(tmp_path, "  job   /ci-user/evalchemy-e2e-qwen3-0-6b   RUNNING")
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B", iris_bin=iris_bin)
    prov.__exit__(None, None, None)
    calls = log.read_text().splitlines()
    stop_calls = [c for c in calls if "job stop" in c]
    assert stop_calls == ["--cluster marin job stop /ci-user/evalchemy-e2e-qwen3-0-6b"]


def test_exit_without_resolvable_id_never_calls_stop(tmp_path):
    iris_bin, log = _fake_iris(tmp_path, "no jobs")
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B", iris_bin=iris_bin)
    prov.__exit__(None, None, None)
    calls = log.read_text().splitlines() if log.exists() else []
    assert not [c for c in calls if "job stop" in c]


def test_match_line_exports_job_id_to_github_env(tmp_path, monkeypatch):
    env_file = tmp_path / "github_env"
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B")
    prov._match_line("  job          /app/evalchemy-e2e-test")
    assert env_file.read_text() == "IRIS_JOB_ID=/app/evalchemy-e2e-test\n"


# --- full runner telemetry: real HTTP, subprocess, and filesystem boundaries --


def _fake_eval_python(tmp_path: Path, exit_code: int = 0) -> str:
    script = tmp_path / f"fake-eval-python-{exit_code}"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json
            import pathlib
            import sys

            if {exit_code}:
                raise SystemExit({exit_code})
            output_dir = pathlib.Path(sys.argv[sys.argv.index("--output_path") + 1])
            results_dir = output_dir / "served-model"
            results_dir.mkdir(parents=True, exist_ok=True)
            payload = {{
                "results": {{
                    "gsm8k": {{
                        "exact_match,strict-match": 0.75,
                        "exact_match_stderr,strict-match": 0.125,
                    }}
                }},
                "n-samples": {{"gsm8k": {{"original": 4, "effective": 3}}}},
            }}
            (results_dir / "results_test.json").write_text(json.dumps(payload))
            """
        )
    )
    script.chmod(0o755)
    return str(script)


def _runner_args(server: HTTPServer, output_dir: Path, python_bin: str) -> list[str]:
    port = server.server_address[1]
    return [
        "--provider",
        "endpoint",
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        "served-model",
        "--tasks",
        "gsm8k",
        "--output-dir",
        str(output_dir),
        "--python",
        python_bin,
        "--run-id",
        "eval-run-42",
    ]


def _telemetry_records() -> list[dict]:
    return [record for batch in _ModelsHandler.telemetry_batches for record in batch["records"]]


def test_runner_exports_terminal_results_through_real_boundaries(tmp_path):
    server = _serve()
    try:
        output_dir = tmp_path / "results"
        args = _runner_args(server, output_dir, _fake_eval_python(tmp_path))
        port = server.server_address[1]
        result = CliRunner().invoke(main, [*args, "--telemetry-endpoint", f"http://127.0.0.1:{port}/v1/telemetry"])
    finally:
        server.shutdown()

    assert result.exit_code == 0, result.output
    assert "gsm8k  (samples: 3)" in result.output
    assert "exact_match,strict-match" in result.output
    assert (output_dir / "served-model" / "results_test.json").is_file()

    records = _telemetry_records()
    event_names = {record["name"] for record in records if record["kind"] == "event"}
    metric_names = {record["name"] for record in records if record["kind"] != "event"}
    assert all(not name.startswith("evalchemy") for name in metric_names)
    assert {
        "run_started",
        "provider_starting",
        "readiness_terminal",
        "provider_terminal",
        "subprocess_started",
        "subprocess_terminal",
        "cleanup_terminal",
        "results_persisted",
        "run_terminal",
    } <= event_names

    resources = {json.dumps(batch["resource"], sort_keys=True) for batch in _ModelsHandler.telemetry_batches}
    assert resources == {
        json.dumps(
            {
                "service": "evalchemy",
                "attributes": {
                    "model": "served-model",
                    "provider": "endpoint",
                    "run_id": "eval-run-42",
                    "tasks": "gsm8k",
                },
            },
            sort_keys=True,
        )
    }
    task_samples = next(record for record in records if record["name"] == "task_samples")
    assert task_samples["value"] == 3
    assert task_samples["attributes"]["task"] == "gsm8k"
    task_metric = next(
        record
        for record in records
        if record["name"] == "task_metric" and record["attributes"]["metric"] == "exact_match,strict-match"
    )
    assert task_metric["value"] == 0.75
    exit_code = next(record for record in records if record["name"] == "subprocess_exit_code")
    assert exit_code["value"] == 0


def test_runner_exports_observed_subprocess_failure_and_cleanup(tmp_path):
    server = _serve()
    try:
        args = _runner_args(server, tmp_path / "results", _fake_eval_python(tmp_path, exit_code=7))
        port = server.server_address[1]
        result = CliRunner().invoke(main, [*args, "--telemetry-endpoint", f"http://127.0.0.1:{port}/v1/telemetry"])
    finally:
        server.shutdown()

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)
    records = _telemetry_records()
    exit_code = next(record for record in records if record["name"] == "subprocess_exit_code")
    assert exit_code["value"] == 7
    failure = next(record for record in records if record["name"] == "failure")
    assert failure["attributes"]["stage"] == "evaluation"
    assert failure["body"]["error_type"] == "RuntimeError"
    terminal = next(record for record in records if record["name"] == "run_terminal")
    assert terminal["attributes"]["state"] == "failed"
    cleanup = next(record for record in records if record["name"] == "cleanup_terminal")
    assert cleanup["attributes"] == {"outcome": "succeeded", "provider": "endpoint"}


@pytest.mark.parametrize("eval_exit_code", [0, 7])
def test_telemetry_delivery_failure_does_not_change_runner_output_or_status(tmp_path, eval_exit_code):
    server = _serve(telemetry_status=503)
    try:
        output_dir = tmp_path / "results"
        args = _runner_args(server, output_dir, _fake_eval_python(tmp_path, exit_code=eval_exit_code))
        without_telemetry = CliRunner().invoke(main, args)
        port = server.server_address[1]
        started = time.monotonic()
        with_broken_telemetry = CliRunner().invoke(
            main,
            [*args, "--telemetry-endpoint", f"http://127.0.0.1:{port}/v1/telemetry"],
        )
        elapsed = time.monotonic() - started
    finally:
        server.shutdown()

    assert with_broken_telemetry.exit_code == without_telemetry.exit_code
    assert with_broken_telemetry.output == without_telemetry.output
    assert elapsed < 3
