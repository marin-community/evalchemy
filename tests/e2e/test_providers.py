# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the e2e providers. Uses a stdlib stub server; no model/Marin."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from eval.e2e.providers import (
    EndpointProvider,
    MarinServeProvider,
    build_provider,
    wait_for_models,
)


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


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_wait_for_models_succeeds():
    _ModelsHandler._hits["n"] = 0
    _ModelsHandler.ready_after = 0
    server = _serve()
    try:
        port = server.server_address[1]
        wait_for_models(f"http://127.0.0.1:{port}/v1", None, timeout_s=5, interval_s=0.1)
    finally:
        server.shutdown()


def test_wait_for_models_times_out():
    _ModelsHandler._hits["n"] = 0
    _ModelsHandler.ready_after = 10_000  # never ready
    server = _serve()
    try:
        port = server.server_address[1]
        with pytest.raises(TimeoutError):
            wait_for_models(f"http://127.0.0.1:{port}/v1", None, timeout_s=0.5, interval_s=0.1)
    finally:
        server.shutdown()


def test_endpoint_provider_yields_normalized_served_model():
    _ModelsHandler._hits["n"] = 0
    _ModelsHandler.ready_after = 0
    server = _serve()
    try:
        port = server.server_address[1]
        # Pass a URL that redundantly includes the adapter path; provider must
        # normalize to the /v1 root.
        prov = EndpointProvider(
            base_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            model="m",
            readiness_timeout_s=5,
        )
        with prov as served:
            assert served.base_url == f"http://127.0.0.1:{port}/v1"
            assert served.model == "m"
            assert served.tokenizer == "m"  # defaults to model
    finally:
        server.shutdown()


def test_endpoint_provider_can_skip_readiness():
    prov = EndpointProvider(base_url="http://unreachable.invalid/v1", model="m", wait_ready=False)
    with prov as served:
        assert served.base_url == "http://unreachable.invalid/v1"


def test_build_provider_endpoint_requires_base_url():
    with pytest.raises(ValueError):
        build_provider("endpoint", "m")


def test_build_provider_unknown():
    with pytest.raises(ValueError):
        build_provider("nope", "m")


def test_marin_serve_command_is_deterministic():
    prov = build_provider(
        "marin-serve", "Qwen/Qwen3-0.6B", cluster="marin", tpu="v6e-8", name="job1", region="europe-west4"
    )
    assert isinstance(prov, MarinServeProvider)
    cmd = prov._command("job1")
    assert cmd[0] == "marin-serve"
    assert "--cluster" in cmd and cmd[cmd.index("--cluster") + 1] == "marin"
    assert "--name" in cmd and cmd[cmd.index("--name") + 1] == "job1"
    assert "--wait" in cmd  # blocks holding the tunnel; provider runs it in background
    # mint mode: default access is a PUBLIC capability URL.
    assert cmd[cmd.index("--access") + 1] == "link"
    assert cmd[cmd.index("--region") + 1] == "europe-west4"


def test_marin_serve_default_job_name_is_dot_free():
    # marin-serve rejects '.' in the endpoint name (/serve/<name>), so a model id
    # like Qwen3-0.6B must not leak a dot into the job name.
    name = MarinServeProvider.default_job_name("Qwen/Qwen3-0.6B")
    assert "." not in name and "/" not in name
    assert name == "evalchemy-e2e-qwen3-0-6b"


def test_marin_serve_teardown_uses_toplevel_cluster_flag():
    # `iris --cluster X job stop <id>` -- --cluster is a top-level flag, not a
    # `job stop` option. Guard the exact ordering.
    prov = MarinServeProvider(model="Qwen/Qwen3-0.6B", cluster="marin", name="job1")
    prov._job_id = "/app/evalchemy-e2e-qwen3-0-6b"
    calls = {}

    def _fake_run(cmd, **kwargs):
        calls["cmd"] = cmd

        class _R:
            returncode = 0

        return _R()

    import subprocess as _sp

    orig = _sp.run
    _sp.run = _fake_run
    try:
        prov.__exit__(None, None, None)
    finally:
        _sp.run = orig
    cmd = calls["cmd"]
    assert cmd == ["iris", "--cluster", "marin", "job", "stop", "/app/evalchemy-e2e-qwen3-0-6b"]


def test_read_until_ready_parses_capability_url_from_a_blocking_child():
    # Regression for the PTY fix: marin-serve prints the capability `base_url`,
    # then blocks holding the tunnel WITHOUT exiting. `_read_until_ready` must
    # return that URL (and capture the job id), stripping ANSI colour, while the
    # child is still alive -- not deadlock waiting for more output / an exit.
    import os
    import pty
    import subprocess
    import sys
    import textwrap

    # A stand-in serve CLI: emits the marin-serve lines (base_url is ANSI-coloured)
    # then sleeps, mimicking the real "tunnel held open" block.
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
