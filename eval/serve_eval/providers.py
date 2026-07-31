# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Resolve a served model (base_url + model + optional key) for the run.

* :class:`EndpointProvider` -- attach to an already-running OpenAI-compatible
  server at a ``/v1`` root.
* :class:`MarinServeProvider` -- run the ``marin-serve`` CLI (Iris -> vLLM on a TPU
  slice) as a background process, parse the endpoint URL it prints, and stop the
  Iris job in a ``finally`` (killing the CLI leaves the job running). Needs Marin +
  cluster access + a TPU slice.

``marin-serve iris`` returns a capability URL with a scoped token in its path, so
no bearer header is needed.
"""

from __future__ import annotations

import errno
import logging
import os
import pty
import re
import select
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from eval.serve_eval.observability import cleanup_terminal, failure as record_failure, readiness_terminal

logger = logging.getLogger("eval.serve_eval.providers")

# Once vLLM is ready, marin-serve prints a capability URL:
#     base_url   https://iris.oa.dev/proxy/t/<token>/serve.<name>/v1
# The token is in the path, so no auth header is needed.
_CAPABILITY_LINE = re.compile(r"^\s*base_url\s+(?P<url>https?://\S+)")
# The submitted job id is printed as "  job          /app/<name>".
_JOB_LINE = re.compile(r"^\s*job\s+(?P<job>/\S+)\s*$")
# marin-serve runs on a PTY (see __enter__), so it may color its output; strip
# ANSI CSI/SGR escapes before matching the lines above.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# lm-eval requires a non-empty api_key; the capability URL already carries the
# credential in its path, so any placeholder works.
_CAPABILITY_API_KEY = "capability-url-carries-credential"


@dataclass(frozen=True)
class ServedModel:
    """A model served behind an OpenAI-compatible endpoint.

    ``base_url`` is the ``.../v1`` API root. ``api_key`` is sent by lm-eval as
    ``Authorization: Bearer <api_key>`` (carries an IAP bearer for the Marin
    controller proxy, or any server that wants a key).
    """

    base_url: str
    model: str
    api_key: Optional[str] = None
    tokenizer: Optional[str] = None


def api_root(base_url: str) -> str:
    """Normalize any OpenAI URL to its ``/v1`` root (drop a trailing adapter path)."""
    root = base_url.rstrip("/")
    for path in ("chat/completions", "completions"):
        if root.endswith("/" + path):
            root = root[: -(len(path) + 1)].rstrip("/")
    return root


def wait_for_models(
    base_url: str,
    api_key: Optional[str],
    timeout_s: float,
    interval_s: float = 3.0,
    *,
    provider: str = "endpoint",
) -> None:
    """Poll ``<base_url>/models`` until it returns HTTP 200 or ``timeout_s`` elapses."""
    url = api_root(base_url) + "/models"
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    started = time.monotonic()
    deadline = time.monotonic() + timeout_s
    last_err: Optional[str] = None
    attempts = 0
    while time.monotonic() < deadline:
        attempts += 1
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - trusted operator-supplied URL
                if 200 <= resp.status < 300:
                    logger.info("endpoint ready: %s", url)
                    readiness_terminal(provider, time.monotonic() - started, attempts, "ready")
                    return
                last_err = "HTTP %s" % resp.status
        except urllib.error.HTTPError as exc:
            last_err = "HTTP %s" % exc.code
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_err = str(exc)
        time.sleep(interval_s)
    readiness_terminal(provider, time.monotonic() - started, attempts, "timeout")
    raise TimeoutError(f"endpoint {url!r} not ready after {timeout_s:.0f}s (last: {last_err})")


class Provider:
    """Context manager that yields a :class:`ServedModel` and owns its teardown."""

    def __enter__(self) -> ServedModel:  # pragma: no cover - trivial
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb) -> bool:  # pragma: no cover - trivial
        return False


@dataclass
class EndpointProvider(Provider):
    """Attach to an already-running OpenAI-compatible endpoint (``/v1`` root)."""

    base_url: str
    model: str
    api_key: Optional[str] = None
    tokenizer: Optional[str] = None
    readiness_timeout_s: float = 120.0
    wait_ready: bool = True

    def __enter__(self) -> ServedModel:
        try:
            root = api_root(self.base_url)
            if self.wait_ready:
                wait_for_models(root, self.api_key, self.readiness_timeout_s)
            else:
                readiness_terminal("endpoint", 0.0, 0, "skipped")
            return ServedModel(
                base_url=root, model=self.model, api_key=self.api_key, tokenizer=self.tokenizer or self.model
            )
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc, tb) -> bool:
        cleanup_terminal("endpoint", 0.0, "succeeded")
        return False


@dataclass
class MarinServeProvider(Provider):
    """Serve ``model`` via the real ``marin-serve`` CLI on an Iris TPU slice."""

    model: str
    cluster: str = "marin"
    tpu: str = "v6e-8"
    name: Optional[str] = None
    region: Optional[str] = None
    api_key: Optional[str] = None
    tokenizer: Optional[str] = None
    wait_timeout_s: float = 1800.0
    timeout_hours: float = 2.0
    extra_args: Optional[List[str]] = None
    marin_serve_bin: str = "marin-serve"
    iris_bin: str = "iris"
    readiness_timeout_s: float = 300.0
    wait_ready: bool = True
    # cwd for the marin-serve subprocess. marin-serve bundles cwd as the Iris job
    # workspace and the container `uv sync`s it, so this MUST be a marin workspace
    # checkout (a dir whose pyproject resolves marin-core[tpu,vllm]). Running from
    # an unrelated dir bundles the wrong project and the job fails in its build
    # step (see marin-community/marin serving DX issue). Defaults to $MARIN_WORKSPACE
    # or the current directory.
    marin_workspace: Optional[str] = None

    _proc: Optional[subprocess.Popen] = None
    _job_id: Optional[str] = None
    _master_fd: Optional[int] = None

    @staticmethod
    def default_job_name(model: str) -> str:
        """Deterministic, dot-free job name (marin-serve rejects '.' in the endpoint)."""
        slug = re.sub(r"[^a-z0-9-]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")
        return "evalchemy-e2e-" + (slug or "model")

    def _command(self, job_name: str) -> List[str]:
        cmd = [
            self.marin_serve_bin,
            "iris",
            self.model,
            "--cluster",
            self.cluster,
            "--tpu",
            self.tpu,
            "--name",
            job_name,
            "--wait",
            "--wait-timeout",
            str(int(self.wait_timeout_s)),
            "--timeout-hours",
            str(self.timeout_hours),
        ]
        if self.region:
            cmd += ["--region", self.region]
        if self.extra_args:
            cmd += list(self.extra_args)
        return cmd

    def __enter__(self) -> ServedModel:
        try:
            return self._start()
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _start(self) -> ServedModel:
        # An explicit --name makes teardown deterministic. marin-serve derives a
        # random name otherwise, which we could not then stop reliably.
        job_name = self.name or self.default_job_name(self.model)
        self._job_id = None  # replaced by the exact printed job id once parsed
        cmd = self._command(job_name)
        cwd = self.marin_workspace or os.environ.get("MARIN_WORKSPACE") or None
        logger.info("starting (cwd=%s): %s", cwd or os.getcwd(), " ".join(shlex.quote(c) for c in cmd))
        # Run marin-serve on a PTY, not a pipe. marin-serve is a Python/click CLI
        # that prints the capability `base_url` we parse, then blocks holding the
        # tunnel (`while True: sleep`). Through a pipe, click's own wrapped stdout
        # block-buffers that final block and it never flushes -- `_read_until_ready`
        # then deadlocks (PYTHONUNBUFFERED does NOT help: click buffers in its own
        # stream layer). A PTY makes stdout look like a terminal, so click
        # line-buffers and flushes each line as it is written.
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._master_fd, slave_fd = pty.openpty()
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - operator-supplied command
                cmd,
                cwd=cwd,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,  # detach from our TTY/pgrp so Ctrl-C is ours to send
                env=env,
            )
        finally:
            os.close(slave_fd)  # the child holds the only writer end now
        base_url = self._read_until_ready(self.wait_timeout_s + 120.0)
        api_key = self.api_key or _CAPABILITY_API_KEY
        if self.wait_ready:
            wait_for_models(base_url, None, self.readiness_timeout_s, provider="marin-serve")
        else:
            readiness_terminal("marin-serve", 0.0, 0, "skipped")
        return ServedModel(
            base_url=api_root(base_url),
            model=self.model,
            api_key=api_key,
            tokenizer=self.tokenizer or self.model,
        )

    def _match_line(self, line: str) -> Optional[str]:
        """Inspect one marin-serve output line; return an endpoint URL when found.

        Records the job id as a side effect. Returns the capability ``base_url``
        or ``None`` for other lines.
        """
        job_match = _JOB_LINE.match(line)
        if job_match:
            self._job_id = job_match.group("job")
            # Hand the canonical id to later workflow steps: `iris job stop` rejects
            # bare names, so an external cleanup step needs this exact id.
            github_env = os.environ.get("GITHUB_ENV")
            if github_env:
                with open(github_env, "a", encoding="utf-8") as f:
                    f.write(f"IRIS_JOB_ID={self._job_id}\n")
        cap_match = _CAPABILITY_LINE.match(line)
        if cap_match:
            return cap_match.group("url")
        return None

    def _read_until_ready(self, timeout_s: float) -> str:
        """Stream marin-serve's PTY output until the endpoint URL; capture the job id.

        Returns the capability ``base_url``. Reads the PTY master fd and assembles
        lines by hand (a PTY has no readline); ANSI escapes are stripped before
        matching.
        """
        assert self._proc is not None and self._master_fd is not None
        deadline = time.monotonic() + timeout_s
        buf = ""
        while time.monotonic() < deadline:
            rlist, _, _ = select.select([self._master_fd], [], [], 1.0)
            if rlist:
                try:
                    chunk = os.read(self._master_fd, 4096)
                except OSError as exc:
                    # EIO on the master fd is the normal "child closed the PTY" EOF.
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if chunk:
                    buf += chunk.decode("utf-8", "replace")
                    while "\n" in buf:
                        raw, buf = buf.split("\n", 1)
                        line = _ANSI.sub("", raw).rstrip("\r")
                        logger.info("[marin-serve] %s", line.rstrip())
                        url = self._match_line(line)
                        if url is not None:
                            return url
                    continue
            # No data this tick (or EOF): only now is an exit terminal.
            if self._proc.poll() is not None:
                raise RuntimeError(f"marin-serve exited early (code {self._proc.returncode}) before serving")
        raise TimeoutError(f"marin-serve did not print an endpoint URL within {timeout_s:.0f}s")

    def __exit__(self, exc_type, exc, tb) -> bool:
        started = time.monotonic()
        try:
            outcome = self._cleanup()
        except Exception as error:
            outcome = "failed"
            record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "cleanup"})
            logger.warning("marin-serve cleanup failed (%s); check for a live Iris job", error)
        cleanup_terminal("marin-serve", time.monotonic() - started, outcome)
        return False

    def _cleanup(self) -> str:
        # Detach the tunnel-holding CLI first, then stop the Iris job it left
        # running. Both are best-effort so teardown never masks a run failure.
        outcome = "succeeded"
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired as error:
                    outcome = "failed"
                    record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "wait"})
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired as kill_error:
                        record_failure(
                            "cleanup",
                            kill_error,
                            attributes={"provider": "marin-serve", "operation": "kill"},
                        )
            except (OSError, subprocess.SubprocessError) as error:
                outcome = "failed"
                record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "stop_cli"})
                logger.warning("marin-serve CLI cleanup failed (%s)", error)
        if self._master_fd is not None:
            try:
                os.close(self._master_fd)
            except OSError as error:
                outcome = "failed"
                record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "close_pty"})
            self._master_fd = None
        # `iris job stop` accepts only canonical /<user>/<job> ids; a bare name makes
        # it raise, which check=False would swallow -- resolve one before stopping.
        job_id = self._job_id or self._resolve_job_id()
        if job_id is None:
            name = self.name or self.default_job_name(self.model)
            error = RuntimeError(f"no canonical Iris job ID found for {name}")
            record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "find_job"})
            logger.warning(
                "no canonical iris job id (marin-serve never printed one and `job list` shows no '%s'); "
                "if a slice is still up, stop it manually: %s --cluster %s job stop /<user>/%s",
                name,
                self.iris_bin,
                self.cluster,
                name,
            )
            return "job_not_found" if outcome == "succeeded" else outcome
        # NB: --cluster is a TOP-LEVEL iris flag (`iris --cluster X job stop <id>`),
        # not a `job stop` option.
        stop = [self.iris_bin, "--cluster", self.cluster, "job", "stop", job_id]
        logger.info("stopping iris job: %s", " ".join(shlex.quote(c) for c in stop))
        try:
            result = subprocess.run(stop, timeout=120, check=False)  # noqa: S603 - operator-supplied command
            if result.returncode != 0:
                outcome = "failed"
                error = RuntimeError(f"iris job stop exited with code {result.returncode}")
                record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "stop_job"})
                logger.warning("%s; stop it manually: %s", error, " ".join(stop))
        except (OSError, subprocess.SubprocessError) as error:
            outcome = "failed"
            record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "stop_job"})
            logger.warning("iris job stop failed (%s); stop it manually: %s", error, " ".join(stop))
        return outcome

    def _resolve_job_id(self) -> Optional[str]:
        """Find this run's canonical /<user>/<job> id via `iris job list`.

        Covers the case where marin-serve died before printing its job line --
        exactly when teardown matters most.
        """
        name = self.name or self.default_job_name(self.model)
        cmd = [self.iris_bin, "--cluster", self.cluster, "job", "list"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)  # noqa: S603
        except (OSError, subprocess.SubprocessError) as error:
            record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "list_jobs"})
            logger.warning("iris job list failed (%s)", error)
            return None
        if result.returncode != 0:
            error = RuntimeError(f"iris job list exited with code {result.returncode}")
            record_failure("cleanup", error, attributes={"provider": "marin-serve", "operation": "list_jobs"})
            logger.warning("%s", error)
            return None
        match = re.search(rf"(/\S+/{re.escape(name)})\b", result.stdout)
        return match.group(1) if match else None


def build_provider(
    provider: str,
    model: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tokenizer: Optional[str] = None,
    cluster: str = "marin",
    tpu: str = "v6e-8",
    name: Optional[str] = None,
    region: Optional[str] = None,
    marin_workspace: Optional[str] = None,
    wait_timeout_s: float = 1800.0,
    timeout_hours: float = 2.0,
    readiness_timeout_s: float = 300.0,
    wait_ready: bool = True,
    extra_args: Optional[List[str]] = None,
) -> Provider:
    """Factory: ``endpoint`` or ``marin-serve``."""
    if provider == "endpoint":
        if not base_url:
            raise ValueError("--base-url is required for provider 'endpoint'")
        return EndpointProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            tokenizer=tokenizer,
            readiness_timeout_s=readiness_timeout_s,
            wait_ready=wait_ready,
        )
    if provider == "marin-serve":
        return MarinServeProvider(
            model=model,
            cluster=cluster,
            tpu=tpu,
            name=name,
            region=region,
            marin_workspace=marin_workspace,
            api_key=api_key,
            tokenizer=tokenizer,
            wait_timeout_s=wait_timeout_s,
            timeout_hours=timeout_hours,
            readiness_timeout_s=readiness_timeout_s,
            wait_ready=wait_ready,
            extra_args=extra_args,
        )
    raise ValueError(f"unknown provider {provider!r}; expected 'endpoint' or 'marin-serve'")
