# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage that the ``LiveBench/lcb_runner`` stack is bounded.

Issue #147: ``compute_code_generation_metrics.check_correctness`` wrapped ``run_test`` in a
``multiprocessing.Manager`` plus a ``Process`` and joined on
``(timeout + 1) * len(inputs) + 5`` -- unbounded in both memory and time. The rewrite routes
through the same bounded wrapper (pipe result transport + parent-side wall-clock watchdog +
deterministic reap) so no Manager server or worker survives a timeout.

These tests need the ``livebench`` extra (``pyext``); ``pytest.importorskip`` keeps them
runnable in a full env and skipped (not failing) in the lean ``graders`` CI job, which runs
``tests/graders/`` with ``--extra dev`` only. Same opt-in shape as the HumanEval reference tests.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

pytest.importorskip("pyext", reason="LiveBench code-exec grading needs the `livebench` extra")


def _compute_module():
    import importlib

    sys.path.insert(0, str(REPO / "eval" / "chat_benchmarks" / "LiveBench"))
    return importlib.import_module("livebench.lcb_runner.evaluation.compute_code_generation_metrics")


def _running_children() -> set[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "command="], capture_output=True, text=True).stdout
    me = str(os.getpid())
    running = set()
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1] == me and "resource_tracker" not in parts[2] and parts[2].split()[0].rsplit("/", 1)[-1] != "ps":
            running.add(int(parts[0]))
    return running


def test_livebench_check_correctness_is_bounded():
    """A runaway through LiveBench's routed ``check_correctness`` is cut to the deadline and reaps."""
    module = _compute_module()
    sample = {"input_output": json.dumps({"inputs": ["[1]"], "outputs": ["[0]"], "fn_name": "hang"})}
    runaway = "import time\ndef hang(a):\n    while True:\n        time.sleep(1)\n"

    start = time.perf_counter()
    before = _running_children()
    module.check_correctness(sample, runaway, timeout=1, debug=False, deadline=5)
    elapsed = time.perf_counter() - start

    assert elapsed < 8.0, f"runaway should be cut at the 5s deadline, took {elapsed:.1f}s"
    time.sleep(0.3)
    assert (_running_children() - before) == set(), "no LiveBench worker or manager may survive a timeout"


def test_livebench_check_correctness_no_longer_uses_manager():
    """The unbounded Manager-per-candidate pattern is gone; a pipe carries the result now."""
    import inspect

    module = _compute_module()
    source = inspect.getsource(module.check_correctness)
    assert "multiprocessing.Manager()" not in source
    assert "Manager" not in inspect.getsource(module._worker_run)
    # and the source keeps the bounded pipe transport
    assert "multiprocessing.Pipe" in source
