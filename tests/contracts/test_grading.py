"""Shared grader resources preserve isolation and propagate failures."""

import asyncio
import os

import pytest

from eval.runtime.resources import Resources


def worker_pid():
    return os.getpid()


@pytest.mark.parametrize("mode", ["serial", "thread_safe", "sandboxed", "process_isolated"])
def test_grader_runs_under_its_declared_isolation(mode):
    resources = Resources(workers=2)
    try:
        pid = asyncio.run(resources.call(worker_pid, (), mode))
    finally:
        resources.close()
    assert (pid != os.getpid()) == (mode == "process_isolated")


def test_grader_failure_propagates_without_becoming_a_score():
    def broken():
        raise ValueError("invalid reference")
    resources = Resources()
    try:
        with pytest.raises(ValueError, match="invalid reference"):
            asyncio.run(resources.call(broken, (), "thread_safe"))
    finally:
        resources.close()
