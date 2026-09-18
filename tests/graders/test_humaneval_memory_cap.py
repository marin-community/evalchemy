# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the memory cap in the HumanEval grader.

Issue #147: ``eval.graders.humaneval`` has the same latent gap as the vendored LiveCodeBench
stacks -- ``reliability_guard`` accepts ``maximum_memory_bytes`` but ``check_correctness``
never passes it, so the cap is never armed. Before the fix the cap default was ``None`` and
a runaway candidate ran unbounded; these tests assert the cap is wired by default and that a
candidate exhausting it is resolved deterministically (no hang).

The *enforcement* of the cap (failing the candidate) is asserted on Linux, where ``RLIMIT_AS``
is honored. macOS refuses to lower it / overcommits, so there the cap is best-effort (the
grader must not crash) and only the "wired" and "returns without hanging" guarantees hold.
"""

import pathlib
import platform
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import humaneval  # noqa: E402

LINUX = platform.system() == "Linux"


def test_check_correctness_wires_the_memory_cap_by_default():
    """A runaway allocation must be bounded by default, not left unbounded as before #147."""
    defaults = humaneval.check_correctness.__defaults__
    assert defaults is not None and all(v is not None for v in defaults), (
        "check_correctness must wire a memory cap by default"
    )


def test_memory_cap_default_is_reasonable():
    assert humaneval.DEFAULT_MEMORY_BYTES is not None
    assert 100 * 1024**2 <= humaneval.DEFAULT_MEMORY_BYTES <= 4 * 1024**3


def test_a_candidate_that_exhausts_memory_returns_without_hanging():
    """A 1 GiB allocation is resolved quickly (MemoryError, kill, or pass) -- never a hang."""
    start = time.perf_counter()
    humaneval.check_correctness("x = bytearray(1 * 1024 * 1024 * 1024)\n", timeout=10.0)
    assert time.perf_counter() - start < 15.0, "deterministic resolution, not a hang"


@pytest.mark.skipif(not LINUX, reason="macOS refuses to lower RLIMIT_AS / overcommits, so the cap is best-effort there")
def test_cap_is_enforced_on_linux():
    """On Linux the RLIMIT_AS actually trips: a 1 GiB alloc under a 1 GiB cap is a failure."""
    outcome = humaneval.check_correctness("x = bytearray(1 * 1024 * 1024 * 1024)\n", timeout=10.0)
    assert outcome != humaneval.PASSED

