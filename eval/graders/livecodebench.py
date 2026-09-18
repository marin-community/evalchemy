"""Self-contained, bounded grader for the LiveCodeBench (LCB) test format.

One grader replaces the four vendored LCB execution stacks (``LiveCodeBench``,
``LiveCodeBenchv5``, ``LiveCodeBenchv5_official`` -- the NovaSky ``testing_util``
Manager-per-candidate pattern -- and ``LiveBench/lcb_runner``): all of them left the
worker unbounded in memory and time. ``reliability_guard(maximum_memory_bytes=...)``
accepted a cap that no call site ever passed, and the ``join`` deadline scaled with the
test count, so a runaway candidate ran on with no cap while the parent waited as long as
the tests said it might. The same unbounded pattern OOM-killed a 64-GPU MarinSkyRL run
(issue #147, incident echo.oa.dev/wiki/471).

The shape follows ``eval.graders.humaneval`` -- a single child process, pipe result
transport, join-grace kill, restorable guard -- with the bounds the vendored stacks lack:

- a memory cap wired by default (``RLIMIT_AS``/``RLIMIT_DATA``/``RLIMIT_STACK``), applied
  best-effort (some platforms, e.g. Darwin, refuse to lower it, and the grader must not
  crash on the guard there);
- a wall-clock deadline *independent of the test count*, in addition to the per-test
  ``SIGALRM``;
- deterministic child reaping, with **no** ``multiprocessing.Manager`` -- a pipe carries
  the per-test outcomes, so a timeout leaves no Manager server to reap;
- child identification at spawn (pid, test count, deadline).

It is library-consumable, not only route-internal: MarinSkyRL's reward path calls
:func:`run` synchronously per trajectory and derives binary and fractional rewards from
the per-test outcomes. Inputs are the normalized LCB test list -- call-based or
stdin, with an optional ``fn_name`` -- plus the extracted program. Both
stop-on-failure short-circuiting and collect-all are preserved. Concurrent callers are
supported: every run is its own process and the parent never waits unbounded.
"""

import contextlib
import faulthandler
import io
import logging
import multiprocessing
import os
import signal
import tempfile
import threading
import time
import types

from eval.contracts.grading import GraderExecutionMode

logger = logging.getLogger(__name__)

# A candidate that outruns this is failed deterministically. 1 GiB fits a real LCB
# solution and its test harness while bounding the runaway-allocator class from #147.
DEFAULT_MAX_MEMORY_BYTES = 1 * 1024 * 1024 * 1024

# Per-test alarm; a wall-clock deadline overrides it. The deadline is NOT
# (test_timeout + 1) * len(tests) + 5 (the old join) -- it bounds the whole run.
TIMEOUT = 6.0
DEFAULT_DEADLINE = 30.0

# The parent gives a killed child this much to die and be reaped before giving up; a
# SIGKILL'd process reaps immediately, so this only matters for the (rare) failed kill.
_JOIN_GRACE = 1.0

# A per-test detail string is candidate-controlled and the pipe holds ~64 KiB before
# send() would block; cap it so a verbose failure cannot wedge the child.
_MAX_DETAIL_CHARS = 8192

PASSED = "passed"
FAILED = "failed"
TIMED_OUT = "timeout"


class TimeoutException(Exception):
    pass


class redirect_stdin(contextlib._RedirectStream):  # noqa: N801
    """``contextlib`` ships ``redirect_stdout``/``redirect_stderr`` only; this is the stdin twin (mirrors ``humaneval``)."""

    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds):
    """Bound one test with a SIGALRM (main thread of the child process only)."""

    def _handler(signum, frame):
        raise TimeoutException()

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _reliability_guard(maximum_memory_bytes):
    """Disable destructive calls and, where the platform allows, cap memory. Returns a restore callable.

    The rlimit is applied best-effort: Darwin and some sandboxes refuse to lower
    ``RLIMIT_AS``, and the guard must degrade there rather than crash the child -- the wall-clock
    deadline plus kill is the bound that holds everywhere.
    """
    if maximum_memory_bytes:
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
        except (ValueError, OSError):
            pass  # platform refused to lower the cap; the deadline still bounds the run

    faulthandler.disable()
    import builtins

    builtins.exit = None
    builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"

    import shutil
    import subprocess
    import sys

    saved = []
    _DISABLED = {
        os: ("kill", "system", "putenv", "remove", "removedirs", "rmdir", "fchdir", "setuid", "fork",
             "forkpty", "killpg", "rename", "renames", "truncate", "replace", "unlink", "chdir",
             "getcwd", "chroot"),
        shutil: ("rmtree", "move", "chown"),
        subprocess: ("Popen",),
    }
    for module, names in _DISABLED.items():
        for name in names:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, None)
    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None

    def restore():
        for module, name, original in saved:
            setattr(module, name, original)

    return restore


def _resolve_call_target(ns, fn_name):
    """Find the candidate object to call, by name or by inferring the first defined function."""
    if fn_name:
        return ns[fn_name]
    for name, value in ns.items():
        if not name.startswith("_") and isinstance(value, (types.FunctionType, types.MethodType)):
            return value
    target = next((v for n, v in ns.items() if isinstance(v, type)), None)
    if target is not None:
        return target()
    raise RuntimeError("no call target inferred from candidate")


def _score(value, expected):
    """Compare a candidate output to its expected value, tolerating int/str/float and list/tuple."""
    if value == expected:
        return True
    try:
        if isinstance(expected, (list, tuple)) and isinstance(value, (list, tuple)):
            if [int(x) for x in expected] == [int(y) for y in value]:
                return True
        if float(value) == float(expected):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _run_stdin(completion, test_input):
    """Run a stdin candidate and return its captured stdout."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(io.StringIO()), redirect_stdin(io.StringIO(test_input)):
        exec(completion, {})  # noqa: S102  # the graded candidate, run per test
    return captured.getvalue().strip()


def _run_call(target, test_input):
    """Call a call-based candidate with its arguments and return the result."""
    if isinstance(test_input, (list, tuple)):
        return target(*test_input)
    if isinstance(test_input, dict):
        return target(**test_input)
    return target(test_input)


def _run_tests(tests, completion, fn_name, stdin_mode, test_timeout, collect_all, connection, cap):
    """Child side: run the candidate's tests one bounded test at a time over ``connection``."""
    scratch = tempfile.mkdtemp(prefix="lcb-grader-")
    os.chdir(scratch)
    restore = _reliability_guard(cap)
    try:
        target = None
        if not stdin_mode:
            ns = {}
            exec(completion, ns)  # noqa: S102  # define the candidate's functions once
            target = _resolve_call_target(ns, fn_name)
        for index, test in enumerate(tests):
            input_value = test.get("input")
            expected = test.get("output")
            status = FAILED
            detail = ""
            with _time_limit(test_timeout):
                try:
                    if stdin_mode:
                        output = _run_stdin(completion, input_value)
                        status = PASSED if _score(output, expected) else FAILED
                    else:
                        output = _run_call(target, input_value)
                        status = PASSED if _score(output, expected) else FAILED
                except TimeoutException:
                    status = TIMED_OUT
                    detail = "per-test alarm expired"
                except BaseException as exc:  # noqa: BLE001  # any candidate failure is a failed test
                    status = FAILED
                    detail = f"{type(exc).__name__}: {exc}"
            connection.send(("test", index, status, detail[:_MAX_DETAIL_CHARS]))
            if status != PASSED and not collect_all:
                break
    finally:
        restore()
        connection.send(("done",))  # the parent treats a missing/failed done as a timeout
        connection.close()


def _terminate(pid):
    if pid is not None:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass  # already gone


def run(
    tests,
    completion,
    fn_name=None,
    stdin_mode=False,
    test_timeout=TIMEOUT,
    deadline=DEFAULT_DEADLINE,
    collect_all=False,
    maximum_memory_bytes=DEFAULT_MAX_MEMORY_BYTES,
) -> dict:
    """Run a candidate program against a normalized LCB test list, bounded.

    Args:
        tests: The normalized LCB tests, ``[{"input": ..., "output": ...}, ...]``.
        completion: The extracted program (the candidate's functions or, for stdin, its script).
        fn_name: The function to call; inferred from the candidate when omitted.
        stdin_mode: Feed ``input`` to stdin and compare stdout rather than calling a function.
        test_timeout: Seconds per test before a single test is scored ``timeout``.
        deadline: Wall-clock bound for the whole run, independent of the test count.
        collect_all: On a failure, run the remaining tests; otherwise stop at the first.
        maximum_memory_bytes: The memory cap applied to the child; best-effort per platform.

    Returns:
        A dict with ``passed``, ``num_passed``, ``num_tests``, ``timed_out``,
        ``child_pid``, and per-test ``outcomes`` (``{"index", "outcome", "detail"}``) for a
        binary/fractional reward; ``timed_out`` is True when any test hit the alarm or the run
        was cut off by the deadline.
    """
    tests = list(tests)
    if deadline is None:
        deadline = DEFAULT_DEADLINE

    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_run_tests,
        args=(
            tests,
            completion,
            fn_name,
            stdin_mode,
            test_timeout,
            collect_all,
            sender,
            maximum_memory_bytes,
        ),
    )
    process.start()
    sender.close()
    pid = process.pid
    logger.debug(
        "lcb grader spawned pid=%s tests=%d test_timeout=%.2fs deadline=%.2fs cap=%s",
        pid,
        len(tests),
        test_timeout,
        deadline,
        maximum_memory_bytes,
    )

    watchdog = threading.Timer(deadline, _terminate, args=(pid,))
    watchdog.daemon = True
    watchdog.start()

    process.join(timeout=deadline + _JOIN_GRACE)
    watchdog.cancel()
    if process.is_alive():
        _terminate(pid)
        process.join(timeout=_JOIN_GRACE)

    outcomes: list[dict] = []
    done = False
    deadline_end = time.monotonic() + _JOIN_GRACE
    while time.monotonic() < deadline_end:
        if receiver.poll(0.02):
            try:
                message = receiver.recv()
            except (EOFError, OSError, ConnectionResetError):
                break  # child died mid-write: partial frame; bounded, no hang
            if message[0] == "test":
                _, index, status, detail = message
                outcomes.append({"index": index, "outcome": status, "detail": detail})
            elif message[0] == "done":
                done = True
                break
        elif process.poll():
            break
    receiver.close()

    timed_out = (not done) or any(o["outcome"] == TIMED_OUT for o in outcomes)
    num_passed = sum(1 for o in outcomes if o["outcome"] == PASSED)
    all_passed = (
        done
        and len(outcomes) == len(tests)
        and {o["index"] for o in outcomes} == set(range(len(tests)))
        and all(o["outcome"] == PASSED for o in outcomes)
    )

    return {
        "passed": all_passed,
        "num_passed": num_passed,
        "num_tests": len(tests),
        "timed_out": timed_out,
        "child_pid": pid,
        "outcomes": outcomes,
    }


def grade(tests, completion, **kwargs) -> float:
    """Return the fraction of tests the candidate passed, for a binary/fractional reward."""
    result = run(tests, completion, **kwargs)
    if result["num_tests"] == 0:
        return 0.0
    return result["num_passed"] / result["num_tests"]


# The grader declares its own isolation so a driver that dispatches it (``eval.contracts.grading``)
# runs it outside the threaded pool, as ``execute_grading_jobs(PROCESS_ISOLATED)`` does for the
# Plus-style graders. The RL reward path that calls ``run`` directly is already process-isolated.
GRADER_EXECUTION_MODE = GraderExecutionMode.PROCESS_ISOLATED
