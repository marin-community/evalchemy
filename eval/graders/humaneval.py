"""Self-contained grader for the lm-eval-harness ``humaneval`` task.

Ports the candidate construction and ``pass@1`` scoring of
``lm_eval/tasks/humaneval/utils.py`` at the pinned lm-evaluation-harness
``v0.4.12`` so grading needs only the problem text, the model's solution, and
the reference answer. The reference routes through
``evaluate.load("code_eval")``, which downloads a metric module on first use;
this module needs no hub access.

Execution semantics are reproduced exactly, including the sandbox: the candidate
runs in a fresh child process, under ``reliability_guard`` inside a temporary
working directory, with stdio swallowed and a SIGALRM timeout. What is dropped
is pure overhead -- upstream's ``check_correctness`` starts a
``multiprocessing.Manager`` server *in addition to* the worker process, solely
to carry one result string back, so every candidate costs two process spawns
instead of one. A pipe carries the same string.

The per-candidate process is not an optimization target. ``reliability_guard``
mutates its process irreversibly, so a worker cannot be reused, and running
model-generated code in-process would be faster only by removing the isolation
that makes grading safe at all.

The sandbox helpers below are ported from HuggingFace ``evaluate``'s
``code_eval`` metric (Apache-2.0), itself derived from OpenAI's HumanEval
release. As upstream warns, ``reliability_guard`` is a guard against accidental
damage, not a security sandbox; untrusted code still deserves a real one.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile

TIMEOUT = 3.0
# Upstream waits one second beyond the in-process alarm before killing the
# child, so a wedged candidate is reported as a timeout rather than a hang.
_JOIN_GRACE = 1.0

PASSED = "passed"
TIMED_OUT = "timed out"


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        return False


class redirect_stdin(contextlib._RedirectStream):  # noqa: N801
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), redirect_stdin(stream):
        yield


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname, chdir(dirname):
        yield dirname


_DISABLED_OS_CALLS = (
    "kill",
    "system",
    "putenv",
    "remove",
    "removedirs",
    "rmdir",
    "fchdir",
    "setuid",
    "fork",
    "forkpty",
    "killpg",
    "rename",
    "renames",
    "truncate",
    "replace",
    "unlink",
    "fchmod",
    "fchown",
    "chmod",
    "chown",
    "chroot",
    "lchflags",
    "lchmod",
    "lchown",
    "getcwd",
    "chdir",
)
_DISABLED_SHUTIL_CALLS = ("rmtree", "move", "chown")


def reliability_guard(maximum_memory_bytes=None):
    """Disable destructive calls so a candidate cannot damage the host.

    Returns a callable that restores the patched `os`/`shutil`/`subprocess` symbols,
    which the caller invokes only after the candidate has finished -- restoring
    earlier would hand the candidate the calls this is meant to withhold.

    Not a security sandbox: it blocks accidents and casual misbehavior, not a
    determined escape. Run untrusted code in a real sandbox as well.
    """
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"

    import shutil
    import subprocess

    saved = []
    for module, names in ((os, _DISABLED_OS_CALLS), (shutil, _DISABLED_SHUTIL_CALLS), (subprocess, ("Popen",))):
        for name in names:
            if hasattr(module, name):
                saved.append((module, name, getattr(module, name)))
                setattr(module, name, None)

    builtins.help = None

    import sys

    for name in ("ipdb", "joblib", "resource", "psutil", "tkinter"):
        sys.modules[name] = None

    def restore():
        for module, name, original in saved:
            setattr(module, name, original)

    return restore


def _run_candidate(check_program, connection, timeout):
    """Execute one candidate in this (throwaway) process and report the outcome."""
    with create_tempdir():
        restore = reliability_guard()

        try:
            exec_globals = {}
            with swallow_io(), time_limit(timeout):
                exec(check_program, exec_globals)  # noqa: S102  # the graded candidate
            outcome = PASSED
        except TimeoutException:
            outcome = TIMED_OUT
        except BaseException as exc:  # noqa: BLE001  # any failure is a failed candidate
            outcome = f"failed: {exc}"

        restore()
        # Reported from inside the sandbox, where upstream appends to its manager
        # list. A candidate that leaves a file behind can make tempdir cleanup
        # raise, and the verdict has to be on the wire before that can happen.
        connection.send(outcome)
        connection.close()


def check_correctness(check_program: str, timeout: float = TIMEOUT) -> str:
    """Run one candidate program in an isolated process and return its outcome."""
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_run_candidate, args=(check_program, sender, timeout))
    process.start()
    sender.close()
    process.join(timeout=timeout + _JOIN_GRACE)
    if process.is_alive():
        process.kill()
        process.join()

    outcome = TIMED_OUT
    if receiver.poll():
        try:
            outcome = receiver.recv()
        except EOFError:
            outcome = TIMED_OUT
    receiver.close()
    return outcome


def build_candidate(problem: str, solution: str) -> str:
    """Join the prompt and the completion, as the task's ``create_test`` filter does."""
    return problem + solution


def grade(problem: str, solution: str, reference_answer: str, timeout: float = TIMEOUT) -> float:
    """Grade one completion by running it against the reference tests.

    Args:
        problem: The HumanEval prompt, which the completion continues.
        solution: The model's completion.
        reference_answer: The task's target, ``"{test}\\ncheck({entry_point})"``.
        timeout: Seconds the candidate may run before it is scored a timeout.

    Returns:
        ``1.0`` when the candidate passes the tests, else ``0.0``. This is
        ``pass@1`` for the single sample the task generates.
    """
    check_program = build_candidate(problem, solution) + "\n" + reference_answer
    return 1.0 if check_correctness(check_program, timeout) == PASSED else 0.0
