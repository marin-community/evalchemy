"""Prepare isolated shell functions; protected Bash checks retain the assertions."""

import base64
import hashlib
import re
import shlex
import socketserver
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from harbor_config.errors import ErrorCategory
from lm_eval.verifyit_function_worker import MAX_BYTES, read, write
from verifyit.execution.command import run_command
from verifyit.file_ops.read import read_text
from verifyit.grade import Status, finalize_preparation_failure
from verifyit.preparation.errors import InvalidPreparation, PreparationError, PreparationFailure


class ShellPolicy(StrEnum):
    FIRST_FUNCTION = "isolated_first_shell_function_v2"


@dataclass(frozen=True)
class ShellInputs:
    prompt: str
    test: str
    generation: str | None


@dataclass(frozen=True)
class PreparedShell:
    raw: ShellInputs
    reference: str
    prediction: str
    provenance: dict


def preparation_error(status, category, error_type, message, stage):
    failure = PreparationFailure(status, category, error_type, message, stage)
    verdict = finalize_preparation_failure(**asdict(failure))
    return (InvalidPreparation if status is Status.INVALID_TASK else PreparationError)(failure, verdict)


def structure_shell(problem, generation):
    prompt, test = problem.get("prompt"), problem.get("test")
    if not isinstance(prompt, str) or not isinstance(test, str):
        raise preparation_error(Status.INVALID_TASK, ErrorCategory.UNKNOWN, "InvalidTask", "Shell references require strings", "structure")
    if generation is not None and not isinstance(generation, str):
        raise preparation_error(Status.SCORED, ErrorCategory.AGENT, "InvalidCandidate", "Shell completion must be a string or null", "structure")
    return ShellInputs(prompt, test, generation)


def prepare_shell(problem, generation, *, timeout, max_bytes=MAX_BYTES, policy=ShellPolicy.FIRST_FUNCTION):
    raw = structure_shell(problem, generation)
    try:
        policy = ShellPolicy(policy)
    except ValueError as error:
        raise preparation_error(Status.INVALID_TASK, ErrorCategory.UNKNOWN, "InvalidTask", str(error), "policy") from error
    match = re.search(r"([A-Za-z_][A-Za-z_0-9]*)\(\)\s*\{\s*$", raw.prompt)
    if match is None or not raw.test.startswith("}\n"):
        raise preparation_error(Status.INVALID_TASK, ErrorCategory.UNKNOWN, "InvalidTask", "Shell reference must close the source function before tests", "policy")
    entry, test = match.group(1), raw.test[2:]
    # Parse trusted assertions before any candidate-dependent execution.
    try:
        syntax = run_command(["bash", "-n"], Path.cwd(), timeout, stdin_text=entry + "() { :; }\n" + test)
    except OSError as error:
        raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, type(error).__name__, str(error), "reference_parse") from error
    if syntax.timed_out:
        raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, "TimeoutError", "Trusted shell parse timed out", "reference_parse")
    if syntax.returncode:
        raise preparation_error(Status.INVALID_TASK, ErrorCategory.UNKNOWN, "InvalidTask", syntax.stderr, "reference_parse")
    provenance = {
        "policy": policy.value,
        "null_candidate": "empty_source",
        "source_selection": "first_entry_function; newline suffix ignored; same-line suffix rejected by Bash import",
        "initialization": "none; only function invocation executes candidate code",
        "max_rpc_bytes": max_bytes,
        "prompt_sha256": hashlib.sha256(raw.prompt.encode()).hexdigest(),
        "test_sha256": hashlib.sha256(raw.test.encode()).hexdigest(),
        "candidate_sha256": None if raw.generation is None else hashlib.sha256(raw.generation.encode()).hexdigest(),
    }
    reference = (
        "from eval.graders.verifyit_shell import run_checks, preparation_error, Status, ErrorCategory\n"
        "def check(candidate):\n"
        "    try:\n"
        f"        result = run_checks(candidate, {entry!r}, {test!r}, max_bytes={max_bytes})\n"
        "    except OSError as error:\n"
        "        raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, type(error).__name__, str(error), 'shell_protocol') from error\n"
        "    assert result.returncode == 0\n"
        "check(shell_candidate)\n"
    )
    code = (raw.generation or "") + "\n}\n"
    definition = re.match(r"\A(?:[ \t]*(?:#[^\n]*)?\n)*[ \t]*" + re.escape(entry) + r"\s*\(\s*\)", code)
    function_source = None if definition is None else "()" + code[definition.end():]
    import_check = ("command builtin" if entry == "builtin" else "builtin") + ' declare -F -- "$1"'
    prediction = (
        "import base64\nimport os\nimport subprocess\nimport tempfile\nfrom pathlib import Path\n"
        "def shell_candidate(args, merge_streams=False):\n"
        "    environment = {key: value for key, value in os.environ.items() if key not in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS') and not key.startswith('BASH_FUNC_')}\n"
        "    with tempfile.TemporaryDirectory() as directory:\n"
        "        path = Path(directory) / 'candidate.sh'\n"
        f"        path.write_text({code!r})\n"
        "        syntax = subprocess.run(['bash', '--noprofile', '--norc', '-n', str(path)], env=environment, capture_output=True)\n"
        "        if syntax.returncode:\n"
        "            return {'candidate_failure': 'ShellSyntaxError'}\n"
        f"        definition = {function_source!r}\n"
        "        if definition is None:\n"
        "            return {'candidate_failure': 'ShellFunctionDefinitionError'}\n"
        f"        environment[{'BASH_FUNC_' + entry + '%%'!r}] = definition\n"
        f"        imported = subprocess.run(['bash', '--noprofile', '--norc', '-c', {import_check!r}, '--', {entry!r}], env=environment, capture_output=True)\n"
        "        if imported.returncode:\n"
        "            return {'candidate_failure': 'ShellFunctionImportError'}\n"
        f"        result = subprocess.run(['bash', '--noprofile', '--norc', '-c', '\"$@\"', '--', {entry!r}, *args], env=environment, "
        "stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_streams else subprocess.PIPE)\n"
        "    return {'stdout': base64.b64encode(result.stdout).decode('ascii'), "
        "'stderr': base64.b64encode(result.stderr or b'').decode('ascii'), 'returncode': result.returncode}\n"
    )
    return PreparedShell(raw, reference, prediction, provenance)


def run_checks(candidate, entry, test, *, max_bytes=MAX_BYTES):
    """Transport observations and terminal failures; protected Pytest grades checks."""
    proxy = Path(__file__).with_name("verifyit_shell_proxy.py")
    with tempfile.TemporaryDirectory(prefix="vs-", dir="/tmp") as directory:
        path = Path(directory)
        address = str(path / "rpc.sock")
        calls, completed = path / "calls", path / "completed"
        calls.touch()
        completed.touch()
        failures = []

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                stage = "shell_protocol"
                try:
                    request = read(self.rfile, max_bytes)
                    if set(request) != {"args", "merge_streams"} or type(request["merge_streams"]) is not bool or not isinstance(request["args"], list) or any(not isinstance(arg, str) for arg in request["args"]):
                        raise ValueError("Malformed shell request")
                    stage = "candidate_call"
                    result = candidate(request["args"], merge_streams=request["merge_streams"])
                    stage = "shell_observation"
                    if "candidate_failure" in result:
                        raise preparation_error(Status.SCORED, ErrorCategory.AGENT, str(result["candidate_failure"]), "Candidate shell failed before invocation", "candidate_prepare")
                    if type(result["returncode"]) is not int or not -127 <= result["returncode"] <= 255:
                        raise ValueError("Invalid shell return status")
                    for channel in ("stdout", "stderr"):
                        base64.b64decode(result[channel], validate=True)
                    write(self.wfile, result, max_bytes)
                except PreparationError as error:
                    failures.append(error)
                except Exception as error:
                    status = Status.SCORED if stage == "candidate_call" else Status.INFRA_ERROR
                    category = ErrorCategory.AGENT if stage == "candidate_call" else ErrorCategory.UNKNOWN
                    failures.append(preparation_error(status, category, type(error).__name__, str(error), stage))

        with socketserver.UnixStreamServer(address, Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                invocation = shlex.join([sys.executable, str(proxy), address, str(completed), str(max_bytes)])
                script = path / "tests.sh"
                script.write_text(entry + "() { printf . >> " + shlex.quote(str(calls)) + "; " + invocation + ' "$@"; }\n' + test)
                result = subprocess.run(["bash", str(script)], capture_output=True)
            finally:
                server.shutdown()
                thread.join(timeout=2)
            if failures:
                raise failures[0]
            if read_text(calls) != read_text(completed):
                raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, "ShellProtocolError", "Shell proxy did not complete every requested call", "shell_protocol")
            return result
