"""Trusted bash assertions must observe the original candidate transport data."""

import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from eval.graders.verifyit_shell import prepare_shell, run_checks
from eval.graders.verifyit_shell_evaluation import evaluate_functional_correctness, load_records
from eval.graders import verifyit_shell
from lm_eval import verifyit_humaneval
from verifyit.grade import InvalidTask, Status
from verifyit.preparation.errors import InvalidPreparation, PreparationError


@pytest.mark.parametrize(
    "body,checks",
    [
        ("return 7", "candidate; code=$?; test \"$code\" -eq 7"),
        ("printf 'answer\\n\\n'", "test \"$(candidate)\" = answer"),
        ("printf '\\377'", "test \"$(candidate | od -An -tu1 | tr -d ' ')\" = 255"),
        ("printf diagnostic >&2", "test \"$(candidate 2>&1)\" = diagnostic"),
        ("printf A; printf B >&2", "test \"$(candidate 2>&1)\" = AB"),
        ("printf A >&2; printf B", "test \"$(candidate 2>&1)\" = AB"),
    ],
)
def test_trusted_checks_preserve_shell_observations(body, checks, tmp_path):
    source = tmp_path / "source.sh"
    source.write_text("candidate() { " + body + "; }\n" + checks)
    original = subprocess.run(["bash", str(source)], capture_output=True, timeout=5)

    def candidate(args, *, merge_streams=False):
        result = subprocess.run(["bash", "-c", body, "--", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge_streams else subprocess.PIPE, timeout=5)
        return {
            "stdout": base64.b64encode(result.stdout).decode("ascii"),
            "stderr": base64.b64encode(result.stderr or b"").decode("ascii"),
            "returncode": result.returncode,
        }

    # `return` must run inside a function, as it does in the candidate container.
    body = "candidate() { " + body + '; }; candidate "$@"'
    transported = run_checks(candidate, "candidate", checks)
    assert original.returncode == transported.returncode == 0


def test_trusted_shell_cannot_swallow_failed_candidate_transport():
    def unavailable(args, **kwargs):
        raise EOFError("candidate worker exited")

    with pytest.raises(PreparationError) as failure:
        run_checks(unavailable, "candidate", "candidate || true")
    assert failure.value.verdict.status is Status.SCORED
    assert failure.value.verdict.reward == 0


@pytest.mark.parametrize("stdout", ["not base64!", "YWFh" * 262144], ids=["invalid", "oversized"])
def test_trusted_shell_cannot_swallow_malformed_observations(stdout):
    def malformed(args, **kwargs):
        return {"stdout": stdout, "stderr": "", "returncode": 0}

    with pytest.raises(PreparationError) as failure:
        run_checks(malformed, "candidate", "candidate || true")
    assert failure.value.verdict.status is Status.INFRA_ERROR


def test_shell_preparation_preserves_newline_before_source_closing_brace():
    problem = {"prompt": "candidate() {", "test": "}\ncandidate\n"}
    prepared = prepare_shell(problem, "candidate() {\nprintf answer # comment", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    result = namespace["shell_candidate"]([])
    assert base64.b64decode(result["stdout"]) == b"answer"
    assert result["returncode"] == 0


@pytest.mark.parametrize("suffix", ["\n)\n", "; exit 0\n"])
def test_candidate_source_failure_cannot_be_swallowed(suffix):
    problem = {"prompt": "candidate() {", "test": "}\ncandidate || true\n"}
    prepared = prepare_shell(problem, "candidate() { printf answer; }" + suffix + "{ :", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    with pytest.raises(PreparationError) as failure:
        run_checks(namespace["shell_candidate"], "candidate", "candidate || true")
    assert failure.value.verdict.status is Status.SCORED
    assert failure.value.verdict.reward == 0


def test_valid_source_last_status_does_not_prevent_function_invocation():
    problem = {"prompt": "candidate() {", "test": "}\ntest \"$(candidate)\" = answer\n"}
    prepared = prepare_shell(problem, "candidate() { printf answer; }\n{ false", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    assert run_checks(namespace["shell_candidate"], "candidate", problem["test"][2:]).returncode == 0


def test_invalid_trusted_shell_is_rejected_before_candidate_execution(tmp_path):
    marker = tmp_path / "executed"
    problem = {"prompt": "candidate() {", "test": "}\ncandidate\n)\n"}
    with pytest.raises(InvalidPreparation) as failure:
        prepare_shell(problem, "candidate() { touch " + str(marker), timeout=5)
    assert failure.value.verdict.status is Status.INVALID_TASK
    assert not marker.exists()


def test_humaneval_106_partial_source_parse_cannot_receive_credit(tmp_path):
    data = Path(__file__).resolve().parents[2] / "eval/chat_benchmarks/HumanEval/data/humaneval-sh.jsonl"
    problem = next(row for row in map(json.loads, data.read_text().splitlines()) if row["task_id"] == "HumanEval_106_f")
    generation = (
        'f() { local n=$1 i fact=1 out="" val; for ((i=1;i<=n;i++)); do fact=$((fact*i)); '
        'if (( i%2==0 )); then val=$fact; else val=$((i*(i+1)/2)); fi; '
        'out+="${out:+ }$val"; done; printf "%s" "$out"; }\n)'
    )
    original = tmp_path / "original.sh"
    original.write_text(generation + "\n" + problem["test"])
    assert subprocess.run(["bash", str(original)], capture_output=True, timeout=5).returncode == 2
    prepared = prepare_shell(problem, generation, timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    with pytest.raises(PreparationError) as failure:
        run_checks(namespace["shell_candidate"], "f", problem["test"][2:])
    assert failure.value.verdict.status is Status.SCORED
    assert failure.value.verdict.reward == 0


def test_proxy_start_failure_cannot_be_swallowed(monkeypatch):
    monkeypatch.setattr(verifyit_shell.sys, "executable", "/missing-shell-proxy-python")

    def candidate(args, **kwargs):
        raise AssertionError("Unavailable proxy cannot reach candidate")

    with pytest.raises(PreparationError) as failure:
        run_checks(candidate, "candidate", "candidate || true")
    assert failure.value.verdict.status is Status.INFRA_ERROR






@pytest.mark.parametrize("suffix", [
    '\nexit 0\n{ :',
    '\nprintf loaded > "$2"; exit 0\n{ :',
    '\ncandidate() { printf second;',
    '\nvalue=second\nhelper() { :',
])
def test_first_function_policy_ignores_source_initialization(suffix):
    problem = {"prompt": "candidate() {", "test": '}\ntest "$(candidate)" = first\n'}
    prepared = prepare_shell(problem, "candidate() { printf first; }" + suffix, timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    assert run_checks(namespace["shell_candidate"], "candidate", problem["test"][2:]).returncode == 0
    assert prepared.provenance["policy"] == "isolated_first_shell_function_v2"


def test_first_function_policy_preserves_function_exit125():
    problem = {"prompt": "candidate() {", "test": '}\ncandidate; test "$?" -eq 125\n'}
    prepared = prepare_shell(problem, "candidate() { exit 125", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    assert run_checks(namespace["shell_candidate"], "candidate", problem["test"][2:]).returncode == 0


def test_function_import_does_not_execute_redirections_or_startup(tmp_path, monkeypatch):
    startup = tmp_path / "startup.sh"
    marker = tmp_path / "executed"
    startup_marker = tmp_path / "startup-executed"
    startup.write_text(f"touch {startup_marker}\n")
    monkeypatch.setenv("BASH_ENV", str(startup))
    monkeypatch.setenv("BASH_FUNC_builtin%%", f"() {{ touch {startup_marker}; }}")
    problem = {"prompt": "candidate() {", "test": "}\ncandidate\n"}
    # The redirection belongs to the function invocation; import must not execute it.
    prepared = prepare_shell(problem, f"candidate() {{ printf answer; }} > >(printf x >> {marker}; cat)\n{{ :", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    result = namespace["shell_candidate"]([])
    assert result["returncode"] == 0
    assert base64.b64decode(result["stdout"]) == b"answer"
    assert marker.read_text() == "x"
    assert not startup_marker.exists()


def test_unknown_shell_policy_rejects_before_candidate(tmp_path):
    marker = tmp_path / "executed"
    problem = {"prompt": "candidate() {", "test": "}\ncandidate\n"}
    with pytest.raises(InvalidPreparation) as failure:
        prepare_shell(problem, f"candidate() {{ touch {marker}", timeout=5, policy="source")
    assert failure.value.verdict.status is Status.INVALID_TASK
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["builtin", "command", "declare"])
def test_function_named_after_builtin_is_not_invoked_during_import(entry):
    problem = {"prompt": entry + "() {", "test": "}\n" + entry + "\n"}
    prepared = prepare_shell(problem, entry + "() { printf invoked", timeout=5)
    namespace = {}
    exec(prepared.prediction, namespace)
    result = namespace["shell_candidate"]([])
    assert base64.b64decode(result["stdout"]) == b"invoked"
    assert result["returncode"] == 0



def test_reference_prevalidation_consumes_the_sample_preparation_budget(tmp_path, monkeypatch):
    bash = tmp_path / "bash"
    bash.write_text("#!/bin/sh\nsleep 0.6\nexec /bin/bash \"$@\"\n")
    bash.chmod(0o700)
    docker = tmp_path / "docker"
    marker = tmp_path / "runtime-started"
    docker.write_text("#!/bin/sh\ntouch " + str(marker) + "\nexit 1\n")
    docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(verifyit_humaneval, "DOCKER", str(docker))
    problem = {"task_id": "budget", "prompt": "candidate() {", "test": "}\ncandidate\n"}
    problems, samples = tmp_path / "problems.jsonl", tmp_path / "samples.jsonl"
    problems.write_text(json.dumps(problem) + "\n")
    samples.write_text(json.dumps({"task_id": "budget", "generation": "candidate() { printf answer"}) + "\n")
    with pytest.raises(PreparationError) as failure:
        evaluate_functional_correctness(input_file=samples, problem_file=problems, timeout=1)
    assert failure.value.verdict.status is Status.INFRA_ERROR
    assert failure.value.verdict.reward == 0
    assert not marker.exists()


def test_shell_record_io_failure_is_distinct_from_malformed_records(tmp_path):
    with pytest.raises(PreparationError) as failure:
        load_records(tmp_path / "missing.jsonl")
    assert failure.value.verdict.status is Status.INFRA_ERROR
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"task_id": 1, "task_id": 2}\n')
    with pytest.raises(InvalidTask):
        load_records(malformed)
