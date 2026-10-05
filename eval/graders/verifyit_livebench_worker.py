"""Run the pinned LiveBench coding or symbolic callback in its source runtime."""

import ast
import base64
import dataclasses
import signal
import json
import math
from pathlib import Path
import pickle
import sys
import zlib


def coding_tests(question):
    public = json.loads(question["public_test_cases"])
    try:
        private = json.loads(question["private_test_cases"])
    except json.JSONDecodeError:
        private = json.loads(pickle.loads(zlib.decompress(base64.b64decode(question["private_test_cases"]))))
    if not isinstance(public, list) or not isinstance(private, list) or not public + private:
        raise ValueError("Coding requires a nonempty trusted test population")
    for test in public + private:
        if not isinstance(test, dict) or test.get("testtype") not in ("stdin", "functional"):
            raise ValueError("Invalid coding test descriptor")
        if not isinstance(test.get("input"), str) or not isinstance(test.get("output"), str):
            raise ValueError("Coding test input and output must be strings")
    metadata = json.loads(question["original_json"]["metadata"])
    if not isinstance(metadata, dict) or (
        metadata.get("func_name") is not None
        and (not isinstance(metadata["func_name"], str) or not metadata["func_name"].strip())
    ):
        raise ValueError("Invalid coding metadata")
    return public + private, metadata.get("func_name")


def validate_question(question, task):
    if task in ("coding_completion", "LCB_generation"):
        coding_tests(question)
    else:
        reference = question.get("ground_truth")
        if isinstance(reference, list) and reference:
            reference = reference[-1]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("AMPS requires a nonempty trusted reference")
        normalized = reference.replace("\\left", "").replace("\\right", "").replace(" ^", "^").replace("\\ ", "*")
        if not normalized.strip():
            raise ValueError("AMPS reference normalizes to empty text")


def grade_coding(question, response, name, *, image):
    from lm_eval.verifyit_humaneval import candidate_function
    from livebench.lcb_runner.utils.extraction_utils import extract_code
    from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards, scored
    from verifyit.modes.grade_json_schema import grade_json_schema_candidate
    from verifyit.modes.grade_stdio import grade_stdio_candidate
    from verifyit.spec import Compare, StdioSpec

    try:
        tests, function = coding_tests(question)
        inputs = (
            [[json.loads(line) for line in test["input"].split("\n")] for test in tests]
            if function
            else [test["input"] for test in tests]
        )
        expected = [json.loads(test["output"]) for test in tests] if function else [test["output"] for test in tests]
        if any(test["testtype"] != ("functional" if function else "stdin") for test in tests):
            raise ValueError("Mixed coding input contracts")
        stdio = StdioSpec(command="", compare=Compare.DECIMAL_LINES)
        for target in expected:
            if function:
                grade_json_schema_candidate({"const": target}, target)
            else:
                grade_stdio_candidate(stdio, "", target)
    except (KeyError, TypeError, ValueError, IndexError, pickle.UnpicklingError, zlib.error) as error:
        raise InvalidTask(f"Invalid LiveBench coding tests: {error}") from error
    code = extract_code(response, lmstyle=None) if isinstance(response, str) else ""
    partial = question.get("partial_solution")
    if partial is not None and not isinstance(partial, str):
        raise InvalidTask("Coding partial solution must be text")
    if partial:
        code = partial + "\n" + code
    if not code.strip():
        return scored(0, reason="missing_code")
    source = (
        Path(__file__).resolve().parents[1]
        / "chat_benchmarks/LiveBench/livebench/lcb_runner/evaluation/testing_util.py"
    )
    prelude = next(
        node.value.value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "sol" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
    if function:
        prediction = (
            prelude
            + code
            + "\n_verifyit_entry = getattr(Solution(), "
            + repr(function)
            + ") if 'Solution' in globals() else globals()["
            + repr(function)
            + "]\n"
        )
    else:
        prediction = (
            "import contextlib, io, sys\n"
            "def _verifyit_entry(raw):\n"
            "    output = io.StringIO()\n"
            "    saved = sys.stdin\n"
            "    try:\n"
            "        sys.stdin = io.TextIOWrapper(io.BytesIO(raw.encode()))\n"
            "        with contextlib.redirect_stdout(output):\n"
            "            try:\n"
            f"                exec({(prelude + code)!r}, {{'__name__': '__main__'}})\n"
            "            except SystemExit as error:\n"
            "                if error.code not in (None, 0): raise RuntimeError('candidate exit')\n"
            "        return output.getvalue()\n"
            "    finally:\n"
            "        sys.stdin = saved\n"
        )

    def expired(signum, frame):
        raise TimeoutError("Coding candidate exceeded its six-second deadline")

    prior = signal.signal(signal.SIGALRM, expired)
    grades = []
    try:
        signal.setitimer(signal.ITIMER_REAL, 6)
        with candidate_function(prediction, "_verifyit_entry", name, image=image, memory_bytes=1024**3) as candidate:
            for arguments, target in zip(inputs, expected):
                signal.setitimer(signal.ITIMER_REAL, 6)
                actual = candidate(*arguments) if function else candidate(arguments)
                signal.setitimer(signal.ITIMER_REAL, 0)
                if function:
                    actual = list(actual) if isinstance(actual, tuple) else actual
                    verdict = grade_json_schema_candidate({"const": target}, actual)
                else:
                    verdict = grade_stdio_candidate(stdio, actual, target)
                grades.append(verdict)
                if verdict.reward == 0:
                    break
    except (ValueError, BrokenPipeError, EOFError, TimeoutError) as error:
        grades.append(scored(0, reason="candidate_execution_failed", error=str(error)))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    verdict = aggregate_rewards(grades, expected_total=len(expected), policy=Aggregation.ALL)
    verdict.detail["case_verdicts"] = [dataclasses.asdict(value) for value in grades]
    verdict.detail["test_count"] = len(expected)
    verdict.detail["container_name"] = name
    return verdict


def main():
    payload = json.loads(Path(sys.argv[1]).read_text())
    question = payload["question"]
    task = question["subtask"] if "subtask" in question else question["task"]
    try:
        validate_question(question, task)
    except (KeyError, TypeError, ValueError, IndexError, pickle.UnpicklingError, zlib.error) as error:
        Path(sys.argv[2]).write_text(json.dumps({"status": "invalid_task", "error": str(error)}))
        return
    if task in ("coding_completion", "LCB_generation"):
        raise ValueError("Coding grading must run through the trusted core supervisor")
    elif "amps_hard" in task:
        from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results

        score = (
            amps_hard_process_results(question["ground_truth"], payload["response"], False)
            if payload["response"].strip()
            else 0
        )
    else:
        raise ValueError("Unsupported LiveBench runtime task")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Source callback returned an invalid score")
    Path(sys.argv[2]).write_text(json.dumps({"score": float(score)}, allow_nan=False))


if __name__ == "__main__":
    main()
