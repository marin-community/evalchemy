"""Evaluate shell completions with protected core Pytest and core aggregation."""

import json
import os
import time
import uuid
from pathlib import Path

from harbor_config.errors import ErrorCategory
from lm_eval.verifyit_humaneval import grade_function
from verifyit.grade import Aggregation, InvalidTask, Status, aggregate_rewards
from verifyit.json_objects import unique_object
from verifyit.file_ops.read import read_text

from eval.contracts.sample_results import PER_TASK_PASS_RATE_FIELD
from eval.graders.verifyit_shell import ShellPolicy, preparation_error, prepare_shell

IMAGE = os.environ.get("VERIFYIT_CODE_IMAGE", "verifyit-code:python-v1")


def grade_completion(problem, generation, *, timeout, shell_policy=ShellPolicy.FIRST_FUNCTION):
    deadline = time.monotonic() + timeout
    frame_limit = problem.get("_verifyit_max_rpc_bytes", 1048576)
    if type(frame_limit) is not int or frame_limit not in (1048576, 16 * 1048576):
        raise InvalidTask("Unsupported trusted function frame bound")
    observation = problem.get("_verifyit_result_observation", "identity")
    if observation not in ("identity", "bool_or_presence"):
        raise InvalidTask("Unsupported trusted result observation")
    prepared = prepare_shell(problem, generation, timeout=max(0.0, deadline - time.monotonic()), max_bytes=frame_limit, policy=shell_policy)
    reference, generation = prepared.reference, prepared.prediction
    preparation = prepared.provenance
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, "TimeoutError", "Shell preparation exceeded its deadline", "shell_prepare")
    return grade_function(
        reference, generation, "verifyit-code-" + uuid.uuid4().hex,
        timeout=remaining, image=IMAGE, result_observation=observation, max_bytes=frame_limit,
        preparation=preparation,
    )


def load_records(path):
    try:
        rows = [json.loads(line, object_pairs_hook=unique_object) for line in read_text(Path(path)).splitlines() if line.strip()]
    except OSError as error:
        raise preparation_error(Status.INFRA_ERROR, ErrorCategory.UNKNOWN, type(error).__name__, str(error), "input_read") from error
    except ValueError as error:
        raise InvalidTask("Code records must contain unambiguous JSON objects") from error
    if any(not isinstance(row, dict) or type(row.get("task_id")) not in (str, int) for row in rows):
        raise InvalidTask("Code records require string or integer task IDs")
    return rows


def evaluate_functional_correctness(
    input_file=None, tmp_dir="./", n_workers=32, timeout=10.0, problem_file=None,
    out_dir=None, k=(1,), test_groundtruth=False, language="sh", shell_policy=ShellPolicy.FIRST_FUNCTION,
):
    if language != "sh":
        raise InvalidTask("Shell grading requires language=sh")
    if test_groundtruth or out_dir is not None:
        raise InvalidTask("Code grading requires a completion file without output side effects")
    if not isinstance(k, (list, tuple)) or len(k) != 1 or type(k[0]) is not int or k[0] != 1:
        raise InvalidTask("Shell grading supports pass@1 only")
    problems = {}
    for row in load_records(problem_file):
        identifier = row["task_id"]
        if identifier in problems:
            raise InvalidTask("Duplicate trusted task ID")
        problems[identifier] = row
    samples = load_records(input_file)
    if not samples:
        raise InvalidTask("Code evaluation requires completions")
    identifiers = [sample.get("task_id") for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise InvalidTask("Code cutover requires one completion per task")
    if any(identifier not in problems for identifier in identifiers):
        raise InvalidTask("Unknown code task ID")
    # Prepare every trusted reference before executing any candidate.
    remaining_budgets = {}
    for identifier in identifiers:
        started = time.monotonic()
        prepare_shell(problems[identifier], "", timeout=timeout, policy=shell_policy)
        remaining_budgets[identifier] = timeout - (time.monotonic() - started)
    verdicts = {
        sample["task_id"]: grade_completion(
            problems[sample["task_id"]], sample.get("generation"), timeout=remaining_budgets[sample["task_id"]], shell_policy=shell_policy,
        )
        for sample in samples
    }
    combined = aggregate_rewards(list(verdicts.values()), expected_total=len(samples), policy=Aggregation.MEAN)
    if combined.status is Status.INVALID_TASK:
        raise InvalidTask(str(combined.detail))
    if combined.status is not Status.SCORED:
        raise RuntimeError(f"Code grading failed: {combined.detail}")
    metrics = {"pass@1": combined.reward}
    metrics[PER_TASK_PASS_RATE_FIELD] = {identifier: verdict.reward for identifier, verdict in verdicts.items()}
    return metrics
