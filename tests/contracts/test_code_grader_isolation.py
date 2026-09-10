# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for functional-correctness sandbox launch isolation."""

import json
import os
from pathlib import Path

import pytest


def _record_worker_process(task_id, sample, _language, _timeout, _tmp_dir, completion_id):
    Path(sample["pid_file"]).write_text(str(os.getpid()))
    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "passed": True,
    }


def _write_evaluation_fixture(tmp_path, is_mbpp):
    problem = {
        "task_id": "python/0",
        "prompt": "def answer():\n",
        "test": ["assert answer() == 42"] if is_mbpp else "assert answer() == 42",
    }
    sample = {
        "task_id": "python/0",
        "prompt": problem["prompt"],
        "generation": "def answer():\n    return 42",
    }
    problem_file = tmp_path / "problems.jsonl"
    input_file = tmp_path / "samples.jsonl"
    problem_file.write_text(json.dumps(problem) + "\n")
    input_file.write_text(json.dumps(sample) + "\n")
    return problem_file, input_file, sample


@pytest.mark.parametrize(
    ("module_path", "is_mbpp"),
    [
        ("eval.chat_benchmarks.HumanEvalPlus.human_eval_plus.evaluation", False),
        ("eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation", True),
    ],
)
def test_functional_correctness_checks_run_outside_evaluator_process(
    monkeypatch,
    tmp_path,
    module_path,
    is_mbpp,
):
    module = __import__(module_path, fromlist=["evaluation"])
    pid_file = tmp_path / "worker.pid"
    problem_file, input_file, sample = _write_evaluation_fixture(tmp_path, is_mbpp)
    sample["pid_file"] = str(pid_file)
    input_file.write_text(json.dumps(sample) + "\n")
    monkeypatch.setattr(module, "check_correctness", _record_worker_process)

    result = module.evaluate_functional_correctness(
        input_file=str(input_file),
        tmp_dir=str(tmp_path),
        n_workers=1,
        problem_file=str(problem_file),
        is_mbpp=is_mbpp,
        k=[1],
    )

    assert result["scored_count"] == 1
    assert "pass@1" in result
    assert int(pid_file.read_text()) != os.getpid()


@pytest.mark.parametrize(
    ("module_path", "is_mbpp"),
    [
        ("eval.chat_benchmarks.HumanEvalPlus.human_eval_plus.evaluation", False),
        ("eval.chat_benchmarks.MBPPPlus.mbpp_plus.evaluation", True),
    ],
)
def test_spawned_worker_can_launch_functional_correctness_sandbox(tmp_path, module_path, is_mbpp):
    module = __import__(module_path, fromlist=["evaluation"])
    problem_file, input_file, _sample = _write_evaluation_fixture(tmp_path, is_mbpp)

    result = module.evaluate_functional_correctness(
        input_file=str(input_file),
        tmp_dir=str(tmp_path),
        n_workers=1,
        timeout=1.0,
        problem_file=str(problem_file),
        is_mbpp=is_mbpp,
        k=[1],
    )

    assert result["scored_count"] == 1
    assert "pass@1" in result
