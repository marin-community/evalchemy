# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for the shared task-preparation boundary."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from lm_eval.api.instance import Instance

from eval.contracts.preflight import (
    EvaluationPreflightError,
    ModelRequestValidationError,
    PackageFileRequirement,
    PythonDependencyRequirement,
    TaskPreparationStatus,
    prepare_requested_tasks,
    validate_task_preparations,
)
from eval.contracts.task_outcome import FailureCategory, TaskRoute
from eval.task import BaseBenchmark, TaskManager


class _Benchmark(BaseBenchmark):
    RESOURCE_REQUIREMENTS = ()

    def generate_responses(self, model):
        raise NotImplementedError

    def evaluate_responses(self, results):
        raise NotImplementedError


class _CustomManager:
    load_failures = {}

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def get_benchmark(self, task_name):
        return self.benchmark


def test_package_file_and_dependency_are_validated():
    benchmark = _Benchmark()
    benchmark.RESOURCE_REQUIREMENTS = (
        PackageFileRequirement("eval.contracts", "__init__.py"),
        PythonDependencyRequirement("json"),
    )

    preparation = benchmark.prepare()

    assert preparation.status is TaskPreparationStatus.READY
    assert [resource.kind for resource in preparation.resources] == ["package_file", "python_dependency"]


def test_missing_resource_is_a_typed_preflight_failure():
    benchmark = _Benchmark()
    benchmark.RESOURCE_REQUIREMENTS = (
        PackageFileRequirement("eval.contracts", "missing.txt"),
    )

    preparation = benchmark.prepare()

    assert preparation.status is TaskPreparationStatus.FAILED
    assert preparation.failure.category is FailureCategory.RESOURCE
    assert preparation.failure.exception_type == "FileNotFoundError"


def test_persisted_preparation_rejects_unknown_schema():
    preparation = _Benchmark().prepare("task").to_dict()
    preparation["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        validate_task_preparations({"task": preparation}, ["task"])


def test_requested_custom_load_failure_is_not_silently_omitted(tmp_path):
    broken = tmp_path / "Broken"
    broken.mkdir()
    (broken / "eval_instruct.py").write_text("raise RuntimeError('broken import')\n")
    manager = TaskManager(benchmarks_dir=str(tmp_path), task_list=["Broken"])

    with pytest.raises(EvaluationPreflightError) as raised:
        prepare_requested_tasks(
            ["Broken"],
            {"Broken": TaskRoute.CUSTOM},
            manager,
            SimpleNamespace(load_task_or_group=lambda tasks: None),
        )

    assert raised.value.preparations[0].failure.exception_type == "RuntimeError"


def test_lm_eval_task_is_constructed_during_preflight():
    calls = []
    lm_manager = SimpleNamespace(load_task_or_group=lambda tasks: calls.append(tasks) or {"arc_easy": object()})

    preparations = prepare_requested_tasks(
        ["arc_easy"],
        {"arc_easy": TaskRoute.LM_EVAL},
        _CustomManager(None),
        lm_manager,
    )

    assert calls == [["arc_easy"]]
    assert preparations[0].status is TaskPreparationStatus.READY
    assert preparations[0].route is TaskRoute.LM_EVAL


def test_invalid_representative_request_fails_before_generation():
    class _Model:
        rank = 0
        world_size = 1

        def generate_until(self, requests):
            raise AssertionError("generation must not receive an invalid request")

    request = Instance("generate_until", {}, ("", {}), 0)

    with pytest.raises(ModelRequestValidationError):
        _Benchmark().compute(_Model(), [request])


def test_mmlupro_prompt_resolution_is_independent_of_cwd(tmp_path):
    script = """
from eval.chat_benchmarks.MMLUPro.eval_instruct import generate_cot_prompt
record = {
    "question": "Q?",
    "options": ["A", "B"],
    "answer": "A",
    "answer_index": 0,
    "cot_content": "A: Let's think step by step. A",
    "category": "math",
}
prompt = generate_cot_prompt([record], record, 1)
assert "Q?" in prompt
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(tmp_path),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2])},
    )

    assert completed.returncode == 0, completed.stderr
