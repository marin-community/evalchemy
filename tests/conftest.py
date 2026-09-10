# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared fakes for evaluation-boundary tests."""

from typing import Any

import pytest

from eval.task import BaseBenchmark


class RecordingBenchmark(BaseBenchmark):
    """Benchmark fake with controllable generation and grading outcomes."""

    def __init__(
        self,
        generation_result: Any,
        scored_result: Any = None,
        generation_error: BaseException | None = None,
        grading_error: BaseException | None = None,
    ):
        super().__init__()
        self.generation_result = generation_result
        self.scored_result = scored_result
        self.generation_error = generation_error
        self.grading_error = grading_error

    def generate_responses(self, model):
        if self.generation_error is not None:
            raise self.generation_error
        return self.generation_result

    def evaluate_responses(self, results):
        if self.grading_error is not None:
            raise self.grading_error
        return self.scored_result


class CustomTaskManager:
    """Registry fake that exposes one benchmark through the production lookup."""

    def __init__(self, task_name: str, benchmark: RecordingBenchmark):
        self.tasks = {task_name: benchmark}
        self.benchmark = benchmark

    def get_benchmark(self, task_name):
        return self.benchmark


@pytest.fixture
def evaluation_model():
    return type("EvaluationModel", (), {"rank": 0, "world_size": 1})()


@pytest.fixture
def benchmark_factory():
    return RecordingBenchmark


@pytest.fixture
def custom_task_manager_factory():
    return CustomTaskManager
