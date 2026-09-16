# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for the shared task-preparation boundary."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest
from lm_eval.api.instance import Instance
from lm_eval.models.api_models import JsonChatStr

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
from eval.eval import cli_evaluate, setup_custom_parser
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


class _RecordingModel:
    rank = 0
    world_size = 1

    def __init__(self):
        self.requests = []

    def generate_until(self, requests):
        self.requests.extend(requests)
        return ["answer" for _ in requests]

    def apply_chat_template(self, messages):
        return JsonChatStr(json.dumps(messages))


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


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        JsonChatStr('[{"role": "user", "content": ""}]'),
        JsonChatStr("not-json"),
    ],
)
def test_invalid_representative_request_fails_before_generation(prompt):
    model = _RecordingModel()
    request = Instance("generate_until", {}, (prompt, {}), 0)

    with pytest.raises(ModelRequestValidationError):
        _Benchmark().compute(model, [request])

    assert model.requests == []


def test_json_chat_request_reaches_generation():
    model = _RecordingModel()
    request = Instance(
        "generate_until",
        {},
        (JsonChatStr('[{"role": "user", "content": "question"}]'), {}),
        0,
    )

    assert _Benchmark().compute(model, [request]) == ["answer"]
    assert model.requests == [request]


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


@pytest.mark.parametrize(
    ("num_fewshot", "includes_validation"),
    [(0, False), (1, True)],
)
def test_mmlupro_respects_shot_count_with_model_tokenizer(monkeypatch, num_fewshot, includes_validation):
    validation = {
        "question": "Validation demonstration?",
        "options": ["demo-a", "demo-b"],
        "answer": "A",
        "answer_index": 0,
        "cot_content": "A: Let's think step by step. demo-a",
        "category": "math",
    }
    question = {
        "question": "Held-out question?",
        "options": ["test-a", "test-b"],
        "answer": "B",
        "answer_index": 1,
        "cot_content": "A: Let's think step by step. test-b",
        "category": "math",
    }
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: {"validation": [validation], "test": [question]},
    )

    model = _RecordingModel()

    class _ModelTokenizer:
        """Callable tokenizer stand-in that can only count encoded ids."""

        @staticmethod
        def encode(_prompt):
            if num_fewshot == 0:
                raise AssertionError("zero-shot prompting must not load or call a tokenizer")
            return [0] * 32

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("prompt sizing must count encoded ids, not build tensors")

    def fail_auto_tokenizer(*_args, **_kwargs):
        raise AssertionError("few-shot prompting must use the evaluation model's tokenizer")

    model.tokenizer = _ModelTokenizer()
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        fail_auto_tokenizer,
    )
    args = setup_custom_parser().parse_args(
        [
            "--model",
            "hf",
            "--tasks",
            "MMLUPro",
            "--num_fewshot",
            str(num_fewshot),
            "--batch_size",
            "1",
            "--max_length",
            "65536",
            "--max_tokens",
            "1024",
            "--apply_chat_template",
            "--resume-mode",
            "off",
        ]
    )
    args.model = model
    monkeypatch.setattr("eval.eval.setup_evaluation_tracker", lambda *_args: None)
    monkeypatch.setattr("eval.eval.add_results_metadata", lambda *_args: None)
    monkeypatch.setattr("eval.eval.handle_evaluation_output", lambda *_args: None)

    cli_evaluate(args)

    messages = json.loads(model.requests[0].args[0].prompt)
    assert "Held-out question?" in messages[0]["content"]
    assert ("Validation demonstration?" in messages[0]["content"]) is includes_validation
