# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry-wide coverage for the shared benchmark lifecycle schema."""

import ast
from pathlib import Path

import pytest

from eval.contracts.conformance import (
    build_task_contract_registry,
    discover_task_contracts,
    validate_custom_benchmark_class,
)
from eval.contracts.task_outcome import TaskRoute
from eval.task import BaseBenchmark


def test_every_registered_custom_and_lm_eval_name_has_a_complete_contract():
    custom_root = Path("eval/chat_benchmarks")
    custom_names = {path.parent.name for path in custom_root.glob("*/eval_instruct.py")}
    contracts = discover_task_contracts(custom_root, Path("eval/lm_eval_tasks"))

    assert custom_names == {contract.task_name for contract in contracts if contract.route is TaskRoute.CUSTOM}
    assert any(contract.route is TaskRoute.LM_EVAL for contract in contracts)


def test_every_custom_benchmark_uses_a_shared_generation_boundary():
    root = Path("eval/chat_benchmarks")
    for path in root.glob("*/eval_instruct.py"):
        tree = ast.parse(path.read_text())
        # Some variants inherit generation unchanged from another benchmark.
        if not any(isinstance(node, ast.FunctionDef) and node.name == "generate_responses" for node in ast.walk(tree)):
            continue
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(
            isinstance(call, ast.Attribute)
            and call.attr in {"compute", "generate_seeded_repeats"}
            and isinstance(call.value, ast.Name)
            and call.value.id == "self"
            for call in calls
        ), path
        assert not any(
            isinstance(call, ast.Attribute)
            and call.attr in {"generate_until", "loglikelihood", "loglikelihood_rolling"}
            and isinstance(call.value, ast.Name)
            and call.value.id == "model"
            for call in calls
        ), path


def test_custom_class_audit_rejects_untyped_resources_and_execution_modes():
    class Benchmark(BaseBenchmark):
        def generate_responses(self, model):
            return {}

        def evaluate_responses(self, results):
            return {"score": 1.0}

    Benchmark.RESOURCE_REQUIREMENTS = []
    with pytest.raises(TypeError, match="must be a tuple"):
        validate_custom_benchmark_class(Benchmark, BaseBenchmark)

    Benchmark.RESOURCE_REQUIREMENTS = ()
    Benchmark.GRADER_EXECUTION_MODE = "background_magic"
    with pytest.raises(ValueError, match="background_magic"):
        validate_custom_benchmark_class(Benchmark, BaseBenchmark)


def test_custom_class_audit_rejects_overriding_the_shared_inference_guard():
    class Benchmark(BaseBenchmark):
        def generate_responses(self, model):
            return {}

        def evaluate_responses(self, results):
            return {"score": 1.0}

        def compute(self, model, inputs):
            return model.generate_until(inputs)

    with pytest.raises(TypeError, match="must not override BaseBenchmark.compute"):
        validate_custom_benchmark_class(Benchmark, BaseBenchmark)


def test_a_task_cannot_ambiguously_belong_to_both_routes():
    with pytest.raises(ValueError, match="both .* and lm-eval"):
        build_task_contract_registry(["collision"], ["collision"])
