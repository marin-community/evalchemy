# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry-wide coverage for the shared benchmark lifecycle schema."""

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

    assert custom_names == {
        contract.task_name for contract in contracts if contract.route is TaskRoute.CUSTOM
    }
    assert any(contract.route is TaskRoute.LM_EVAL for contract in contracts)


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


def test_a_task_cannot_ambiguously_belong_to_both_routes():
    with pytest.raises(ValueError, match="both .* and lm-eval"):
        build_task_contract_registry(["collision"], ["collision"])
