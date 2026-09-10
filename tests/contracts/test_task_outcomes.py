# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for the task-outcome boundary."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from eval.contracts.task_outcome import EvaluationRunError, FailureCategory, TaskStatus
from eval.eval import CHAT_BENCHMARK_ROUTE, LM_EVAL_ROUTE, evaluate, handle_evaluation_output
from eval.serve_eval.results import EvalResults


class _NoCustomTasks:
    tasks = {}


def _args(**overrides):
    values = {
        "model": "local-completions",
        "log_samples": False,
        "model_args": "model=test-model",
        "gen_kwargs": None,
        "num_fewshot": 0,
        "max_tokens": None,
        "limit": None,
        "evaluation_tracker": None,
        "max_batch_size": None,
        "device": None,
        "use_cache": None,
        "check_integrity": False,
        "write_out": False,
        "system_instruction": None,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "verbosity": "INFO",
        "predict_only": False,
        "confirm_run_unsafe_code": False,
        "seed": [0, 1234, 1234, 1234],
    }
    values.update(overrides)
    return Namespace(**values)


def _custom_evaluate(benchmark, evaluation_model, custom_task_manager_factory):
    task = "contract_task"
    return evaluate(
        lm=evaluation_model,
        task_manager=custom_task_manager_factory(task, benchmark),
        pretrain_task_manager=SimpleNamespace(all_tasks={}),
        task_list=[task],
        task_routes={task: CHAT_BENCHMARK_ROUTE},
        batch_sizes_list=[1],
        args=_args(),
    )


def _lm_eval_evaluate(monkeypatch, evaluation_model, result=None, error=None):
    def fake_simple_evaluate(*args, **kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr("eval.resume.lm_eval_native.resume_simple_evaluate", fake_simple_evaluate)
    task = "arc_easy"
    return evaluate(
        lm=evaluation_model,
        task_manager=_NoCustomTasks(),
        pretrain_task_manager=SimpleNamespace(all_tasks={task: object()}),
        task_list=[task],
        task_routes={task: LM_EVAL_ROUTE},
        batch_sizes_list=[1],
        args=_args(),
    )


def test_custom_task_empty_metrics_fail_the_run(
    benchmark_factory,
    custom_task_manager_factory,
    evaluation_model,
):
    benchmark = benchmark_factory({"examples": [{"prompt": "x"}]}, scored_result={})

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark, evaluation_model, custom_task_manager_factory)

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_custom_task_zero_score_is_a_successful_typed_outcome(
    benchmark_factory,
    custom_task_manager_factory,
    evaluation_model,
):
    benchmark = benchmark_factory({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    result = _custom_evaluate(benchmark, evaluation_model, custom_task_manager_factory)

    assert result["results"]["contract_task"] == {"accuracy": 0.0}
    assert result["task_outcomes"]["contract_task"] == {
        "schema_version": 1,
        "task_name": "contract_task",
        "route": CHAT_BENCHMARK_ROUTE,
        "status": TaskStatus.SUCCEEDED,
        "metrics": {"accuracy": 0.0},
        "expected_count": None,
        "generated_count": 1,
        "scored_count": 1,
        "failure": None,
    }


def test_custom_generation_exception_is_classified(
    benchmark_factory,
    custom_task_manager_factory,
    evaluation_model,
):
    benchmark = benchmark_factory(None, generation_error=RuntimeError("endpoint stopped"))

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark, evaluation_model, custom_task_manager_factory)

    assert raised.value.outcomes[0].failure.category is FailureCategory.GENERATION
    assert raised.value.outcomes[0].failure.exception_type == "RuntimeError"


def test_lm_eval_empty_results_fail_the_run(monkeypatch, evaluation_model):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(monkeypatch, evaluation_model, result={"results": {}})

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_lm_eval_result_for_a_different_task_fails_the_requested_task(monkeypatch, evaluation_model):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(
            monkeypatch,
            evaluation_model,
            result={"results": {"arc_challenge": {"acc,none": 1.0}}},
        )

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_lm_eval_zero_score_is_a_successful_typed_outcome(monkeypatch, evaluation_model):
    result = _lm_eval_evaluate(
        monkeypatch,
        evaluation_model,
        result={
            "results": {"arc_easy": {"acc,none": 0.0}},
            "n-samples": {"arc_easy": {"original": 1, "effective": 1}},
        },
    )

    assert result["results"]["arc_easy"] == {"acc,none": 0.0}
    assert result["task_outcomes"]["arc_easy"]["status"] is TaskStatus.SUCCEEDED
    assert result["task_outcomes"]["arc_easy"]["scored_count"] == 1


def test_lm_eval_exception_is_classified_instead_of_becoming_empty_success(monkeypatch, evaluation_model):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(monkeypatch, evaluation_model, error=RuntimeError("grader crashed"))

    assert raised.value.outcomes[0].failure.category is FailureCategory.GRADING
    assert raised.value.outcomes[0].failure.exception_type == "RuntimeError"


@pytest.mark.parametrize(
    "result",
    [
        {"results": {"task": {"accuracy": 1.0}}},
        {
            "results": {"task": {"accuracy": 1.0}},
            "task_outcomes": {"task": {"status": "succeeded"}},
        },
    ],
)
def test_aggregate_writer_rejects_results_without_a_valid_task_outcome(result):
    args = Namespace(log_samples=False, show_config=False, wandb_args=None)
    tracker = SimpleNamespace(
        save_results_aggregated=lambda **_kwargs: pytest.fail("invalid results must not be persisted")
    )

    with pytest.raises(EvaluationRunError):
        handle_evaluation_output(result, args, tracker)


def test_persisted_result_reader_retains_typed_task_outcomes(
    benchmark_factory,
    custom_task_manager_factory,
    evaluation_model,
):
    benchmark = benchmark_factory({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    loaded = EvalResults.model_validate(_custom_evaluate(benchmark, evaluation_model, custom_task_manager_factory))

    assert loaded.task_outcomes["contract_task"].status is TaskStatus.SUCCEEDED
    assert loaded.task_outcomes["contract_task"].metrics == {"accuracy": 0.0}
