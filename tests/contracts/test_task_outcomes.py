# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for the task-outcome boundary."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from eval.contracts.failures import ModelRequestValidationError
from eval.contracts.grading import GenerationArtifactManifest
from eval.contracts.sample_manifest import SampleManifest, SampleRequest
from eval.contracts.task_outcome import (
    EvaluationRunError,
    FailureCategory,
    FailurePhase,
    TaskRoute,
    TaskStatus,
    classify_task_exception,
    lm_eval_task_outcome,
    validate_result_document,
)
from eval.eval import CHAT_BENCHMARK_ROUTE, LM_EVAL_ROUTE, evaluate, handle_evaluation_output
from eval.robust_api import request_failure_placeholder
from eval.serve_eval.results import EvalResults
from eval.task import BaseBenchmark


class _Model:
    rank = 0
    world_size = 1


class _Benchmark(BaseBenchmark):
    def __init__(self, generation_result, scored_result=None, generation_error=None, grading_error=None):
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


class _CustomTasks:
    def __init__(self, task_name, benchmark):
        self.tasks = {task_name: benchmark}
        self.benchmark = benchmark

    def get_benchmark(self, task_name):
        return self.benchmark


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


def _custom_evaluate(benchmark, **arg_overrides):
    task = "contract_task"
    return evaluate(
        lm=_Model(),
        task_manager=_CustomTasks(task, benchmark),
        pretrain_task_manager=SimpleNamespace(all_tasks={}),
        task_list=[task],
        task_routes={task: CHAT_BENCHMARK_ROUTE},
        batch_sizes_list=[1],
        args=_args(**arg_overrides),
    )


def _lm_eval_evaluate(monkeypatch, result=None, error=None):
    def fake_simple_evaluate(*args, **kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr("eval.resume.lm_eval_native.resume_simple_evaluate", fake_simple_evaluate)
    task = "arc_easy"
    return evaluate(
        lm=_Model(),
        task_manager=_NoCustomTasks(),
        pretrain_task_manager=SimpleNamespace(all_tasks={task: object()}),
        task_list=[task],
        task_routes={task: LM_EVAL_ROUTE},
        batch_sizes_list=[1],
        args=_args(),
    )


def test_custom_task_empty_metrics_fail_the_run():
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={})

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark)

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_generation_exception_taxonomy_distinguishes_policy_and_transport_failures():
    assert (
        classify_task_exception(FailurePhase.GENERATION, ModelRequestValidationError("empty prompt"))
        is FailureCategory.GENERATION_POLICY
    )
    assert (
        classify_task_exception(FailurePhase.GENERATION, TimeoutError("endpoint timeout"))
        is FailureCategory.MODEL_TRANSPORT
    )


def test_custom_task_zero_score_is_a_successful_typed_outcome():
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    result = _custom_evaluate(benchmark)

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


def test_custom_grader_infrastructure_failure_cannot_become_a_score():
    benchmark = _Benchmark(
        {"examples": [{"prompt": "x"}]},
        grading_error=RuntimeError("sandbox failed"),
    )

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark)

    assert raised.value.outcomes[0].failure.category is FailureCategory.GRADER_INFRASTRUCTURE


def test_endpoint_transport_failure_cannot_reach_the_grader_as_model_text():
    class _TransportFailureBenchmark(_Benchmark):
        def generate_responses(self, model):
            return {
                "examples": [
                    {"output": request_failure_placeholder(TimeoutError("endpoint timeout"))}
                ]
            }

        def evaluate_responses(self, results):
            raise AssertionError("transport failures must not be scored")

    benchmark = _TransportFailureBenchmark({})

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark)

    assert raised.value.outcomes[0].failure.category is FailureCategory.MODEL_TRANSPORT


def test_artifacts_are_cleaned_when_grader_infrastructure_fails():
    artifacts = GenerationArtifactManifest.temporary()
    artifact = artifacts.write_jsonl("generated", "generated.jsonl", [{"id": "a"}], expected_count=1)
    benchmark = _Benchmark(
        {"examples": [{"prompt": "x"}], "artifacts": artifacts},
        grading_error=RuntimeError("grader crashed"),
    )

    with pytest.raises(EvaluationRunError):
        _custom_evaluate(benchmark)

    assert not artifact.path.parent.exists()


def test_successful_artifact_manifest_is_persisted_before_cleanup():
    artifacts = GenerationArtifactManifest.temporary()
    artifact = artifacts.write_jsonl("generated", "generated.jsonl", [{"id": "a"}], expected_count=1)
    benchmark = _Benchmark(
        {"examples": [{"prompt": "x"}], "artifacts": artifacts},
        scored_result={"accuracy": 1.0},
    )

    result = _custom_evaluate(benchmark)

    assert result["generation_artifacts"]["contract_task"]["artifacts"][0]["expected_count"] == 1
    assert not artifact.path.parent.exists()
    validate_result_document(result)


def test_requested_sample_serialization_failure_is_typed_and_terminal():
    class _SerializationFailureBenchmark(_Benchmark):
        def to_samples(self, generation_result, scored_result):
            raise TypeError("sample is not serializable")

    benchmark = _SerializationFailureBenchmark(
        {"examples": [{"prompt": "x"}]},
        scored_result={"accuracy": 1.0},
    )

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark, log_samples=True)

    assert raised.value.outcomes[0].failure.category is FailureCategory.SERIALIZATION


def test_custom_task_rejects_scoring_coverage_drift():
    benchmark = _Benchmark(
        {"examples": [{"prompt": "a"}, {"prompt": "b"}]},
        scored_result={"accuracy": 1.0, "scored_count": 1},
    )
    benchmark._sample_manifest = SampleManifest("contract_task")
    entries = benchmark.sample_manifest.plan_batch(
        [SampleRequest(source_id="a", ordinal=0), SampleRequest(source_id="b", ordinal=1)]
    )
    benchmark.sample_manifest.mark_generated(entries, ["x", "y"])

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark)

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION
    assert "manifest" in raised.value.outcomes[0].failure.message


def test_custom_generation_exception_is_classified():
    benchmark = _Benchmark(None, generation_error=RuntimeError("endpoint stopped"))

    with pytest.raises(EvaluationRunError) as raised:
        _custom_evaluate(benchmark)

    assert raised.value.outcomes[0].failure.category is FailureCategory.GENERATION
    assert raised.value.outcomes[0].failure.exception_type == "RuntimeError"


def test_lm_eval_empty_results_fail_the_run(monkeypatch):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(monkeypatch, result={"results": {}})

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_lm_eval_result_for_a_different_task_fails_the_requested_task(monkeypatch):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(
            monkeypatch,
            result={"results": {"arc_challenge": {"acc,none": 1.0}}},
        )

    assert raised.value.outcomes[0].failure.category is FailureCategory.INCOMPLETE_EVALUATION


def test_lm_eval_zero_score_is_a_successful_typed_outcome(monkeypatch):
    result = _lm_eval_evaluate(
        monkeypatch,
        result={
            "results": {"arc_easy": {"acc,none": 0.0}},
            "n-samples": {"arc_easy": {"original": 1, "effective": 1}},
        },
    )

    assert result["results"]["arc_easy"] == {"acc,none": 0.0}
    assert result["task_outcomes"]["arc_easy"]["status"] is TaskStatus.SUCCEEDED
    assert result["task_outcomes"]["arc_easy"]["scored_count"] == 1


def test_lm_eval_outcome_rejects_result_count_that_disagrees_with_manifest():
    manifest = SampleManifest("arc_easy")
    entries = manifest.plan_batch([SampleRequest(source_id="a", ordinal=0), SampleRequest(source_id="b", ordinal=1)])
    manifest.mark_generated(entries, ["x", "y"])

    outcome = lm_eval_task_outcome(
        "arc_easy",
        TaskRoute.LM_EVAL,
        {
            "results": {"arc_easy": {"acc,none": 1.0}},
            "n-samples": {"arc_easy": {"original": 2, "effective": 1}},
        },
        manifest,
    )

    assert outcome.status is TaskStatus.FAILED
    assert outcome.failure.category is FailureCategory.INCOMPLETE_EVALUATION
    assert "manifest" in outcome.failure.message


def test_lm_eval_exception_is_classified_instead_of_becoming_empty_success(monkeypatch):
    with pytest.raises(EvaluationRunError) as raised:
        _lm_eval_evaluate(monkeypatch, error=RuntimeError("grader crashed"))

    assert raised.value.outcomes[0].failure.category is FailureCategory.GRADER_INFRASTRUCTURE
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


def test_persisted_result_reader_retains_typed_task_outcomes():
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    loaded = EvalResults.model_validate(_custom_evaluate(benchmark))

    assert loaded.task_outcomes["contract_task"].status is TaskStatus.SUCCEEDED
    assert loaded.task_outcomes["contract_task"].metrics == {"accuracy": 0.0}


def test_persisted_result_reader_rejects_legacy_results_without_outcomes():
    with pytest.raises(EvaluationRunError, match="no task outcomes"):
        EvalResults.model_validate({"results": {"task": {"accuracy": 1.0}}})
