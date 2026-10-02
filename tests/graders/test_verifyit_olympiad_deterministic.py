"""Deterministic Olympiad batches retain Math rewards and fail atomically."""

import copy

import pytest
from verifyit.grade import Status

from eval.contracts.failures import GradingBoundaryError
from eval.graders.verifyit_olympiad_deterministic import grade_batch


def test_raw_first_box_and_alternative_gold_drive_math_rewards():
    examples = [{"problem": "Find a value", "answer": ["$1$, $2$"],
                 "model_outputs": [r"\boxed{2} then \boxed{9}", r"\boxed{9}", None],
                 "model_answers": ["9", "2", "2"]}]
    original = copy.deepcopy(examples)
    records = grade_batch(examples, 3)
    assert [record["reward"] for record in records[0]] == [1, 0, 0]
    assert all(record["status"] == Status.SCORED for record in records[0])
    assert examples == original


def test_invalid_later_alternative_precedes_missing_provider_output():
    examples = [{"problem": "Find a value", "answer": ["2", r"\displaystyle"]}]
    with pytest.raises(GradingBoundaryError) as raised:
        grade_batch(examples, 1)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert raised.value.verdict["reward"] == 0


def test_formatting_only_candidate_scores_zero_for_meaningful_gold():
    records = grade_batch([{"problem": "Find a value", "answer": ["2"], "model_output": r"\boxed{\displaystyle}"}], 1)
    assert records[0][0]["reward"] == 0
    assert records[0][0]["status"] == Status.SCORED


def test_subclass_errors_leave_entire_result_unchanged():
    from eval.chat_benchmarks.OlympiadBenchDeterministic.eval_instruct import OlympiadBenchDeterministicBenchmark

    benchmark = OlympiadBenchDeterministicBenchmark(verifyit_enabled=True)
    results = {"examples": [{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}", "model_answer": "2"},
                            {"problem": "bad", "answer": [r"\displaystyle"], "model_output": r"\boxed{2}", "model_answer": "2"}]}
    original = copy.deepcopy(results)
    with pytest.raises(GradingBoundaryError) as raised:
        benchmark.evaluate_responses(results)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert results == original


def test_unrepresentable_timeout_is_invalid_task():
    with pytest.raises(GradingBoundaryError) as raised:
        grade_batch([{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}"}], 1, timeout=10**400)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert raised.value.verdict["reward"] == 0


@pytest.mark.parametrize("num_samples,marker", [(2, {}), (1, {"pass_at_k": True})])
def test_task_manager_reports_inconsistent_pass_at_k_without_mutation(num_samples, marker):
    from eval.eval import _CustomTaskWork, _score_custom_task
    from eval.task import TaskManager

    manager = TaskManager(task_list=["OlympiadBenchDeterministic"], verifyit_enabled=True, num_samples=num_samples)
    benchmark = manager.benchmark_instances["OlympiadBenchDeterministic"]
    results = {"examples": [{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}"}], **marker}
    original = copy.deepcopy(results)
    outcome, scored = _score_custom_task(_CustomTaskWork("OlympiadBenchDeterministic", benchmark, results))
    assert scored == {}
    assert outcome.metrics == {}
    assert outcome.failure.grading_verdict["status"] == Status.INVALID_TASK
    assert outcome.failure.grading_verdict["reward"] == 0
    assert results == original


def test_invalid_reference_precedes_candidate_extraction_backend_failure(monkeypatch):
    from verifyit.grade import InvalidTask

    from eval.graders import answer_extraction
    from eval.graders.verifyit_olympiad_deterministic import POLICY, _grade_batch

    def unavailable_extractor(response):
        raise RuntimeError("candidate extraction backend unavailable")

    monkeypatch.setattr(answer_extraction, "extract_boxed_answer", unavailable_extractor)
    examples = [{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}"},
                {"problem": "bad", "answer": [r"\displaystyle"], "model_output": r"\boxed{2}"}]
    with pytest.raises(InvalidTask, match="boxed reference"):
        _grade_batch(examples, 1, POLICY, 30)


@pytest.mark.parametrize("stage", ["staging", "aggregation"])
def test_public_route_rejects_late_success_without_mutation(monkeypatch, stage):
    import time

    from eval.chat_benchmarks.OlympiadBench import eval_instruct as native_olympiad
    from eval.eval import _CustomTaskWork, _score_custom_task
    from eval.graders import verifyit_olympiad_deterministic as adapter
    from eval.task import TaskManager

    manager = TaskManager(task_list=["OlympiadBenchDeterministic"], verifyit_enabled=True)
    benchmark = manager.benchmark_instances["OlympiadBenchDeterministic"]
    benchmark.verifyit_timeout = 0.01 if stage == "staging" else 2
    results = {"examples": [{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}", "model_answer": "2"}]}
    original = copy.deepcopy(results)
    target = adapter if stage == "staging" else native_olympiad
    name = "stage_results" if stage == "staging" else "record_sample_metrics"
    original_function = getattr(target, name)

    def consume_budget(*args, **kwargs):
        time.sleep(benchmark.verifyit_timeout + 0.01)
        return original_function(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target, name, consume_budget)
        outcome, scored = _score_custom_task(_CustomTaskWork("OlympiadBenchDeterministic", benchmark, results))
    assert scored == {}
    assert outcome.metrics == {}
    assert outcome.failure.grading_verdict["status"] == Status.INFRA_ERROR
    assert outcome.failure.grading_verdict["reward"] == 0
    assert results == original

    benchmark.verifyit_timeout = 30
    recovered = benchmark.evaluate_responses(results)
    assert recovered["accuracy"] == 1
    assert recovered["examples"][0]["verifyit_grades"][0]["effective_options"]["timeout"] == 30


@pytest.mark.parametrize("timeout", [True, 10**400])
def test_explicit_batch_timeout_is_validated_inside_outer_deadline(timeout):
    from eval.graders.verifyit_olympiad_deterministic import total_deadline

    with total_deadline(30):
        with pytest.raises(GradingBoundaryError) as raised:
            grade_batch([{"problem": "p", "answer": ["2"], "model_output": r"\boxed{2}"}], 1, timeout=timeout)
    assert raised.value.verdict["status"] == Status.INVALID_TASK
    assert raised.value.verdict["reward"] == 0
