"""Task outcomes from the single execution lifecycle."""

import asyncio
from argparse import Namespace
from dataclasses import replace

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView

from eval.completion_response import CompletionContentPolicy, CompletionText, FailedGeneration, completion_response_from_chat_choice
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.failures import FailureCategory
from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.task_outcome import EvaluationRunError, TaskRoute, TaskStatus, validate_result_document
from eval.eval import handle_evaluation_output
from eval.eval_tracker import DCEvaluationTracker
from eval.execution.driver import SourcePlan, run_evaluations
from eval.execution.grading import Grader
from eval.execution.metrics import SourceMetrics
from eval.serve_eval.results import EvalResults


class Model:
    world_size = 1
    rank = 0

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def generate_until(self, requests):
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return [output for _ in requests]


def spec(task="task", ordinal=0):
    return EvalSpec(task, SourceRef("fixture"), str(ordinal), ordinal,
                    (ModelRequest(RequestKind.GENERATE, "question"),), GraderSpec("exact_match"), target="yes")


def plan(task="task", items=None, graders=None):
    return SourcePlan(task, TaskRoute.CUSTOM, (items if items is not None else [spec(task)],),
                      (task,), SourceMetrics(repeated=False), graders or {})


def run(outputs, plans=None, **kwargs):
    return asyncio.run(run_evaluations(Model(outputs), plans or [plan()], workers=1, **kwargs))


def test_zero_score_is_a_successful_typed_outcome():
    result = run(["no"])
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    assert outcome["metrics"]["accuracy"] == 0.0
    assert (outcome["expected_count"], outcome["generated_count"], outcome["scored_count"]) == (1, 1, 1)
    validate_result_document(result)
    assert EvalResults.model_validate(result).task_outcomes["task"].status is TaskStatus.SUCCEEDED


def test_samples_are_durable_before_aggregate_commit(tmp_path):
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "run", ["task"]) as output:
        result = run(["yes"], sink=output)
        assert "samples" not in result
        assert ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1
        output.finish(result)


@pytest.mark.parametrize(("error", "category"), [
    (ConnectionError("offline"), FailureCategory.MODEL_TRANSPORT),
    (TimeoutError(), FailureCategory.AGENT_TIMEOUT),
])
def test_generation_failure_has_no_score_and_does_not_stop_other_tasks(error, category):
    result = run([error, "yes"], [plan("broken"), plan("healthy")])
    assert result["task_outcomes"]["broken"]["failure"]["category"] is category
    assert "broken" not in result["results"]
    assert result["task_outcomes"]["healthy"]["status"] is TaskStatus.SUCCEEDED


def test_exhausted_transport_is_not_an_incorrect_answer():
    result = run([FailedGeneration("model_transport")])
    assert result["results"] == {}
    outcome = result["task_outcomes"]["task"]
    assert outcome["failure"]["category"] is FailureCategory.MODEL_TRANSPORT
    assert outcome["failure_counts"] == {FailureCategory.MODEL_TRANSPORT: 1}
    record = result["samples"]["task"][0]
    assert record["failure_category"] == "model_transport"
    assert record["metrics"] == []


def test_grader_failure_is_classified_at_the_grading_boundary():
    def broken(_spec, _outputs):
        raise ConnectionError("judge unavailable")

    task = plan(items=[replace(spec(), grader=GraderSpec("broken"))], graders={"broken": Grader(broken)})
    result = run(["yes"], [task])
    assert result["results"] == {}
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE


def test_missing_per_sample_metrics_cannot_become_an_aggregate_success():
    task = plan(items=[replace(spec(), grader=GraderSpec("empty"))], graders={"empty": Grader(lambda *_: Grade({}))})
    result = run(["yes"], [task])
    assert result["results"] == {}
    assert result["task_outcomes"]["task"]["status"] is TaskStatus.FAILED


def test_sample_serialization_failure_is_typed_and_closes_the_sink():
    closed = []

    class Sink:
        def write_sample(self, task, record):
            raise ValueError("invalid sample")

        def finish_task(self):
            closed.append(True)

    result = run(["yes"], sink=Sink())
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.SERIALIZATION
    assert closed == [True]


def completion(content=None, reasoning=None):
    choice = {"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}
    response = completion_response_from_chat_choice({"choices": [choice]}, choice)
    return CompletionText(response.normalized_content(), response, CompletionContentPolicy.COMBINE)


@pytest.mark.parametrize("outputs, status", [
    ([completion(), completion("yes")], TaskStatus.FAILED),
    ([completion(reasoning="yes"), completion("yes")], TaskStatus.SUCCEEDED),
])
def test_empty_response_quality_guard_preserves_reasoning_scores(outputs, status):
    result = run(outputs, [plan(items=[spec(ordinal=0), spec(ordinal=1)])])
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is status
    assert sum(outcome["completion_response_summary"].values()) == 2
    assert len(result["samples"]["task"]) == 2


def test_empty_source_is_an_incomplete_evaluation():
    result = run([], [plan(items=[])])
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


@pytest.mark.parametrize("result", [{"results": {}}, {"results": {"task": {"accuracy": 1.0}}, "task_outcomes": {"task": {"status": "succeeded"}}}])
def test_aggregate_writer_rejects_missing_or_invalid_task_outcomes(result, tmp_path):
    args = Namespace(show_config=False, wandb_args=None, finestore_output_path=str(tmp_path / "archive"))
    with pytest.raises(EvaluationRunError):
        handle_evaluation_output(result, args, DCEvaluationTracker())
    assert not (tmp_path / "archive").exists()


def test_result_reader_rejects_results_without_outcomes():
    with pytest.raises(EvaluationRunError, match="no task outcomes"):
        EvalResults.model_validate({"results": {"task": {"accuracy": 1.0}}})


def test_predict_only_exports_through_the_same_pool_without_invoking_a_grader():
    task = plan(items=[replace(spec(), grader=GraderSpec("must_not_run"))])
    result = run(["candidate output"], [task], predict_only=True)
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.EXPORTED
    assert outcome["generated_count"] == 1
    assert outcome["scored_count"] is None
    assert result["results"] == {}
    assert result["samples"]["task"][0]["resps"] == [["candidate output"]]
