# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Behavioral coverage for the task-outcome boundary."""

from argparse import Namespace
from collections import Counter
import json
from types import SimpleNamespace

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from lm_eval.models.openai_completions import LocalChatCompletion

from eval.chat_benchmarks.MRCR.eval_instruct import MRCRBenchmark
from eval.completion_response import CompletionClassification, CompletionContentPolicy, FailedGeneration
from eval.contracts.aggregation import repeated_accuracy_results
from eval.contracts.custom_streaming import aggregate_responses, score_responses
from eval.contracts.failures import ModelRequestValidationError
from eval.contracts.finestore_output import FineStoreOutput, read_finestore_output
from eval.contracts.grading import GenerationArtifactManifest
from eval.contracts.sample_manifest import SampleManifest, SampleRequest
from eval.contracts.sample_results import record_repeated_sample_metrics, record_sample_metrics
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
from eval.eval import (
    CHAT_BENCHMARK_ROUTE,
    LM_EVAL_ROUTE,
    _CustomTaskWork,
    _score_custom_task,
    evaluate,
    handle_evaluation_output,
)
from eval.eval_tracker import DCEvaluationTracker
from eval.robust_api import record_completion_responses, record_endpoint_failure
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

    def grade_responses(self, results):
        if self.grading_error is not None:
            raise self.grading_error
        if self.scored_result and "accuracy" in self.scored_result:
            examples = results.get("examples", [])
            correct_count = round(self.scored_result["accuracy"] * len(examples))
            for index, example in enumerate(examples):
                record_sample_metrics(example, accuracy=index < correct_count)
        return results

    def aggregate_results(self, results):
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
        "model_args": "model=test-model",
        "gen_kwargs": None,
        "num_fewshot": 0,
        "max_tokens": None,
        "limit": None,
        "evaluation_tracker": None,
        "max_batch_size": None,
        "device": None,
        "check_integrity": False,
        "write_out": False,
        "system_instruction": None,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "verbosity": "INFO",
        "predict_only": False,
        "confirm_run_unsafe_code": False,
        "seed": [0, 1234, 1234, 1234],
        "finestore_output_prefix": "run",
    }
    values.update(overrides)
    return Namespace(**values)


def _custom_evaluate(benchmark, output_sink=None, **arg_overrides):
    task = "contract_task"
    return evaluate(
        lm=_Model(),
        task_manager=_CustomTasks(task, benchmark),
        pretrain_task_manager=SimpleNamespace(all_tasks={}),
        task_list=[task],
        task_routes={task: CHAT_BENCHMARK_ROUTE},
        batch_sizes_list=[1],
        args=_args(**arg_overrides),
        output_sink=output_sink,
    )


def _lm_eval_evaluate(monkeypatch, result=None, error=None, **arg_overrides):
    def fake_simple_evaluate(*args, **kwargs):
        if error is not None:
            raise error
        if callable(result):
            return result(*args, **kwargs)
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
        args=_args(**arg_overrides),
    )


def test_custom_task_empty_metrics_are_reported_without_a_score():
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={})

    result = _custom_evaluate(benchmark)

    assert result["results"] == {}
    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION
    validate_result_document(result)


@pytest.mark.parametrize("stream_samples", [False, True])
def test_partial_mrcr_uses_its_own_aggregator_and_preserves_failed_samples(tmp_path, stream_samples):
    examples = [
        {
            "answer": "nonce answer",
            "model_output": "nonce answer",
            "score": 0.25,
            "prefix_hit": 1.0,
            "n_needles": 2,
            "mrcr_bin_upper": 8192,
        },
        {
            "answer": "nonce answer",
            "model_output": "nonce answer",
            "score": 0.75,
            "prefix_hit": 1.0,
            "n_needles": 4,
            "mrcr_bin_upper": 16384,
        },
        {
            "answer": "nonce answer",
            "model_output": FailedGeneration("model_transport"),
            "score": 0.0,
            "prefix_hit": 0.0,
            "n_needles": 2,
            "mrcr_bin_upper": 8192,
        },
    ]
    benchmark = MRCRBenchmark()
    entries = benchmark.sample_manifest.plan_batch(
        [SampleRequest(source_id=str(index), ordinal=index) for index in range(len(examples))]
    )
    benchmark.sample_manifest.mark_generated(entries, [example["model_output"] for example in examples])
    work = _CustomTaskWork("MRCR", benchmark, {"examples": examples})
    root = str(tmp_path / "archive")
    if stream_samples:
        with FineStoreOutput(root, "mrcr", ["MRCR"]) as output:
            scored = _score_custom_task(work, output)
        rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).to_pylist(maps_as_pydicts="strict")
        assert len(rows) == 3
        assert sum("EVALCHEMY_INFRASTRUCTURE_ERROR" in (row["output"] or "") for row in rows) == 1
    else:
        scored = _score_custom_task(work)
        assert len(benchmark.to_samples(scored.graded_result, scored.metrics)) == 3

    outcome, metrics = scored.outcome, scored.metrics

    assert outcome.status is TaskStatus.SUCCEEDED
    assert outcome.generated_count == 3
    assert outcome.scored_count == 2
    assert metrics["mrcr_accuracy"] == 0.5
    assert metrics["prefix_hit_rate"] == 1.0
    assert metrics["mrcr_8192_2needle"] == 0.25
    assert metrics["mrcr_8192_2needle_count"] == 1
    assert metrics["mrcr_16384_4needle"] == 0.75
    assert metrics["mrcr_16384_4needle_count"] == 1


def test_partial_custom_task_without_an_aggregator_fails_instead_of_guessing():
    class MissingAggregator(_Benchmark):
        def aggregate_results(self, results):
            raise NotImplementedError("this benchmark has no reducer for the graded subset")

    benchmark = MissingAggregator(
        {"examples": [{"prompt": "x", "model_output": FailedGeneration("model_transport")}]},
        scored_result={"accuracy": 0.0},
    )

    result = _custom_evaluate(benchmark)

    assert result["results"] == {}
    outcome = result["task_outcomes"]["contract_task"]
    assert outcome["status"] is TaskStatus.FAILED
    assert outcome["failure"]["exception_type"] == "NotImplementedError"


def test_partial_aggregation_uses_filtered_namespaces_and_task_weights():
    class WeightedBenchmark(_Benchmark):
        def sample_groups(self, generation_result):
            return {"weighted": generation_result["examples"]}

        def grade_responses(self, results):
            for example in results["examples"]:
                record_sample_metrics(example, accuracy=example["value"])
            return results

        def aggregate_results(self, results):
            examples = results["examples"]

            return {
                "weighted_score": sum(row["value"] * row["weight"] for row in examples)
                / sum(row["weight"] for row in examples)
            }

    benchmark = WeightedBenchmark(
        {
            "examples": [
                {"model_output": "a", "value": 0.4, "weight": 1},
                {"model_output": "b", "value": 0.8, "weight": 3},
                {"model_output": FailedGeneration("model_transport"), "value": 0.0, "weight": 9},
            ]
        }
    )

    result = _custom_evaluate(benchmark)

    assert result["results"]["contract_task"]["weighted_score"] == pytest.approx(0.7)
    assert result["task_outcomes"]["contract_task"]["scored_count"] == 2
    assert len(result["samples"]["contract_task"]) == 3


def test_partial_repeated_samples_fail_after_preserving_each_attempt(tmp_path):
    class RepeatedBenchmark(_Benchmark):
        def grade_responses(self, results):
            record_repeated_sample_metrics(results["examples"][0], accuracy=[1.0, 0.0])
            return results

        def aggregate_results(self, results):
            return repeated_accuracy_results(results["examples"])

    generation = {
        "examples": [
            {
                "answer": "4",
                "model_outputs": ["4", FailedGeneration("model_transport")],
                "model_answers": ["4", ""],
            }
        ]
    }
    benchmark = RepeatedBenchmark(generation)
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "repeated", ["repeated"]) as output:
        scored = _score_custom_task(_CustomTaskWork("repeated", benchmark, generation), output)

    assert scored.outcome.status is TaskStatus.FAILED
    assert scored.metrics == {}
    rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).to_pylist(maps_as_pydicts="strict")
    assert len(rows) == 2
    assert sum("EVALCHEMY_INFRASTRUCTURE_ERROR" in (row["output"] or "") for row in rows) == 1


def test_copied_reordered_grades_keep_exported_source_identity():
    class CopyingBenchmark(_Benchmark):
        def grade_responses(self, results):
            examples = [dict(example) for example in reversed(results["examples"])]
            for example in examples:
                record_sample_metrics(example, accuracy=True)
            return {**results, "examples": examples}

    benchmark = CopyingBenchmark(
        {"examples": [{"model_output": "A"}, {"model_output": "B"}]}, scored_result={"accuracy": 1.0}
    )
    benchmark.bind_task_name("contract_task")
    entries = benchmark.sample_manifest.plan_batch(
        [SampleRequest(source_id="A", ordinal=0), SampleRequest(source_id="B", ordinal=1)]
    )
    benchmark.sample_manifest.mark_generated(entries, ["A", "B"])

    result = _custom_evaluate(benchmark)

    assert result["task_outcomes"]["contract_task"]["status"] is TaskStatus.SUCCEEDED
    assert [(row["source_id"], row["resps"]) for row in result["samples"]["contract_task"]] == [
        ("A", [["A"]]), ("B", [["B"]]),
    ]


def test_omitted_returned_grade_rejects_aggregate_but_keeps_emitted_evidence(tmp_path):
    class TruncatedBenchmark(_Benchmark):
        def grade_responses(self, results):
            for example in results["examples"]:
                record_sample_metrics(example, accuracy=True)
            return {**results, "examples": results["examples"][:1]}

    benchmark = TruncatedBenchmark(
        {"examples": [{"model_output": "A"}, {"model_output": "B"}]}, scored_result={"accuracy": 1.0}
    )
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "truncated", ["contract_task"]) as output:
        result = _custom_evaluate(benchmark, output_sink=output)

    assert result["results"] == {}
    assert result["task_outcomes"]["contract_task"]["status"] is TaskStatus.FAILED
    assert ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 2


def test_json_replay_preserves_mrcr_failures_and_uses_only_grading_evidence():
    benchmark = MRCRBenchmark()
    generated = {
        "examples": [
            {"model_output": "ok", "score": 0.75, "prefix_hit": 1.0, "n_needles": 2, "mrcr_bin_upper": 8192},
            {"model_output": FailedGeneration("model_transport"), "score": 0.0, "prefix_hit": 0.0, "n_needles": 2, "mrcr_bin_upper": 8192},
        ]
    }
    fresh = score_responses("MRCR", benchmark, generated)
    saved = json.loads(json.dumps(fresh.graded_result))
    for example in saved["examples"]:
        example.pop("score")
        example.pop("prefix_hit")
    saved["examples"].reverse()

    restored = aggregate_responses("MRCR", benchmark, saved)

    assert restored.excluded_count == fresh.excluded_count == 1
    samples = benchmark.to_samples(restored.graded_result, restored.metrics)
    assert [sample["resps"] for sample in samples] == [[["ok"]], [[""]]]
    assert samples[1]["failure_category"] == "model_transport"
    for metric in ("mrcr_accuracy", "prefix_hit_rate", "mrcr_8192_2needle", "mrcr_8192_2needle_count"):
        assert restored.metrics[metric] == fresh.metrics[metric]


def test_generation_exception_taxonomy_distinguishes_policy_and_timeout_failures():
    assert (
        classify_task_exception(FailurePhase.GENERATION, ModelRequestValidationError("empty prompt"))
        is FailureCategory.GENERATION_POLICY
    )
    assert (
        classify_task_exception(FailurePhase.GENERATION, TimeoutError("endpoint timeout"))
        is FailureCategory.AGENT_TIMEOUT
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
        "failure_counts": {},
        "completion_response_summary": {},
    }


def test_custom_task_stages_samples_before_aggregate_result(tmp_path):
    root = str(tmp_path / "archive")
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    with FineStoreOutput(root, "run", ["contract_task"]) as output:
        result = _custom_evaluate(benchmark, output_sink=output)
        assert "samples" not in result
        staged = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        assert staged is not None
        assert staged.num_rows == 1
        output.finish(result)


def test_custom_grader_persists_scored_sample_before_interruption_and_retry(tmp_path):
    root = str(tmp_path / "archive")

    class _InterruptedBenchmark(_Benchmark):
        def grade_responses(self, results):
            record_sample_metrics(results["examples"][0], accuracy=True)
            self.output.finish_task()
            raise RuntimeError("grader interrupted")

    interrupted = _InterruptedBenchmark(
        {"examples": [{"prompt": "first"}, {"prompt": "second"}]},
        scored_result={"accuracy": 1.0},
    )
    with FineStoreOutput(root, "run", ["contract_task"]) as output:
        interrupted.output = output
        result = _custom_evaluate(interrupted, output_sink=output)
        assert result["task_outcomes"]["contract_task"]["status"] is TaskStatus.FAILED
        assert read_finestore_output(root, "run") is None
        staged = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
        assert staged is not None
        assert staged.num_rows == 1

    retry = _Benchmark(
        {"examples": [{"prompt": "first"}, {"prompt": "second"}]},
        scored_result={"accuracy": 1.0},
    )
    with FineStoreOutput(root, "run", ["contract_task"]) as output:
        result = _custom_evaluate(retry, output_sink=output)
        output.finish(result)

    samples = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert samples is not None
    assert samples.num_rows == 2
    assert {row["doc_id"] for row in samples.to_pylist(maps_as_pydicts="strict")} == {"0", "1"}
    assert read_finestore_output(root, "run") is not None


def test_custom_grader_missing_sample_score_fails_task(tmp_path):
    class _PartialBenchmark(_Benchmark):
        def grade_responses(self, results):
            record_sample_metrics(results["examples"][0], accuracy=True)
            return results

    benchmark = _PartialBenchmark({"examples": [{"prompt": "first"}, {"prompt": "second"}]})
    with FineStoreOutput(str(tmp_path / "archive"), "run", ["contract_task"]) as output:
        result = _custom_evaluate(benchmark, output_sink=output)

    assert result["results"] == {}
    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


def test_custom_grader_infrastructure_failure_is_reported_without_a_score():
    benchmark = _Benchmark(
        {"examples": [{"prompt": "x"}]},
        grading_error=RuntimeError("sandbox failed"),
    )

    result = _custom_evaluate(benchmark)

    assert result["results"] == {}
    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE
    validate_result_document(result)


def test_failed_task_is_saved_in_the_results_package(tmp_path):
    result = _custom_evaluate(_Benchmark({"examples": [{"prompt": "x"}]}, grading_error=RuntimeError("sandbox failed")))
    result["config"] = {"batch_sizes": [1]}
    tracker = DCEvaluationTracker()
    tracker.general_config_tracker.model_name_sanitized = "test-model"
    args = _args(
        finestore_output_path=str(tmp_path / "archive"),
        show_config=False,
        use_database=False,
        debug=False,
        annotator_model=None,
        batch_size=1,
    )

    with FineStoreOutput(args.finestore_output_path, args.finestore_output_prefix, ["contract_task"]) as output:
        handle_evaluation_output(result, args, tracker, output_sink=output)

    saved = EvalResults.load_archive(str(tmp_path / "archive"))
    assert saved.task_outcomes["contract_task"].failure.category is FailureCategory.GRADER_INFRASTRUCTURE
    assert saved.task_outcomes["contract_task"].status is TaskStatus.FAILED
    assert saved.results == {}


def test_endpoint_transport_failure_is_reported_without_terminating_the_task():
    class _TransportFailureBenchmark(_Benchmark):
        def generate_responses(self, model):
            record_endpoint_failure(FailureCategory.MODEL_TRANSPORT)
            return {"examples": [{"output": ""}]}

    benchmark = _TransportFailureBenchmark({}, scored_result={"accuracy": 0.0})

    result = _custom_evaluate(benchmark)

    assert result["results"]["contract_task"]["accuracy"] == 0.0
    assert result["task_outcomes"]["contract_task"]["failure_counts"] == {FailureCategory.MODEL_TRANSPORT: 1}
    validate_result_document(result)


def test_custom_task_with_majority_reasoning_only_content_keeps_score_and_diagnostics():
    class _MalformedResponsesBenchmark(_Benchmark):
        def generate_responses(self, model):
            record_completion_responses(
                Counter({CompletionClassification.REASONING_ONLY: 2, CompletionClassification.FINAL: 1})
            )
            record_endpoint_failure(FailureCategory.MALFORMED_MODEL_RESPONSE, 2)
            return {"examples": [{"output": "reasoning"}, {"output": "reasoning"}, {"output": "answer"}]}

    result = _custom_evaluate(_MalformedResponsesBenchmark({}, scored_result={"accuracy": 1.0 / 3.0}))

    outcome = result["task_outcomes"]["contract_task"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    assert outcome["failure"] is None
    assert outcome["generated_count"] == 3
    assert outcome["scored_count"] == 3
    assert outcome["failure_counts"] == {FailureCategory.MALFORMED_MODEL_RESPONSE: 2}
    assert outcome["completion_response_summary"] == {"reasoning_only": 2, "final": 1}
    assert result["results"]["contract_task"]["accuracy"] == 1.0 / 3.0
    assert result["canonical_results"]["contract_task"]["accuracy"] == 1.0 / 3.0
    validate_result_document(result)


def test_artifacts_are_cleaned_when_grader_infrastructure_fails():
    artifacts = GenerationArtifactManifest.temporary()
    artifact = artifacts.write_jsonl("generated", "generated.jsonl", [{"id": "a"}], expected_count=1)
    benchmark = _Benchmark(
        {"examples": [{"prompt": "x"}], "artifacts": artifacts},
        grading_error=RuntimeError("grader crashed"),
    )

    result = _custom_evaluate(benchmark)

    assert not artifact.path.parent.exists()
    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE


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


def test_requested_sample_serialization_failure_is_typed_and_reported():
    class _SerializationFailureBenchmark(_Benchmark):
        def to_samples(self, generation_result, scored_result):
            raise TypeError("sample is not serializable")

    benchmark = _SerializationFailureBenchmark(
        {"examples": [{"prompt": "x"}]},
        scored_result={"accuracy": 1.0},
    )

    result = _custom_evaluate(benchmark)

    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.SERIALIZATION


def test_scored_task_without_sample_records_fails_sample_validation():
    class _AggregateOnlyBenchmark(_Benchmark):
        def to_sample_group(self, namespace, examples):
            return []

    benchmark = _AggregateOnlyBenchmark({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 1.0})

    result = _custom_evaluate(benchmark)

    assert result["results"] == {}
    assert result["task_outcomes"]["contract_task"]["status"] is TaskStatus.FAILED
    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.SERIALIZATION


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

    result = _custom_evaluate(benchmark)

    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


def test_custom_generation_exception_is_classified():
    benchmark = _Benchmark(None, generation_error=RuntimeError("endpoint stopped"))

    result = _custom_evaluate(benchmark)

    assert result["task_outcomes"]["contract_task"]["failure"]["category"] is FailureCategory.GENERATION
    assert result["task_outcomes"]["contract_task"]["failure"]["exception_type"] == "RuntimeError"


def test_lm_eval_empty_results_are_reported_without_a_score(monkeypatch):
    result = _lm_eval_evaluate(monkeypatch, result={"results": {}})

    assert result["task_outcomes"]["arc_easy"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


def test_lm_eval_result_for_a_different_task_fails_the_requested_task(monkeypatch):
    result = _lm_eval_evaluate(
        monkeypatch,
        result={"results": {"arc_challenge": {"acc,none": 1.0}}},
    )

    assert result["task_outcomes"]["arc_easy"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


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


def test_malformed_completion_keeps_other_samples_and_reports_failure(monkeypatch):
    adapter = object.__new__(LocalChatCompletion)
    adapter.completion_content_policy = CompletionContentPolicy.COMBINE
    adapter.completion_responses = []

    def evaluate_responses(*_args, **_kwargs):
        responses = adapter.parse_generations(
            [
                {"choices": [{"index": 0, "message": {"content": "correct"}}]},
                {"choices": [{"index": 0, "message": {"content": "correct"}}]},
                {"choices": [{"index": 4, "message": {"content": "unparseable"}}]},
            ]
        )
        assert responses == ["correct", "correct", ""]
        return {
            "results": {"arc_easy": {"acc,none": 2 / 3}},
            "n-samples": {"arc_easy": {"original": 3, "effective": 3}},
            "samples": {
                "arc_easy": [
                    {"resps": [[response]], "metrics": ["acc,none"], "acc,none": float(index < 2)}
                    for index, response in enumerate(responses)
                ]
            },
        }

    result = _lm_eval_evaluate(monkeypatch, result=evaluate_responses)

    assert result["results"]["arc_easy"]["acc,none"] == 2 / 3
    assert result["task_outcomes"]["arc_easy"]["failure_counts"] == {FailureCategory.MALFORMED_MODEL_RESPONSE: 1}
    assert result["task_outcomes"]["arc_easy"]["completion_response_summary"] == {"final": 2, "empty": 1}
    assert result["samples"]["arc_easy"][2]["resps"] == [[""]]
    assert result["samples"]["arc_easy"][2]["failure_category"] == FailureCategory.MALFORMED_MODEL_RESPONSE
    validate_result_document(result)


def test_lm_eval_task_with_half_reasoning_only_content_keeps_score_and_diagnostics(monkeypatch):
    def evaluate_responses(*_args, **_kwargs):
        record_completion_responses(
            Counter({CompletionClassification.REASONING_ONLY: 1, CompletionClassification.FINAL: 1})
        )
        return {
            "results": {"arc_easy": {"acc,none": 0.5}},
            "n-samples": {"arc_easy": {"original": 2, "effective": 2}},
        }

    result = _lm_eval_evaluate(monkeypatch, result=evaluate_responses)

    outcome = result["task_outcomes"]["arc_easy"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    assert outcome["failure"] is None
    assert outcome["completion_response_summary"] == {"reasoning_only": 1, "final": 1}
    assert result["results"]["arc_easy"]["acc,none"] == 0.5
    validate_result_document(result)


def test_lm_eval_task_with_half_empty_content_has_no_score(monkeypatch):
    def evaluate_responses(*_args, **_kwargs):
        record_completion_responses(
            Counter({CompletionClassification.EMPTY: 1, CompletionClassification.FINAL: 1})
        )
        return {
            "results": {"arc_easy": {"acc,none": 0.5}},
            "n-samples": {"arc_easy": {"original": 2, "effective": 2}},
        }

    result = _lm_eval_evaluate(monkeypatch, result=evaluate_responses)

    outcome = result["task_outcomes"]["arc_easy"]
    assert outcome["status"] is TaskStatus.FAILED
    assert outcome["failure"]["category"] is FailureCategory.MALFORMED_MODEL_RESPONSE
    assert outcome["completion_response_summary"] == {"empty": 1, "final": 1}
    assert result["results"] == {}
    assert result["canonical_results"] == {}
    validate_result_document(result)


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


def test_lm_eval_outcome_accepts_limited_manifest_coverage():
    manifest = SampleManifest("arc_easy")
    entries = manifest.plan_batch([SampleRequest(source_id="a", ordinal=0), SampleRequest(source_id="b", ordinal=1)])
    manifest.mark_generated(entries, ["x", "y"])

    outcome = lm_eval_task_outcome(
        "arc_easy",
        TaskRoute.LM_EVAL,
        {
            "results": {"arc_easy": {"acc,none": 1.0}},
            "n-samples": {"arc_easy": {"original": 100, "effective": 2}},
        },
        manifest,
    )

    assert outcome.status is TaskStatus.SUCCEEDED
    assert outcome.expected_count == 2
    assert outcome.generated_count == 2
    assert outcome.scored_count == 2


def test_lm_eval_exception_is_classified_instead_of_becoming_empty_success(monkeypatch):
    result = _lm_eval_evaluate(monkeypatch, error=RuntimeError("grader crashed"))

    assert result["task_outcomes"]["arc_easy"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE
    assert result["task_outcomes"]["arc_easy"]["failure"]["exception_type"] == "RuntimeError"


def test_lm_eval_timeout_without_message_is_a_valid_agent_timeout(monkeypatch):
    result = _lm_eval_evaluate(monkeypatch, error=TimeoutError())

    failure = result["task_outcomes"]["arc_easy"]["failure"]
    assert failure["category"] is FailureCategory.AGENT_TIMEOUT
    assert failure["message"]
    assert failure["exception_type"] == "TimeoutError"
    validate_result_document(result)


def test_one_task_infrastructure_failure_does_not_stop_other_tasks(monkeypatch):
    def fake_evaluate(*_args, **kwargs):
        if kwargs["tasks"] == ["broken"]:
            raise TimeoutError("endpoint timeout")
        return {
            "results": {"healthy": {"acc,none": 0.75}},
            "n-samples": {"healthy": {"original": 4, "effective": 4}},
        }

    monkeypatch.setattr("eval.resume.lm_eval_native.resume_simple_evaluate", fake_evaluate)
    result = evaluate(
        lm=_Model(),
        task_manager=_NoCustomTasks(),
        pretrain_task_manager=SimpleNamespace(all_tasks={"broken": object(), "healthy": object()}),
        task_list=["broken", "healthy"],
        task_routes={"broken": LM_EVAL_ROUTE, "healthy": LM_EVAL_ROUTE},
        batch_sizes_list=[1, 1],
        args=_args(),
    )

    assert result["task_outcomes"]["broken"]["failure"]["category"] is FailureCategory.AGENT_TIMEOUT
    assert result["results"]["healthy"]["acc,none"] == 0.75
    validate_result_document(result)


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
def test_aggregate_writer_rejects_results_without_a_valid_task_outcome(result, tmp_path):
    args = Namespace(show_config=False, wandb_args=None, finestore_output_path=str(tmp_path / "archive"))
    tracker = DCEvaluationTracker()

    with pytest.raises(EvaluationRunError):
        handle_evaluation_output(result, args, tracker)
    assert not (tmp_path / "archive").exists()


def test_persisted_result_reader_retains_typed_task_outcomes():
    benchmark = _Benchmark({"examples": [{"prompt": "x"}]}, scored_result={"accuracy": 0.0})

    loaded = EvalResults.model_validate(_custom_evaluate(benchmark))

    assert loaded.task_outcomes["contract_task"].status is TaskStatus.SUCCEEDED
    assert loaded.task_outcomes["contract_task"].metrics == {"accuracy": 0.0}


def test_persisted_result_reader_rejects_legacy_results_without_outcomes():
    with pytest.raises(EvaluationRunError, match="no task outcomes"):
        EvalResults.model_validate({"results": {"task": {"accuracy": 1.0}}})
